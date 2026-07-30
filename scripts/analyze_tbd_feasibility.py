"""Test whether weak targets become separable after long-track accumulation.

This is an oracle *analysis* tool: labels define the trajectories, but never
enter the response maps.  For every truth track it compares the accumulated
image response on the real trajectory with constant spatial translations of
the same trajectory.  A high percentile means a blind track-before-detect
search has a realistic chance of finding that track.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from jinsight_track1.deeppro_adapter import _files, _sequences
from jinsight_track1.postprocess import centroids
from jinsight_track1.submission import parse
from jinsight_track1.tracking import assign_track_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--submission-dir")
    parser.add_argument("--coordinate-order", choices=("xy", "yx"), default="xy")
    parser.add_argument("--resolutions", default="640x512")
    parser.add_argument("--sequences")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--random-paths", type=int, default=256)
    parser.add_argument("--min-track-length", type=int, default=8)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--gate-fraction", type=float, default=0.02)
    parser.add_argument("--max-age", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output")
    return parser.parse_args()


def quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {name: None for name in ("min", "p10", "p25", "p50", "p75", "p90", "max")}
    levels = np.quantile(array, (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1))
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p10", "p25", "p50", "p75", "p90", "max"), levels
        )
    }


def frame_truth(mask_path: Path) -> list[tuple[float, float]]:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    return [(point.x, point.y) for point in centroids(mask, 0.5, 1)]


def robust_standardize(score: np.ndarray) -> np.ndarray:
    """Normalize each frame without allowing a few strong clutter pixels to dominate."""
    sample = score[::4, ::4]
    centre = float(np.median(sample))
    mad = float(np.median(np.abs(sample - centre)))
    scale = max(0.25, 1.4826 * mad)
    return (score - centre) / scale


def response_maps(frame: np.ndarray, temporal_background: np.ndarray):
    image = frame.astype(np.float32)
    centre = cv2.GaussianBlur(image, (0, 0), 0.65)
    surround = cv2.GaussianBlur(image, (0, 0), 2.4)
    spatial_dark = robust_standardize(surround - centre)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = robust_standardize(
        cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    )
    temporal_dark = robust_standardize(temporal_background - image)
    # A conservative fusion: a target should be dark both locally and relative
    # to the sequence background, while either cue may be corrupted by clutter.
    fused = np.minimum(np.maximum(spatial_dark, 0), np.maximum(temporal_dark, 0))
    return {
        "spatial_dark": spatial_dark,
        "blackhat": blackhat,
        "temporal_dark": temporal_dark,
        "fused_min": fused,
    }


def matching_flags(
    truth_frames: list[list[tuple[float, float]]],
    tracked_truth,
    prediction_frames: dict[int, list],
    radius: float,
) -> dict[tuple[int, int], bool]:
    result = {}
    for frame_id, truth_points in tracked_truth.items():
        predicted = np.asarray(
            [(point.x, point.y) for point in prediction_frames.get(frame_id, [])],
            dtype=float,
        ).reshape(-1, 2)
        truth = np.asarray(
            [(point.x, point.y) for point in truth_points], dtype=float
        ).reshape(-1, 2)
        matched = set()
        if len(predicted) and len(truth):
            distances = np.linalg.norm(truth[:, None] - predicted[None, :], axis=2)
            rows, columns = linear_sum_assignment(distances)
            matched = {
                int(row)
                for row, column in zip(rows, columns)
                if distances[row, column] <= radius
            }
        for index, point in enumerate(truth_points):
            result[(frame_id, point.track_id)] = index in matched
    return result


def valid_offsets(
    xy: np.ndarray,
    width: int,
    height: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    min_x, min_y = np.floor(xy.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(xy.max(axis=0)).astype(int)
    low_x, high_x = -min_x, width - 1 - max_x
    low_y, high_y = -min_y, height - 1 - max_y
    if high_x < low_x or high_y < low_y:
        return np.zeros((0, 2), dtype=int)
    offsets = np.column_stack(
        (
            rng.integers(low_x, high_x + 1, size=count * 3),
            rng.integers(low_y, high_y + 1, size=count * 3),
        )
    )
    # Exclude translations too close to the truth path; they are neither useful
    # negatives nor independent trials.
    offsets = offsets[np.linalg.norm(offsets, axis=1) >= 8]
    if len(offsets) < count:
        return offsets
    return offsets[:count]


def trajectory_statistics(entries: list[tuple[int, float, float]]) -> dict[str, float]:
    values = np.asarray(entries, dtype=float)
    time = values[:, 0]
    xy = values[:, 1:]
    design = np.column_stack((np.ones(len(time)), time - time.mean()))
    coefficients = np.linalg.lstsq(design, xy, rcond=None)[0]
    fitted = design @ coefficients
    residual = np.linalg.norm(xy - fitted, axis=1)
    if len(values) >= 2:
        delta_t = np.diff(time)
        velocity = np.diff(xy, axis=0) / delta_t[:, None]
        speeds = np.linalg.norm(velocity, axis=1)
    else:
        velocity = np.zeros((0, 2))
        speeds = np.zeros(0)
    if len(velocity) >= 2:
        acceleration = np.diff(velocity, axis=0)
        accelerations = np.linalg.norm(acceleration, axis=1)
    else:
        accelerations = np.zeros(0)
    return {
        "length": int(len(values)),
        "span": int(time[-1] - time[0] + 1),
        "start_frame": int(time[0]),
        "end_frame": int(time[-1]),
        "start_x": float(xy[0, 0]),
        "start_y": float(xy[0, 1]),
        "end_x": float(xy[-1, 0]),
        "end_y": float(xy[-1, 1]),
        "fit_mid_x": float(coefficients[0, 0]),
        "fit_mid_y": float(coefficients[0, 1]),
        "fit_vx": float(coefficients[1, 0]),
        "fit_vy": float(coefficients[1, 1]),
        "linear_rmse": float(np.sqrt(np.mean(residual**2))),
        "linear_p90": float(np.quantile(residual, 0.9)),
        "speed_median": float(np.median(speeds)) if len(speeds) else 0.0,
        "speed_p90": float(np.quantile(speeds, 0.9)) if len(speeds) else 0.0,
        "acceleration_median": (
            float(np.median(accelerations)) if len(accelerations) else 0.0
        ),
        "acceleration_p90": (
            float(np.quantile(accelerations, 0.9)) if len(accelerations) else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    selected_resolutions = set(args.resolutions.split(",")) if args.resolutions else None
    selected_sequences = set(args.sequences.split(",")) if args.sequences else None
    rng = np.random.default_rng(args.seed)
    all_rows = []
    sequence_rows = []

    sequences = _sequences(Path(args.root))
    if selected_sequences:
        sequences = [sequence for sequence in sequences if sequence.name in selected_sequences]
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]

    for sequence in sequences:
        images = _files(sequence / "img")
        masks = _files(sequence / "mask")
        stems = sorted(set(images) & set(masks))
        if not stems:
            continue
        width, height = Image.open(images[stems[0]]).size
        resolution = f"{width}x{height}"
        if selected_resolutions and resolution not in selected_resolutions:
            continue

        truths = [frame_truth(masks[stem]) for stem in stems]
        tracked = assign_track_ids(
            truths,
            (height, width),
            gate_fraction=args.gate_fraction,
            max_age=args.max_age,
        )
        track_entries: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        frame_entries: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        for frame_id, points in tracked.items():
            for point in points:
                entry = (frame_id, point.x, point.y)
                track_entries[point.track_id].append(entry)
                frame_entries[frame_id].append((point.track_id, point.x, point.y))

        prediction_frames = {}
        if args.submission_dir:
            prediction_path = Path(args.submission_dir) / f"{sequence.name}.txt"
            if prediction_path.is_file():
                prediction_frames = parse(
                    prediction_path.read_text(encoding="ascii"),
                    sequence.name,
                    args.coordinate_order,
                ).frames
        matched = matching_flags(truths, tracked, prediction_frames, args.radius)

        candidates = {}
        for track_id, entries in track_entries.items():
            if len(entries) < args.min_track_length:
                continue
            xy = np.asarray([(x, y) for _, x, y in entries], dtype=float)
            offsets = valid_offsets(
                xy, width, height, args.random_paths, rng
            )
            if not len(offsets):
                continue
            candidates[track_id] = {
                "entries": entries,
                "offsets": offsets,
                "sum": defaultdict(lambda: np.zeros(len(offsets) + 1, dtype=float)),
                "count": np.zeros(len(offsets) + 1, dtype=int),
            }

        # A sparse sequence median is cheap and deliberately independent of labels.
        sample_indices = np.linspace(0, len(stems) - 1, min(41, len(stems))).astype(int)
        background_stack = [
            np.asarray(Image.open(images[stems[index]]).convert("L"), dtype=np.uint8)
            for index in sample_indices
        ]
        temporal_background = np.median(
            np.stack(background_stack), axis=0
        ).astype(np.float32)

        for frame_id, stem in enumerate(stems, 1):
            relevant = [
                item for item in frame_entries.get(frame_id, []) if item[0] in candidates
            ]
            if not relevant:
                continue
            frame = np.asarray(Image.open(images[stem]).convert("L"), dtype=np.uint8)
            maps = response_maps(frame, temporal_background)
            for track_id, x, y in relevant:
                candidate = candidates[track_id]
                offsets = candidate["offsets"]
                query = np.vstack(([0, 0], offsets)) + np.rint([x, y]).astype(int)
                query[:, 0] = np.clip(query[:, 0], 0, width - 1)
                query[:, 1] = np.clip(query[:, 1], 0, height - 1)
                for name, score in maps.items():
                    candidate["sum"][name] += score[query[:, 1], query[:, 0]]
                candidate["count"] += 1

        sequence_track_rows = []
        for track_id, candidate in candidates.items():
            row = {
                "sequence": sequence.name,
                "resolution": resolution,
                "track_id": int(track_id),
                **trajectory_statistics(candidate["entries"]),
            }
            flags = [
                matched.get((frame_id, track_id), False)
                for frame_id, _, _ in candidate["entries"]
            ]
            row["matched_points"] = int(sum(flags))
            row["unseen"] = not any(flags)
            count = np.maximum(candidate["count"], 1)
            for name, sums in candidate["sum"].items():
                means = sums / count
                truth_score = float(means[0])
                random_scores = means[1:]
                row[f"{name}_mean"] = truth_score
                row[f"{name}_percentile"] = float(
                    (np.sum(random_scores < truth_score) + 0.5 * np.sum(random_scores == truth_score))
                    / max(1, len(random_scores))
                )
                row[f"{name}_random_p99"] = float(np.quantile(random_scores, 0.99))
                row[f"{name}_margin_p99"] = truth_score - row[f"{name}_random_p99"]
            sequence_track_rows.append(row)
            all_rows.append(row)

        sequence_rows.append(
            {
                "sequence": sequence.name,
                "resolution": resolution,
                "frames": len(stems),
                "truth_points": sum(map(len, truths)),
                "tracks": len(track_entries),
                "analyzed_tracks": len(sequence_track_rows),
                "unseen_analyzed_tracks": sum(row["unseen"] for row in sequence_track_rows),
            }
        )
        print(
            f"tbd-feasibility {sequence.name} {resolution}: "
            f"tracks={len(sequence_track_rows)}",
            flush=True,
        )

    methods = ("spatial_dark", "blackhat", "temporal_dark", "fused_min")
    groups = {}
    for group_name, predicate in (
        ("all", lambda row: True),
        ("unseen", lambda row: row["unseen"]),
        ("partly_seen", lambda row: not row["unseen"]),
    ):
        rows = [row for row in all_rows if predicate(row)]
        group = {"tracks": len(rows), "truth_points": sum(row["length"] for row in rows)}
        for field in (
            "length",
            "span",
            "linear_rmse",
            "linear_p90",
            "speed_median",
            "speed_p90",
            "acceleration_median",
            "acceleration_p90",
        ):
            group[field] = quantiles([row[field] for row in rows])
        for method in methods:
            percentiles = [row[f"{method}_percentile"] for row in rows]
            margins = [row[f"{method}_margin_p99"] for row in rows]
            group[method] = {
                "percentile": quantiles(percentiles),
                "margin_p99": quantiles(margins),
                "fraction_ge_95pct": (
                    sum(value >= 0.95 for value in percentiles) / len(percentiles)
                    if percentiles
                    else None
                ),
                "fraction_ge_99pct": (
                    sum(value >= 0.99 for value in percentiles) / len(percentiles)
                    if percentiles
                    else None
                ),
            }
        groups[group_name] = group

    report = {
        "settings": vars(args),
        "sequences": sequence_rows,
        "groups": groups,
        "tracks": all_rows,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
