"""Evaluate a blind long-horizon track-before-detect dynamic program.

The detector groups contiguous sequence shards (for example ``000201_01`` to
``000201_04``), builds a label-free temporal background, and follows weak
point-like responses through block-averaged score maps.  Labels are used only
after candidate extraction to report point F1 and candidate purity.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from jinsight_track1.deeppro_adapter import _files, _sequences
from jinsight_track1.postprocess import centroids
from jinsight_track1.tracking import assign_track_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--resolutions", default="640x512")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--transition-radius", type=int, default=1)
    parser.add_argument("--motion-penalty", type=float, default=0.02)
    parser.add_argument("--method", choices=("spatial", "temporal", "fused"), default="fused")
    parser.add_argument("--registration", choices=("none", "affine"), default="none")
    parser.add_argument(
        "--temporal-center",
        choices=("none", "mean", "median"),
        default="none",
        help="remove each registered pixel's persistent response across blocks",
    )
    parser.add_argument("--top-candidates", type=int, default=20)
    parser.add_argument("--endpoint-pool", type=int, default=2000)
    parser.add_argument("--endpoint-separation", type=float, default=6.0)
    parser.add_argument("--path-separation", type=float, default=4.0)
    parser.add_argument("--fit-penalty", type=float, default=0.5)
    parser.add_argument("--max-fit-rmse", type=float, default=4.0)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--output")
    return parser.parse_args()


def robust_standardize(score: np.ndarray) -> np.ndarray:
    sample = score[::4, ::4]
    centre = float(np.median(sample))
    mad = float(np.median(np.abs(sample - centre)))
    return (score - centre) / max(0.25, 1.4826 * mad)


def score_map(frame: np.ndarray, background: np.ndarray, method: str) -> np.ndarray:
    image = frame.astype(np.float32)
    centre = cv2.GaussianBlur(image, (0, 0), 0.65)
    surround = cv2.GaussianBlur(image, (0, 0), 2.4)
    spatial = robust_standardize(surround - centre)
    temporal = robust_standardize(background - image)
    if method == "spatial":
        result = spatial
    elif method == "temporal":
        result = temporal
    else:
        result = np.minimum(np.maximum(spatial, 0), np.maximum(temporal, 0))
    return cv2.GaussianBlur(np.clip(result, -2, 8), (3, 3), 0).astype(np.float32)


def block_images(rows, block_size: int):
    images = []
    times = []
    accumulated = None
    count = 0
    block_times = []
    for time, row in enumerate(rows):
        frame = np.asarray(Image.open(row["image"]).convert("L"), dtype=np.float32)
        if accumulated is None:
            accumulated = np.zeros_like(frame)
        accumulated += frame
        count += 1
        block_times.append(time)
        if count == block_size or time == len(rows) - 1:
            images.append(np.clip(accumulated / count, 0, 255).astype(np.uint8))
            times.append(float(np.mean(block_times)))
            accumulated.fill(0)
            count = 0
            block_times = []
    return images, np.asarray(times)


def estimate_block_transforms(images: list[np.ndarray]) -> np.ndarray:
    """Estimate cumulative affine maps from block zero to every block."""
    cumulative = [np.eye(3, dtype=np.float64)]
    for previous, current in zip(images, images[1:]):
        corners = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=2000,
            qualityLevel=0.01,
            minDistance=6,
            blockSize=7,
        )
        matrix = None
        if corners is not None and len(corners) >= 12:
            tracked, status, error = cv2.calcOpticalFlowPyrLK(
                previous,
                current,
                corners,
                None,
                winSize=(31, 31),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    40,
                    0.01,
                ),
            )
            keep = status.ravel().astype(bool)
            if error is not None:
                keep &= error.ravel() < np.quantile(error.ravel(), 0.9)
            if keep.sum() >= 8:
                matrix, _ = cv2.estimateAffine2D(
                    corners[keep, 0],
                    tracked[keep, 0],
                    method=cv2.RANSAC,
                    ransacReprojThreshold=1.0,
                    maxIters=3000,
                    confidence=0.995,
                    refineIters=20,
                )
        step = np.eye(3, dtype=np.float64)
        if matrix is not None and np.isfinite(matrix).all():
            step[:2] = matrix
        cumulative.append(step @ cumulative[-1])
    return np.stack(cumulative)


def warp_to_reference(
    image: np.ndarray, transform: np.ndarray, width: int, height: int
) -> np.ndarray:
    return cv2.warpAffine(
        image,
        np.linalg.inv(transform)[:2],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def registered_frame_block_maps(
    rows,
    block_size: int,
    averaged_images: list[np.ndarray],
    block_times: np.ndarray,
    method: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Preserve point energy by scoring individual frames before block averaging."""
    transforms = estimate_block_transforms(averaged_images)
    height, width = averaged_images[0].shape
    sample_indices = np.linspace(0, len(rows) - 1, min(101, len(rows))).astype(int)
    registered_samples = []
    for index in sample_indices:
        frame = np.asarray(Image.open(rows[index]["image"]).convert("L"), dtype=np.uint8)
        transform = interpolate_transform(index, block_times, transforms)
        registered_samples.append(warp_to_reference(frame, transform, width, height))
    background = np.median(np.stack(registered_samples), axis=0).astype(np.float32)

    support = np.full((height, width), 255, dtype=np.uint8)
    block_maps = []
    accumulated = np.zeros((height, width), dtype=np.float32)
    count = 0
    for time, row in enumerate(rows):
        frame = np.asarray(Image.open(row["image"]).convert("L"), dtype=np.uint8)
        transform = interpolate_transform(time, block_times, transforms)
        registered = warp_to_reference(frame, transform, width, height)
        valid = warp_to_reference(support, transform, width, height) > 127
        response = score_map(registered, background, method)
        response[~valid] = -2
        accumulated += response
        count += 1
        if count == block_size or time == len(rows) - 1:
            block_score = accumulated / count
            block_score[:4] = block_score[-4:] = -2
            block_score[:, :4] = block_score[:, -4:] = -2
            block_maps.append(block_score.copy())
            accumulated.fill(0)
            count = 0
    return block_maps, transforms


def interpolate_transform(
    time: float, block_times: np.ndarray, transforms: np.ndarray
) -> np.ndarray:
    flat = np.asarray(
        [
            np.interp(time, block_times, transforms[:, row, column])
            for row in range(2)
            for column in range(3)
        ]
    )
    result = np.eye(3, dtype=float)
    result[:2] = flat.reshape(2, 3)
    return result


def transform_point(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform @ np.asarray([point[0], point[1], 1.0]))[:2]


def group_key(sequence: Path) -> str:
    return re.sub(r"_\d+$", "", sequence.name)


def group_sequences(root: Path, resolutions: set[str] | None):
    groups = defaultdict(list)
    for sequence in _sequences(root):
        images = _files(sequence / "img")
        if not images:
            continue
        first = Image.open(next(iter(images.values())))
        resolution = f"{first.width}x{first.height}"
        if resolutions and resolution not in resolutions:
            continue
        groups[(group_key(sequence), resolution)].append(sequence)
    for key in groups:
        groups[key].sort(key=lambda path: path.name)
    return groups


def load_group(sequences: list[Path]):
    rows = []
    for sequence in sequences:
        images = _files(sequence / "img")
        masks = _files(sequence / "mask")
        for local_frame, stem in enumerate(sorted(images), 1):
            rows.append(
                {
                    "sequence": sequence.name,
                    "local_frame": local_frame,
                    "stem": stem,
                    "image": images[stem],
                    "mask": masks.get(stem),
                }
            )
    rows.sort(key=lambda row: (int(row["stem"]) if row["stem"].isdigit() else row["stem"]))
    return rows


def shift_previous(previous: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Return previous[y-dy, x-dx] at each current (x, y)."""
    height, width = previous.shape
    shifted = np.full_like(previous, -np.inf)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    target_x0, target_x1 = source_x0 + dx, source_x1 + dx
    target_y0, target_y1 = source_y0 + dy, source_y1 + dy
    shifted[target_y0:target_y1, target_x0:target_x1] = previous[
        source_y0:source_y1, source_x0:source_x1
    ]
    return shifted


def dynamic_programming(
    block_maps: list[np.ndarray], transition_radius: int, motion_penalty: float
):
    offsets = [
        (dx, dy)
        for dy in range(-transition_radius, transition_radius + 1)
        for dx in range(-transition_radius, transition_radius + 1)
    ]
    previous = block_maps[0].copy()
    pointers = []
    for current_score in block_maps[1:]:
        best = np.full_like(previous, -np.inf)
        pointer = np.zeros(previous.shape, dtype=np.uint8)
        for index, (dx, dy) in enumerate(offsets):
            candidate = shift_previous(previous, dx, dy)
            candidate -= motion_penalty * (dx * dx + dy * dy)
            improve = candidate > best
            best[improve] = candidate[improve]
            pointer[improve] = index
        previous = best + current_score
        pointers.append(pointer)
    return previous, pointers, offsets


def endpoint_candidates(score: np.ndarray, count: int, separation: float):
    work = score.copy()
    found = []
    radius = max(1, int(np.ceil(separation)))
    for _ in range(count * 4):
        flat = int(np.argmax(work))
        value = float(work.flat[flat])
        if not np.isfinite(value):
            break
        y, x = np.unravel_index(flat, work.shape)
        found.append((x, y, value))
        cv2.circle(work, (int(x), int(y)), radius, -np.inf, thickness=-1)
        if len(found) >= count:
            break
    return found


def backtrack(endpoint, pointers, offsets):
    x, y, _ = endpoint
    reverse = [(float(x), float(y))]
    for pointer in reversed(pointers):
        index = int(pointer[int(y), int(x)])
        dx, dy = offsets[index]
        x, y = x - dx, y - dy
        reverse.append((float(x), float(y)))
    return np.asarray(reverse[::-1], dtype=float)


def fit_line(path: np.ndarray, block_times: np.ndarray):
    design = np.column_stack((np.ones(len(block_times)), block_times))
    keep = np.ones(len(block_times), dtype=bool)
    coefficients = np.linalg.lstsq(design, path, rcond=None)[0]
    for _ in range(3):
        fitted = design @ coefficients
        residual = np.linalg.norm(path - fitted, axis=1)
        threshold = max(1.5, float(np.quantile(residual, 0.8)))
        new_keep = residual <= threshold
        if new_keep.sum() < max(4, len(block_times) // 3):
            break
        keep = new_keep
        coefficients = np.linalg.lstsq(design[keep], path[keep], rcond=None)[0]
    fitted = design @ coefficients
    return coefficients, float(np.sqrt(np.mean(np.sum((path - fitted) ** 2, axis=1))))


def line_score(coefficients: np.ndarray, block_maps, block_times: np.ndarray) -> float:
    total = 0.0
    used = 0
    height, width = block_maps[0].shape
    for score, time in zip(block_maps, block_times):
        x, y = np.rint(coefficients[0] + coefficients[1] * time).astype(int)
        if 0 <= x < width and 0 <= y < height:
            total += float(score[y, x])
            used += 1
    return total / max(1, used)


def path_distance(first, second, times):
    first_xy = first[0][None, :] + times[:, None] * first[1][None, :]
    second_xy = second[0][None, :] + times[:, None] * second[1][None, :]
    return float(np.mean(np.linalg.norm(first_xy - second_xy, axis=1)))


def truth_points(mask_path: Path | None):
    if mask_path is None:
        return []
    mask = np.asarray(Image.open(mask_path).convert("L"))
    return [(point.x, point.y) for point in centroids(mask, 0.5, 1)]


def evaluate_candidates(
    candidates,
    rows,
    radius: float,
    block_times: np.ndarray,
    transforms: np.ndarray | None,
):
    results = []
    truth_by_frame = [
        np.asarray(truth_points(row["mask"]), dtype=float).reshape(-1, 2)
        for row in rows
    ]
    for limit in range(1, len(candidates) + 1):
        tp = fp = fn = 0
        for time, truth in enumerate(truth_by_frame):
            predicted = np.asarray(
                [
                    (
                        transform_point(
                            candidate["intercept"] + candidate["velocity"] * time,
                            interpolate_transform(time, block_times, transforms),
                        )
                        if transforms is not None
                        else candidate["intercept"] + candidate["velocity"] * time
                    )
                    for candidate in candidates[:limit]
                ],
                dtype=float,
            ).reshape(-1, 2)
            matches = 0
            if len(predicted) and len(truth):
                distances = np.linalg.norm(predicted[:, None] - truth[None, :], axis=2)
                row_ids, column_ids = linear_sum_assignment(distances)
                matches = sum(
                    distances[row_id, column_id] <= radius
                    for row_id, column_id in zip(row_ids, column_ids)
                )
            tp += matches
            fp += len(predicted) - matches
            fn += len(truth) - matches
        results.append(
            {
                "candidates": limit,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            }
        )
    return results


def oracle_diagnostics(
    rows,
    block_maps,
    block_times,
    final_score,
    pointers,
    offsets,
    transforms: np.ndarray | None,
):
    truth_frames = [truth_points(row["mask"]) for row in rows]
    height, width = block_maps[0].shape
    tracked = assign_track_ids(truth_frames, (height, width), gate_fraction=0.02, max_age=5)
    entries = defaultdict(list)
    for frame_id, points in tracked.items():
        for point in points:
            entries[point.track_id].append((frame_id - 1, point.x, point.y))
    diagnostics = []
    for track_id, track in entries.items():
        if len(track) < 8:
            continue
        values = np.asarray(track, dtype=float)
        times = values[:, 0]
        if transforms is not None:
            registered_xy = []
            for time, x, y in values:
                transform = interpolate_transform(time, block_times, transforms)
                registered_xy.append(transform_point(np.asarray([x, y]), np.linalg.inv(transform)))
            values[:, 1:] = np.asarray(registered_xy)
        design = np.column_stack((np.ones(len(times)), times))
        coefficients = np.linalg.lstsq(design, values[:, 1:], rcond=None)[0]
        fitted = design @ coefficients
        rmse = float(np.sqrt(np.mean(np.sum((values[:, 1:] - fitted) ** 2, axis=1))))
        end_xy = np.rint(coefficients[0] + coefficients[1] * block_times[-1]).astype(int)
        if not (0 <= end_xy[0] < width and 0 <= end_xy[1] < height):
            continue
        endpoint_value = float(final_score[end_xy[1], end_xy[0]])
        endpoint_percentile = float(np.mean(final_score <= endpoint_value))
        recovered_path = backtrack(
            (int(end_xy[0]), int(end_xy[1]), endpoint_value), pointers, offsets
        )
        recovered_coefficients, recovered_rmse = fit_line(recovered_path, block_times)
        diagnostics.append(
            {
                "track_id": int(track_id),
                "length": len(track),
                "start_frame": int(times[0]),
                "end_frame": int(times[-1]),
                "intercept": coefficients[0].tolist(),
                "velocity": coefficients[1].tolist(),
                "truth_linear_rmse": rmse,
                "line_score": line_score(coefficients, block_maps, block_times),
                "endpoint_score": endpoint_value / len(block_maps),
                "endpoint_percentile": endpoint_percentile,
                "backtrack_fit_rmse": recovered_rmse,
                "backtrack_intercept": recovered_coefficients[0].tolist(),
                "backtrack_velocity": recovered_coefficients[1].tolist(),
            }
        )
    return diagnostics


def main() -> None:
    args = parse_args()
    resolutions = set(args.resolutions.split(",")) if args.resolutions else None
    report_groups = []
    for (name, resolution), sequences in group_sequences(
        Path(args.root), resolutions
    ).items():
        rows = load_group(sequences)
        width, height = Image.open(rows[0]["image"]).size
        if args.registration == "affine":
            averages, block_times_array = block_images(rows, args.block_size)
            block_maps, transforms = registered_frame_block_maps(
                rows,
                args.block_size,
                averages,
                block_times_array,
                args.method,
            )
        else:
            transforms = None
            sample_indices = np.linspace(
                0, len(rows) - 1, min(101, len(rows))
            ).astype(int)
            background = np.median(
                np.stack(
                    [
                        np.asarray(Image.open(rows[index]["image"]).convert("L"))
                        for index in sample_indices
                    ]
                ),
                axis=0,
            ).astype(np.float32)

            block_maps = []
            block_times = []
            accumulated = np.zeros((height, width), dtype=np.float32)
            frames_in_block = 0
            times_in_block = []
            for time, row in enumerate(rows):
                frame = np.asarray(
                    Image.open(row["image"]).convert("L"), dtype=np.uint8
                )
                accumulated += score_map(frame, background, args.method)
                frames_in_block += 1
                times_in_block.append(time)
                if frames_in_block == args.block_size or time == len(rows) - 1:
                    block_maps.append(accumulated / frames_in_block)
                    block_times.append(float(np.mean(times_in_block)))
                    accumulated.fill(0)
                    frames_in_block = 0
                    times_in_block = []
            block_times_array = np.asarray(block_times)

        if args.temporal_center != "none":
            stack = np.stack(block_maps)
            if args.temporal_center == "median":
                persistent = np.median(stack, axis=0)
            else:
                persistent = np.mean(stack, axis=0)
            block_maps = [
                np.clip(score - persistent, -2, 8).astype(np.float32)
                for score in block_maps
            ]

        final_score, pointers, offsets = dynamic_programming(
            block_maps, args.transition_radius, args.motion_penalty
        )
        endpoints = endpoint_candidates(
            final_score, args.endpoint_pool, args.endpoint_separation
        )
        raw_candidates = []
        for endpoint in endpoints:
            path = backtrack(endpoint, pointers, offsets)
            coefficients, fit_rmse = fit_line(path, block_times_array)
            if fit_rmse > args.max_fit_rmse:
                continue
            candidate_line_score = line_score(
                coefficients, block_maps, block_times_array
            )
            candidate = {
                "intercept": coefficients[0],
                "velocity": coefficients[1],
                "dp_score": endpoint[2] / len(block_maps),
                "fit_rmse": fit_rmse,
                "line_score": candidate_line_score,
                "rank_score": candidate_line_score - args.fit_penalty * fit_rmse,
                "path": path,
            }
            raw_candidates.append(candidate)
        raw_candidates.sort(key=lambda item: item["rank_score"], reverse=True)
        candidates = []
        for candidate in raw_candidates:
            if any(
                path_distance(
                    (candidate["intercept"], candidate["velocity"]),
                    (kept["intercept"], kept["velocity"]),
                    block_times_array,
                )
                < args.path_separation
                for kept in candidates
            ):
                continue
            candidates.append(candidate)
            if len(candidates) >= args.top_candidates:
                break

        evaluation = evaluate_candidates(
            candidates,
            rows,
            args.radius,
            block_times_array,
            transforms,
        )
        serializable_candidates = [
            {
                "rank": index,
                "intercept": candidate["intercept"].tolist(),
                "velocity": candidate["velocity"].tolist(),
                "dp_score": candidate["dp_score"],
                "fit_rmse": candidate["fit_rmse"],
                "line_score": candidate["line_score"],
                "rank_score": candidate["rank_score"],
            }
            for index, candidate in enumerate(candidates, 1)
        ]
        group_report = {
            "group": name,
            "resolution": resolution,
            "sequences": [sequence.name for sequence in sequences],
            "frames": len(rows),
            "truth_points": sum(len(truth_points(row["mask"])) for row in rows),
            "blocks": len(block_maps),
            "candidates": serializable_candidates,
            "oracle": oracle_diagnostics(
                rows,
                block_maps,
                block_times_array,
                final_score,
                pointers,
                offsets,
                transforms,
            ),
            "evaluation": evaluation,
        }
        report_groups.append(group_report)
        best = max(evaluation, key=lambda item: item["f1"]) if evaluation else None
        print(
            f"tbd-dp {name} frames={len(rows)} candidates={len(candidates)} "
            f"best={best}",
            flush=True,
        )

    report = {"settings": vars(args), "groups": report_groups}
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
