"""Evaluate gap filling and endpoint extension from high-precision point tracks."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from jinsight_track1.deeppro_adapter import _files, _load_sequence_images, _sequences
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.postprocess import centroids
from jinsight_track1.submission import parse
from jinsight_track1.tracking import assign_track_ids


def metric_summary(values):
    tp, fp, fn = (int(values[key]) for key in ("tp", "fp", "fn"))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def endpoint_fit(points, side, window):
    sample = points[:window] if side == "start" else points[-window:]
    times = np.asarray([point.frame_id for point in sample], dtype=float)
    values = np.asarray([(point.x, point.y) for point in sample], dtype=float)
    design = np.column_stack((np.ones(len(times)), times))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = design @ coefficients
    residual = float(np.sqrt(np.mean(np.sum((values - fitted) ** 2, axis=1))))
    return coefficients, residual


def candidate_tracks(point_frames, shape, gate_fraction, max_age, fit_window):
    tracked = assign_track_ids(
        point_frames, shape, gate_fraction=gate_fraction, max_age=max_age
    )
    histories = defaultdict(list)
    for points in tracked.values():
        for point in points:
            histories[point.track_id].append(point)
    candidates = defaultdict(list)
    frame_count = len(point_frames)
    height, width = shape
    for history in histories.values():
        history.sort(key=lambda point: point.frame_id)
        if len(history) < 3:
            continue
        hits = len(history)
        for left, right in zip(history, history[1:]):
            gap = right.frame_id - left.frame_id - 1
            if gap < 1:
                continue
            for frame_id in range(left.frame_id + 1, right.frame_id):
                fraction = (frame_id - left.frame_id) / (
                    right.frame_id - left.frame_id
                )
                x = left.x + fraction * (right.x - left.x)
                y = left.y + fraction * (right.y - left.y)
                candidates[frame_id].append(
                    {
                        "point": (float(x), float(y)),
                        "hits": hits,
                        "kind": "interpolate",
                        "gap": gap,
                        "distance": 0,
                        "residual": 0.0,
                    }
                )
        if len(history) < fit_window:
            continue
        for side in ("start", "end"):
            coefficients, residual = endpoint_fit(history, side, fit_window)
            endpoint = history[0].frame_id if side == "start" else history[-1].frame_id
            frame_range = (
                range(endpoint - 1, 0, -1)
                if side == "start"
                else range(endpoint + 1, frame_count + 1)
            )
            for frame_id in frame_range:
                x, y = coefficients[0] + coefficients[1] * frame_id
                if not (0 <= x < width and 0 <= y < height):
                    break
                candidates[frame_id].append(
                    {
                        "point": (float(x), float(y)),
                        "hits": hits,
                        "kind": "extrapolate",
                        "gap": 0,
                        "distance": abs(frame_id - endpoint),
                        "residual": residual,
                    }
                )
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("val_root")
    parser.add_argument("submission")
    parser.add_argument("output")
    parser.add_argument("--resolutions", default="256x256")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--gate-fraction", type=float, default=0.02)
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--fit-window", type=int, default=10)
    parser.add_argument("--min-hits", default="3,5,10,20")
    parser.add_argument("--max-gaps", default="0,3,10,30,1000000")
    parser.add_argument("--max-extensions", default="0,3,10,30,1000000")
    parser.add_argument("--max-residuals", default="0.5,1,2")
    args = parser.parse_args()

    resolutions = set(args.resolutions.split(",")) if args.resolutions else None
    baseline_total = defaultdict(int)
    records = []
    sequence_rows = []
    for sequence in _sequences(Path(args.val_root)):
        stems, frames = _load_sequence_images(sequence)
        if not len(stems):
            continue
        height, width = frames.shape[1:]
        resolution = f"{width}x{height}"
        if resolutions and resolution not in resolutions:
            continue
        prediction = parse(
            (Path(args.submission) / f"{sequence.name}.txt").read_text(
                encoding="ascii"
            ),
            sequence.name,
        )
        point_frames = [
            [(point.x, point.y) for point in prediction.frames[index]]
            for index in range(1, len(stems) + 1)
        ]
        candidates = candidate_tracks(
            point_frames,
            (height, width),
            args.gate_fraction,
            args.max_age,
            args.fit_window,
        )
        masks = _files(sequence / "mask")
        sequence_candidates = 0
        for frame_id, stem in enumerate(stems, 1):
            truth_mask = np.asarray(Image.open(masks[stem]).convert("L"))
            truth = [(point.x, point.y) for point in centroids(truth_mask)]
            baseline = point_frames[frame_id - 1]
            base_metrics = point_metrics(baseline, truth, args.radius)
            for key in ("tp", "fp", "fn"):
                baseline_total[key] += base_metrics[key]
            if candidates[frame_id]:
                sequence_candidates += len(candidates[frame_id])
                records.append(
                    (baseline, truth, base_metrics, candidates[frame_id])
                )
        sequence_rows.append(
            {
                "sequence": sequence.name,
                "resolution": resolution,
                "frames": len(stems),
                "candidate_points": sequence_candidates,
            }
        )
        print(
            f"track-completion sequence={sequence.name} "
            f"candidates={sequence_candidates}",
            flush=True,
        )

    min_hits_grid = [int(value) for value in args.min_hits.split(",")]
    max_gap_grid = [int(value) for value in args.max_gaps.split(",")]
    max_extension_grid = [int(value) for value in args.max_extensions.split(",")]
    max_residual_grid = [float(value) for value in args.max_residuals.split(",")]
    sweep = []
    for min_hits in min_hits_grid:
        for max_gap in max_gap_grid:
            for max_extension in max_extension_grid:
                for max_residual in max_residual_grid:
                    total = defaultdict(int, baseline_total)
                    added = 0
                    for baseline, truth, base_metrics, frame_candidates in records:
                        selected = []
                        ordered = sorted(
                            frame_candidates,
                            key=lambda item: (
                                item["kind"] != "interpolate",
                                item["gap"],
                                item["distance"],
                                item["residual"],
                            ),
                        )
                        for candidate in ordered:
                            if candidate["hits"] < min_hits:
                                continue
                            if (
                                candidate["kind"] == "interpolate"
                                and candidate["gap"] > max_gap
                            ):
                                continue
                            if (
                                candidate["kind"] == "extrapolate"
                                and (
                                    candidate["distance"] > max_extension
                                    or candidate["residual"] > max_residual
                                )
                            ):
                                continue
                            x, y = candidate["point"]
                            if any(
                                (x - old_x) ** 2 + (y - old_y) ** 2
                                <= args.radius**2
                                for old_x, old_y in baseline + selected
                            ):
                                continue
                            selected.append((x, y))
                        if not selected:
                            continue
                        combined = point_metrics(
                            baseline + selected, truth, args.radius
                        )
                        for key in ("tp", "fp", "fn"):
                            total[key] += combined[key] - base_metrics[key]
                        added += len(selected)
                    row = {
                        "min_hits": min_hits,
                        "max_gap": max_gap,
                        "max_extension": max_extension,
                        "max_residual": max_residual,
                        "added": added,
                        **metric_summary(total),
                    }
                    row["delta_tp"] = row["tp"] - baseline_total["tp"]
                    row["delta_fp"] = row["fp"] - baseline_total["fp"]
                    sweep.append(row)
    sweep.sort(key=lambda row: (row["f1"], row["precision"]), reverse=True)
    report = {
        "settings": vars(args),
        "baseline": metric_summary(baseline_total),
        "candidate_frames": len(records),
        "sequences": sequence_rows,
        "best": sweep[:50],
        "metric_status": "local point-matching proxy; not official scorer",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": report["baseline"], "best": sweep[0]}, indent=2))


if __name__ == "__main__":
    main()
