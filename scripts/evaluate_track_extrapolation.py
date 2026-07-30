"""Evaluate image-gated one-frame extrapolation of stable detection tracks."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from jinsight_track1.deeppro_adapter import _files, _load_sequence_images, _sequences
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.postprocess import centroids
from jinsight_track1.submission import parse
from jinsight_track1.tracking import assign_track_ids


def evidence(frames: np.ndarray, frame_index: int, x: float, y: float):
    height, width = frames.shape[1:]
    ix, iy = int(round(x)), int(round(y))
    if not (5 <= ix < width - 5 and 5 <= iy < height - 5):
        return 0.0, 0.0
    value = frames[frame_index].astype(np.float32)
    local = value[iy - 5 : iy + 6, ix - 5 : ix + 6]
    centre = cv2.GaussianBlur(local, (3, 3), 0)[5, 5]
    ring = local.copy().reshape(-1)
    ring = np.delete(
        ring,
        [
            row * 11 + col
            for row in range(4, 7)
            for col in range(4, 7)
        ],
    )
    spatial = abs(float(centre) - float(np.median(ring))) / max(
        float(np.std(ring)), 2.0
    )
    start, end = max(0, frame_index - 5), min(len(frames), frame_index + 6)
    temporal_values = []
    for index in range(start, end):
        if index == frame_index:
            continue
        patch = frames[index, iy - 1 : iy + 2, ix - 1 : ix + 2]
        temporal_values.append(float(np.mean(patch)))
    current = float(np.mean(frames[frame_index, iy - 1 : iy + 2, ix - 1 : ix + 2]))
    temporal = abs(current - float(np.median(temporal_values))) / max(
        float(np.std(temporal_values)), 2.0
    )
    return spatial, temporal


def summary(values):
    tp, fp, fn = (int(values[key]) for key in ("tp", "fp", "fn"))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("val_root")
    parser.add_argument("submission")
    parser.add_argument("output")
    parser.add_argument("--resolutions", default="256x256")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--gate-fraction", type=float, default=.02)
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--fit-window", type=int, default=5)
    parser.add_argument("--min-hits", default="5,10,20")
    parser.add_argument("--max-residuals", default="0.5,1,2")
    parser.add_argument("--score-thresholds", default="0,1,1.5,2,2.5,3,4")
    args = parser.parse_args()
    selected_resolutions = set(args.resolutions.split(","))
    min_hits_grid = [int(value) for value in args.min_hits.split(",")]
    residual_grid = [float(value) for value in args.max_residuals.split(",")]
    score_grid = [float(value) for value in args.score_thresholds.split(",")]

    baseline_total = defaultdict(int)
    records = []
    raw_candidates = 0
    for sequence in _sequences(Path(args.val_root)):
        stems, frames = _load_sequence_images(sequence)
        if not len(stems):
            continue
        height, width = frames.shape[1:]
        if f"{width}x{height}" not in selected_resolutions:
            continue
        prediction = parse(
            (Path(args.submission) / f"{sequence.name}.txt").read_text(
                encoding="ascii"
            ),
            sequence.name,
        )
        point_frames = [
            [
                (point.x, point.y)
                for point in prediction.frames[index]
            ]
            for index in range(1, len(stems) + 1)
        ]
        tracked = assign_track_ids(
            point_frames,
            (height, width),
            gate_fraction=args.gate_fraction,
            max_age=args.max_age,
        )
        histories = defaultdict(list)
        for points in tracked.values():
            for point in points:
                histories[point.track_id].append(point)
        candidates = defaultdict(list)
        for history in histories.values():
            history.sort(key=lambda point: point.frame_id)
            if len(history) < min(min_hits_grid):
                continue
            for side in ("start", "end"):
                sample = (
                    history[: args.fit_window]
                    if side == "start"
                    else history[-args.fit_window :]
                )
                if len(sample) < args.fit_window:
                    continue
                times = np.asarray([point.frame_id for point in sample], float)
                if not np.all(np.diff(times) == 1):
                    continue
                design = np.column_stack((times, np.ones_like(times)))
                xs = np.asarray([point.x for point in sample])
                ys = np.asarray([point.y for point in sample])
                x_fit = np.linalg.lstsq(design, xs, rcond=None)[0]
                y_fit = np.linalg.lstsq(design, ys, rcond=None)[0]
                fitted_x = design @ x_fit
                fitted_y = design @ y_fit
                residual = float(
                    np.sqrt(np.mean((xs - fitted_x) ** 2 + (ys - fitted_y) ** 2))
                )
                frame_id = int(times[0] - 1 if side == "start" else times[-1] + 1)
                if not (1 <= frame_id <= len(stems)):
                    continue
                x = float(x_fit[0] * frame_id + x_fit[1])
                y = float(y_fit[0] * frame_id + y_fit[1])
                if not (0 <= x < width and 0 <= y < height):
                    continue
                existing = point_frames[frame_id - 1]
                if any(
                    (x - old_x) ** 2 + (y - old_y) ** 2 <= args.radius**2
                    for old_x, old_y in existing
                ):
                    continue
                spatial, temporal = evidence(frames, frame_id - 1, x, y)
                candidates[frame_id].append(
                    {
                        "point": (x, y),
                        "hits": len(history),
                        "residual": residual,
                        "spatial": spatial,
                        "temporal": temporal,
                    }
                )
                raw_candidates += 1
        masks = _files(sequence / "mask")
        for frame_index, stem in enumerate(stems, 1):
            mask = np.asarray(Image.open(masks[stem]).convert("L"))
            truth = [(point.x, point.y) for point in centroids(mask, .5, 1)]
            baseline = point_frames[frame_index - 1]
            baseline_metrics = point_metrics(baseline, truth, args.radius)
            for key in ("tp", "fp", "fn"):
                baseline_total[key] += baseline_metrics[key]
            if candidates[frame_index]:
                records.append(
                    (baseline, truth, baseline_metrics, candidates[frame_index])
                )
        print(
            f"track-extrapolation sequence={sequence.name} "
            f"candidate_frames={len(records)}",
            flush=True,
        )

    sweep = []
    for min_hits in min_hits_grid:
        for max_residual in residual_grid:
            for score_name in ("spatial", "temporal", "minimum", "maximum"):
                for threshold in score_grid:
                    total = defaultdict(int, baseline_total)
                    added = 0
                    for baseline, truth, base_metrics, candidates in records:
                        selected = []
                        for candidate in candidates:
                            if (
                                candidate["hits"] < min_hits
                                or candidate["residual"] > max_residual
                            ):
                                continue
                            if score_name == "minimum":
                                score = min(
                                    candidate["spatial"], candidate["temporal"]
                                )
                            elif score_name == "maximum":
                                score = max(
                                    candidate["spatial"], candidate["temporal"]
                                )
                            else:
                                score = candidate[score_name]
                            if score >= threshold and all(
                                (candidate["point"][0] - old[0]) ** 2
                                + (candidate["point"][1] - old[1]) ** 2
                                > args.radius**2
                                for old in selected
                            ):
                                selected.append(candidate["point"])
                        if not selected:
                            continue
                        union_metrics = point_metrics(
                            baseline + selected, truth, args.radius
                        )
                        for key in ("tp", "fp", "fn"):
                            total[key] += union_metrics[key] - base_metrics[key]
                        added += len(selected)
                    row = {
                        "min_hits": min_hits,
                        "max_residual": max_residual,
                        "score": score_name,
                        "threshold": threshold,
                        "added": added,
                        **summary(total),
                    }
                    row["delta_tp"] = row["tp"] - baseline_total["tp"]
                    row["delta_fp"] = row["fp"] - baseline_total["fp"]
                    sweep.append(row)
    sweep.sort(key=lambda row: (row["f1"], row["precision"]), reverse=True)
    report = {
        "baseline": summary(baseline_total),
        "raw_candidates": raw_candidates,
        "candidate_frames": len(records),
        "best": sweep[:20],
        "metric_status": "local point-matching proxy; not official scorer",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": report["baseline"], "best": sweep[0]}, indent=2))


if __name__ == "__main__":
    main()
