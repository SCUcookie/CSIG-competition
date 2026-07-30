#!/usr/bin/env python3
"""Evaluate a frame-dynamics YOLO detector with point-radius metrics."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from jinsight_track1.evaluation import point_metrics
from jinsight_track1.frame_dynamics_yolo import image_files, sequence_dirs
from jinsight_track1.postprocess import centroids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dataset_root")
    parser.add_argument("raw_val_root")
    parser.add_argument("output")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--prediction-cache")
    parser.add_argument(
        "--conf-grid", type=float, nargs="+",
        default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3],
    )
    return parser.parse_args()


def summary(counts: dict, confidence: float) -> dict:
    tp, fp, fn = (int(counts[k]) for k in ("tp", "fp", "fn"))
    return {
        "confidence": confidence,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.model)
    thresholds = sorted(set(args.conf_grid))
    transformed = Path(args.dataset_root) / "images" / "val"
    totals = {threshold: defaultdict(int) for threshold in thresholds}
    per_sequence = {threshold: {} for threshold in thresholds}
    prediction_cache: dict[str, list[dict]] = {}
    nearest_truth_distances: list[float] = []
    frames = 0
    for sequence in sequence_dirs(args.raw_val_root):
        masks = {p.stem: p for p in image_files(sequence / "mask")}
        sources = sorted(
            transformed.glob(f"{sequence.name}__*.png"),
            key=lambda p: [int(x) if x.isdigit() else x for x in __import__("re").split(r"(\d+)", p.name)],
        )
        sequence_totals = {threshold: defaultdict(int) for threshold in thresholds}
        sequence_cache: list[dict] = []
        returned = 0
        # Ultralytics treats a Python list as one in-memory batch; explicitly
        # chunk it to honour --batch and avoid allocating all video frames on
        # the GPU at once.
        for start in range(0, len(sources), args.batch):
            chunk = sources[start:start + args.batch]
            results = model.predict(
                source=[str(p) for p in chunk],
                device=args.device,
                imgsz=args.imgsz,
                conf=min(thresholds),
                max_det=50,
                iou=0.5,
                verbose=False,
            )
            for source, result in zip(chunk, results):
                returned += 1
                frames += 1
                stem = source.stem.split("__", 1)[1]
                mask = cv2.imread(str(masks[stem]), cv2.IMREAD_GRAYSCALE)
                truth = [(p.x, p.y) for p in centroids(mask, 0.5, 1)]
                boxes = result.boxes
                predicted = []
                if boxes is not None:
                    xywh = boxes.xywh.detach().cpu().numpy()
                    conf = boxes.conf.detach().cpu().numpy()
                    predicted = [
                        (float(row[0]), float(row[1]), float(score))
                        for row, score in zip(xywh, conf)
                    ]
                for threshold in thresholds:
                    points = [(x, y) for x, y, score in predicted if score >= threshold]
                    metric = point_metrics(points, truth, args.radius)
                    for key in ("tp", "fp", "fn"):
                        value = int(metric[key])
                        totals[threshold][key] += value
                        sequence_totals[threshold][key] += value
                candidate_points = np.asarray(
                    [(x, y) for x, y, _ in predicted], dtype=float
                ).reshape(-1, 2)
                truth_points = np.asarray(truth, dtype=float).reshape(-1, 2)
                if len(truth_points):
                    if len(candidate_points):
                        distances = np.linalg.norm(
                            truth_points[:, None] - candidate_points[None, :], axis=2
                        ).min(axis=1)
                        nearest_truth_distances.extend(float(v) for v in distances)
                    else:
                        nearest_truth_distances.extend([float("inf")] * len(truth_points))
                if args.prediction_cache:
                    sequence_cache.append(
                        {
                            "stem": stem,
                            "truth": [[float(x), float(y)] for x, y in truth],
                            "predicted": [
                                [float(x), float(y), float(score)]
                                for x, y, score in predicted
                            ],
                        }
                    )
        if returned != len(sources):
            raise RuntimeError(f"{sequence.name}: got {returned}/{len(sources)} results")
        for threshold in thresholds:
            per_sequence[threshold][sequence.name] = summary(
                sequence_totals[threshold], threshold
            )
        if args.prediction_cache:
            prediction_cache[sequence.name] = sequence_cache
    sweep = [summary(totals[t], t) for t in thresholds]
    best = max(sweep, key=lambda row: (row["f1"], row["recall"]))
    report = {
        "model": args.model,
        "frames": frames,
        "radius": args.radius,
        "best": best,
        "sweep": sweep,
        "best_sequence_breakdown": per_sequence[best["confidence"]],
        "nearest_truth_coverage": {
            str(radius): sum(distance <= radius for distance in nearest_truth_distances)
            / max(1, len(nearest_truth_distances))
            for radius in (2, 5, 10, 15, 20, 30, 40, 60, 80)
        },
    }
    if args.prediction_cache:
        cache_path = Path(args.prediction_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(prediction_cache), encoding="utf-8")
        report["prediction_cache"] = str(cache_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
