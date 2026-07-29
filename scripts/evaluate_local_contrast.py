"""Evaluate training-free local-contrast point detectors on labelled videos."""
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


def responses(
    image: np.ndarray,
    temporal_background: np.ndarray | None = None,
    temporal_deviation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    value = image.astype(np.float32)
    small = cv2.GaussianBlur(value, (0, 0), 0.8)
    large = cv2.GaussianBlur(value, (0, 0), 2.4)
    dog = small - large
    mean = cv2.boxFilter(value, cv2.CV_32F, (11, 11), normalize=True)
    mean2 = cv2.boxFilter(value * value, cv2.CV_32F, (11, 11), normalize=True)
    deviation = np.sqrt(np.maximum(mean2 - mean * mean, 4.0))
    local_z = (value - mean) / deviation
    found = {
        "dog_bright": np.maximum(dog, 0),
        "dog_dark": np.maximum(-dog, 0),
        "z_bright": np.maximum(local_z, 0),
        "z_dark": np.maximum(-local_z, 0),
    }
    for size in (5, 9, 15):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        found[f"tophat_{size}"] = cv2.morphologyEx(
            image, cv2.MORPH_TOPHAT, kernel
        ).astype(np.float32)
        found[f"blackhat_{size}"] = cv2.morphologyEx(
            image, cv2.MORPH_BLACKHAT, kernel
        ).astype(np.float32)
    if temporal_background is not None and temporal_deviation is not None:
        delta = value - temporal_background
        found["temporal_bright"] = np.maximum(delta, 0)
        found["temporal_dark"] = np.maximum(-delta, 0)
        found["temporal_z_bright"] = np.maximum(delta / temporal_deviation, 0)
        found["temporal_z_dark"] = np.maximum(-delta / temporal_deviation, 0)
    return found


def topk_points(
    response: np.ndarray, count: int, nms_radius: float = 3.0
) -> list[tuple[float, float]]:
    dilated = cv2.dilate(response, np.ones((3, 3), np.uint8))
    ys, xs = np.where((response >= dilated) & (response > 0))
    if not len(xs):
        return []
    scores = response[ys, xs]
    candidate_limit = min(len(scores), max(200, count * 50))
    if len(scores) > candidate_limit:
        keep = np.argpartition(scores, -candidate_limit)[-candidate_limit:]
        xs, ys, scores = xs[keep], ys[keep], scores[keep]
    order = np.argsort(scores)[::-1]
    selected: list[tuple[float, float]] = []
    radius2 = nms_radius * nms_radius
    for index in order:
        point = (float(xs[index]), float(ys[index]))
        if all(
            (point[0] - old[0]) ** 2 + (point[1] - old[1]) ** 2 > radius2
            for old in selected
        ):
            selected.append(point)
            if len(selected) == count:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--top-k", default="1,2,3,5,8,10,15,20,30,50")
    parser.add_argument("--resolutions", help="comma-separated WxH values")
    parser.add_argument("--methods", help="comma-separated response methods")
    parser.add_argument("--output")
    args = parser.parse_args()
    top_counts = [int(value) for value in args.top_k.split(",")]
    selected_resolutions = set(args.resolutions.split(",")) if args.resolutions else None
    selected_methods = set(args.methods.split(",")) if args.methods else None
    sequences = _sequences(Path(args.root))
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    sequence_totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    frames_seen = 0
    for sequence in sequences:
        stems, frames = _load_sequence_images(sequence, args.max_frames)
        masks = _files(sequence / "mask")
        height, width = frames.shape[1:]
        resolution = f"{width}x{height}"
        if selected_resolutions and resolution not in selected_resolutions:
            continue
        stride = max(1, len(frames) // 80)
        temporal_sample = frames[::stride].astype(np.float32)
        temporal_background = np.median(temporal_sample, axis=0).astype(np.float32)
        temporal_deviation = np.maximum(temporal_sample.std(axis=0), 2.0)
        for stem, frame in zip(stems, frames):
            mask = np.asarray(Image.open(masks[stem]).convert("L"))
            truth = [(point.x, point.y) for point in centroids(mask, 0.5, 1)]
            for method, response in responses(
                frame, temporal_background, temporal_deviation
            ).items():
                if selected_methods and method not in selected_methods:
                    continue
                largest = topk_points(response, max(top_counts))
                for count in top_counts:
                    metrics = point_metrics(largest[:count], truth, args.radius)
                    bucket = totals[(resolution, method, count)]
                    sequence_bucket = sequence_totals[
                        (sequence.name, resolution, method, count)
                    ]
                    for key in bucket:
                        bucket[key] += metrics[key]
                        sequence_bucket[key] += metrics[key]
            frames_seen += 1
        print(f"local-contrast {sequence.name} frames={frames_seen}", flush=True)
    rows = []
    for (resolution, method, count), values in totals.items():
        tp, fp, fn = (values[key] for key in ("tp", "fp", "fn"))
        rows.append(
            {
                "resolution": resolution,
                "method": method,
                "top_k": count,
                **values,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            }
        )
    best = {}
    for resolution in sorted({row["resolution"] for row in rows}):
        choices = [row for row in rows if row["resolution"] == resolution]
        best[resolution] = max(
            choices, key=lambda row: (row["f1"], row["recall"], row["precision"])
        )
    sequence_rows = []
    for (sequence, resolution, method, count), values in sequence_totals.items():
        tp, fp, fn = (values[key] for key in ("tp", "fp", "fn"))
        sequence_rows.append(
            {
                "sequence": sequence,
                "resolution": resolution,
                "method": method,
                "top_k": count,
                **values,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            }
        )
    best_by_sequence = {}
    for sequence in sorted({row["sequence"] for row in sequence_rows}):
        choices = [row for row in sequence_rows if row["sequence"] == sequence]
        best_by_sequence[sequence] = max(
            choices, key=lambda row: (row["f1"], row["recall"], row["precision"])
        )
    report = {
        "frames": frames_seen,
        "best_by_resolution": best,
        "best_by_sequence": best_by_sequence,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
