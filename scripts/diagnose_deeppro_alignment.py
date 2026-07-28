"""Diagnose spatial or temporal misalignment in DeepPro point predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from jinsight_track1.deeppro_adapter import (
    DeepProDetector,
    _files,
    _load_sequence_images,
    component_points,
)
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.postprocess import centroids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-halo", type=int, default=12)
    return parser.parse_args()


def transform_points(
    points: list[tuple[float, float]], name: str, width: int, height: int
) -> list[tuple[float, float]]:
    transformed = []
    for x, y in points:
        if name.startswith("swap"):
            x, y = y, x
        if "hflip" in name:
            x = width - 1 - x
        if "vflip" in name:
            y = height - 1 - y
        transformed.append((x, y))
    return transformed


def aggregate(
    predicted: list[list[tuple[float, float]]],
    truth: list[list[tuple[float, float]]],
    radius: float,
    transform: str = "identity",
    lag: int = 0,
    width: int = 0,
    height: int = 0,
) -> dict:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for index, points in enumerate(predicted):
        truth_index = index + lag
        if not 0 <= truth_index < len(truth):
            continue
        moved = transform_points(points, transform, width, height)
        metrics = point_metrics(moved, truth[truth_index], radius)
        for key in totals:
            totals[key] += metrics[key]
    tp, fp, fn = (totals[key] for key in ("tp", "fp", "fn"))
    totals.update(
        precision=tp / max(1, tp + fp),
        recall=tp / max(1, tp + fn),
        f1=2 * tp / max(1, 2 * tp + fp + fn),
    )
    return totals


def main() -> None:
    args = parse_args()
    sequence = Path(args.sequence)
    stems, frames = _load_sequence_images(sequence, args.max_frames)
    detector = DeepProDetector(
        args.source_root,
        args.weights,
        device=args.device,
        tile_size=args.tile_size,
        tile_halo=args.tile_halo,
    )
    probabilities = detector.predict(frames)
    masks = _files(sequence / "mask")
    truth: list[list[tuple[float, float]]] = []
    predicted: list[list[tuple[float, float]]] = []
    truth_neighbourhood_max: list[float] = []
    truth_pixel_values: list[float] = []

    for index, stem in enumerate(stems):
        points = []
        if stem in masks:
            mask = np.asarray(Image.open(masks[stem]).convert("L"))
            points = [(point.x, point.y) for point in centroids(mask, 0.5, 1)]
        truth.append(points)
        predicted.append(component_points(probabilities[index], args.threshold))
        for x, y in points:
            xi, yi = int(round(x)), int(round(y))
            x0, x1 = max(0, xi - 2), min(frames.shape[2], xi + 3)
            y0, y1 = max(0, yi - 2), min(frames.shape[1], yi + 3)
            truth_neighbourhood_max.append(
                float(probabilities[index, y0:y1, x0:x1].max(initial=0.0))
            )
            truth_pixel_values.append(float(frames[index, yi, xi]))

    height, width = frames.shape[1:]
    transforms = [
        "identity",
        "hflip",
        "vflip",
        "hflip_vflip",
        "swap",
        "swap_hflip",
        "swap_vflip",
        "swap_hflip_vflip",
    ]
    transform_scores = {
        name: aggregate(
            predicted, truth, args.radius, name, 0, width=width, height=height
        )
        for name in transforms
    }
    lag_scores = {
        str(lag): aggregate(
            predicted, truth, args.radius, "identity", lag, width, height
        )
        for lag in range(-8, 9)
    }

    nearest_residuals = []
    nearest_distances = []
    for pred_frame, truth_frame in zip(predicted, truth):
        if not pred_frame or not truth_frame:
            continue
        pred_array = np.asarray(pred_frame)
        for truth_point in truth_frame:
            residuals = pred_array - np.asarray(truth_point)
            distances = np.linalg.norm(residuals, axis=1)
            nearest = int(np.argmin(distances))
            nearest_residuals.append(residuals[nearest])
            nearest_distances.append(float(distances[nearest]))

    report = {
        "sequence": sequence.name,
        "resolution": f"{width}x{height}",
        "frames": len(stems),
        "threshold": args.threshold,
        "truth_points": sum(map(len, truth)),
        "predicted_points": sum(map(len, predicted)),
        "transform_scores": transform_scores,
        "lag_scores": lag_scores,
        "nearest_residual_median_xy": (
            np.median(nearest_residuals, axis=0).tolist()
            if nearest_residuals
            else None
        ),
        "nearest_distance_percentiles": (
            {
                str(value): float(np.percentile(nearest_distances, value))
                for value in (0, 25, 50, 75, 90, 100)
            }
            if nearest_distances
            else {}
        ),
        "truth_neighbourhood_probability_percentiles": (
            {
                str(value): float(np.percentile(truth_neighbourhood_max, value))
                for value in (0, 25, 50, 75, 90, 100)
            }
            if truth_neighbourhood_max
            else {}
        ),
        "truth_image_value_percentiles": (
            {
                str(value): float(np.percentile(truth_pixel_values, value))
                for value in (0, 25, 50, 75, 90, 100)
            }
            if truth_pixel_values
            else {}
        ),
        "probability_percentiles": {
            str(value): float(np.percentile(probabilities, value))
            for value in (0, 50, 90, 99, 99.9, 99.99, 100)
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
