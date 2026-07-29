"""Evaluate simple temporal-background point detection on labelled sequences."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from jinsight_track1.deeppro_adapter import (
    _files,
    _load_sequence_images,
    _sequences,
    component_points,
)
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.postprocess import centroids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--window-radius", type=int, default=4)
    parser.add_argument("--thresholds", default="1,2,3,4,5,6,8,10,12,16,20")
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int, default=50)
    parser.add_argument(
        "--methods",
        help="comma-separated methods; default evaluates every implemented method",
    )
    return parser.parse_args()


METHODS = [
    "adjacent_abs",
    "adjacent_positive",
    "adjacent_negative",
    "median_abs",
    "median_positive",
    "median_negative",
    "global_median_abs",
    "global_median_positive",
    "global_median_negative",
    *[
        f"lag{lag}_{polarity}"
        for lag in (20, 50, 100)
        for polarity in ("abs", "positive", "negative")
    ],
]


def temporal_score_frames(frames: np.ndarray, method: str, radius: int):
    """Yield one score map at a time to keep long videos below 1 GB RAM."""
    if method.startswith("global_median_"):
        stride = max(1, len(frames) // 80)
        background = np.median(frames[::stride], axis=0).astype(np.float32)
    for index, frame in enumerate(frames):
        current = frame.astype(np.float32)
        if method.startswith("adjacent_"):
            reference = frames[max(0, index - 1)].astype(np.float32)
        elif method.startswith("median_"):
            start, end = max(0, index - radius), min(len(frames), index + radius + 1)
            reference = np.median(frames[start:end], axis=0).astype(np.float32)
        elif method.startswith("global_median_"):
            reference = background
        elif method.startswith("lag"):
            lag = int(method.split("_", 1)[0][3:])
            reference = frames[max(0, index - lag)].astype(np.float32)
        else:  # pragma: no cover - guarded by argparse validation below
            raise ValueError(method)
        residual = current - reference
        if method.endswith("_abs"):
            yield np.abs(residual)
        elif method.endswith("_positive"):
            yield np.maximum(residual, 0)
        else:
            yield np.maximum(-residual, 0)


def main() -> None:
    args = parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    selected = args.methods.split(",") if args.methods else METHODS
    unknown = set(selected) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    sequences = _sequences(Path(args.root))
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    totals: dict[tuple[str, float], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    frames_seen = 0
    for sequence in sequences:
        stems, frames = _load_sequence_images(sequence, args.max_frames)
        masks = _files(sequence / "mask")
        truth_frames = []
        for stem in stems:
            mask = np.asarray(Image.open(masks[stem]).convert("L"))
            truth_frames.append(
                [(point.x, point.y) for point in centroids(mask, 0.5, 1)]
            )
        for method in selected:
            for score, truth in zip(
                temporal_score_frames(frames, method, args.window_radius),
                truth_frames,
            ):
                # A light blur joins the target pixels while retaining its centre.
                score = cv2.GaussianBlur(score, (3, 3), 0)
                for threshold in thresholds:
                    predicted = component_points(
                        score,
                        threshold,
                        min_area=args.min_area,
                        max_area=args.max_area,
                    )
                    metrics = point_metrics(predicted, truth, args.radius)
                    bucket = totals[(method, threshold)]
                    for key in bucket:
                        bucket[key] += metrics[key]
        frames_seen += len(stems)
        print(f"temporal-diff {sequence.name} frames={frames_seen}", flush=True)

    sweep = []
    for (method, threshold), values in totals.items():
        tp, fp, fn = (values[key] for key in ("tp", "fp", "fn"))
        sweep.append(
            {
                "method": method,
                "threshold": threshold,
                **values,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            }
        )
    sweep.sort(key=lambda row: (-row["f1"], row["method"], row["threshold"]))
    print(json.dumps({"frames": frames_seen, "best": sweep[:20]}, indent=2))


if __name__ == "__main__":
    main()
