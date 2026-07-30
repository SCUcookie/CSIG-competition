#!/usr/bin/env python3
"""Evaluate a CSIG-adapted HDNet checkpoint with centroid point F1."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from jinsight_track1.deeppro_adapter import (
    _files,
    _load_sequence_images,
    _sequences,
    _summary,
    component_points,
)
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.hdnet_adapter import HDNetDetector
from jinsight_track1.postprocess import centroids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("weights")
    parser.add_argument("val_root")
    parser.add_argument("output_json")
    parser.add_argument("--device", default="0")
    parser.add_argument("--resolution", default="256x256")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--thresholds", default=".5,.7,.9,.95,.99,.995,.999,.9999")
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int, default=30)
    parser.add_argument("--radius", type=float, default=2.0)
    args = parser.parse_args()

    thresholds = sorted({float(value) for value in args.thresholds.split(",")})
    width, height = (int(value) for value in args.resolution.split("x"))
    detector = HDNetDetector(
        args.source_root,
        args.weights,
        device=args.device,
        batch_size=args.batch_size,
    )
    sequences = []
    for sequence in _sequences(Path(args.val_root)):
        first = next(iter(_files(sequence / "img").values()))
        if Image.open(first).size == (width, height):
            sequences.append(sequence)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    totals = {threshold: defaultdict(float) for threshold in thresholds}
    frame_count = 0
    for sequence_index, sequence in enumerate(sequences, 1):
        stems, frames = _load_sequence_images(sequence, args.max_frames)
        probabilities = detector.predict(frames)
        masks = _files(sequence / "mask")
        for index, stem in enumerate(stems):
            truth = [
                (point.x, point.y)
                for point in centroids(
                    np.asarray(Image.open(masks[stem]).convert("L")), 0.5, 1
                )
            ]
            for threshold in thresholds:
                predicted = component_points(
                    probabilities[index],
                    threshold,
                    min_area=args.min_area,
                    max_area=args.max_area,
                    centroid_mode="peak",
                )
                metrics = point_metrics(predicted, truth, args.radius)
                for key in ("tp", "fp", "fn"):
                    totals[threshold][key] += metrics[key]
            frame_count += 1
        print(
            f"hdnet-eval {sequence_index}/{len(sequences)} {sequence.name}",
            flush=True,
        )
    sweep = [_summary(totals[value], value) for value in thresholds]
    best = max(sweep, key=lambda row: row["f1"])
    report = {
        "model": "HDNet temporal-difference",
        "weights": str(Path(args.weights).resolve()),
        "resolution": args.resolution,
        "sequences": len(sequences),
        "frames": frame_count,
        **best,
        "best_threshold": best["threshold"],
        "sweep": sweep,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
