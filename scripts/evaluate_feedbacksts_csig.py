#!/usr/bin/env python
"""Evaluate FeedbackSTS-Det checkpoints with CSIG centroid point F1."""
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
    component_points,
)
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.postprocess import centroids
from train_feedbacksts_csig import load_feedbacksts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("weights")
    parser.add_argument("val_root")
    parser.add_argument("output")
    parser.add_argument("--device", default="0")
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--resolutions", default="256x256")
    parser.add_argument("--mean", type=float, default=111.47)
    parser.add_argument("--std", type=float, default=22.43)
    parser.add_argument(
        "--thresholds",
        default="0.01,0.03,0.1,0.2,0.3,0.5,0.7,0.9,0.97,0.99",
    )
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int, default=30)
    parser.add_argument(
        "--centroid-mode",
        choices=("binary", "weighted", "peak"),
        default="weighted",
    )
    args = parser.parse_args()

    import torch

    device = torch.device(f"cuda:{args.device}")
    model = load_feedbacksts(args.source_root, torch).to(device)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    resolutions = set(args.resolutions.split(",")) if args.resolutions else None
    sequences = []
    for sequence in _sequences(Path(args.val_root)):
        images = _files(sequence / "img")
        width, height = Image.open(next(iter(images.values()))).size
        if resolutions and f"{width}x{height}" not in resolutions:
            continue
        sequences.append(sequence)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]

    totals = {threshold: defaultdict(int) for threshold in thresholds}
    frame_count = 0
    with torch.inference_mode():
        for sequence_index, sequence in enumerate(sequences, 1):
            stems, frames = _load_sequence_images(sequence, None)
            masks = _files(sequence / "mask")
            probabilities = np.zeros(frames.shape, dtype=np.float32)
            coverage = np.zeros(len(frames), dtype=np.uint16)
            starts = list(range(0, len(frames), args.sequence_length))
            for batch_start in range(0, len(starts), args.batch_size):
                batch_starts = starts[batch_start : batch_start + args.batch_size]
                chunks = []
                valid_counts = []
                for start in batch_starts:
                    valid = min(args.sequence_length, len(frames) - start)
                    chunk = frames[start : start + valid].astype(np.float32)
                    if valid < args.sequence_length:
                        chunk = np.concatenate(
                            (
                                chunk,
                                np.repeat(
                                    chunk[-1:],
                                    args.sequence_length - valid,
                                    axis=0,
                                ),
                            )
                        )
                    chunks.append((chunk - args.mean) / args.std)
                    valid_counts.append(valid)
                tensor = torch.from_numpy(np.stack(chunks)[:, None]).to(device)
                with torch.amp.autocast("cuda"):
                    predicted = model(tensor).float().cpu().numpy()[:, 0]
                for item, (start, valid) in enumerate(
                    zip(batch_starts, valid_counts)
                ):
                    probabilities[start : start + valid] += predicted[item, :valid]
                    coverage[start : start + valid] += 1
            probabilities /= np.maximum(coverage[:, None, None], 1)

            for index, stem in enumerate(stems):
                mask = np.asarray(Image.open(masks[stem]).convert("L"))
                truth = [
                    (point.x, point.y) for point in centroids(mask, 0.5, 1)
                ]
                for threshold in thresholds:
                    predicted = component_points(
                        probabilities[index],
                        threshold,
                        min_area=args.min_area,
                        max_area=args.max_area,
                        centroid_mode=args.centroid_mode,
                    )
                    metrics = point_metrics(predicted, truth, args.radius)
                    for key in ("tp", "fp", "fn"):
                        totals[threshold][key] += metrics[key]
            frame_count += len(stems)
            print(
                f"feedbacksts-eval {sequence_index}/{len(sequences)} "
                f"sequence={sequence.name} frames={frame_count}",
                flush=True,
            )

    sweep = []
    for threshold in thresholds:
        tp, fp, fn = (totals[threshold][key] for key in ("tp", "fp", "fn"))
        sweep.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            }
        )
    report = {
        "weights": args.weights,
        "sequences": len(sequences),
        "frames": frame_count,
        "radius": args.radius,
        "sweep": sweep,
        "best": max(sweep, key=lambda row: (row["f1"], row["recall"])),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
