"""Measure whether MSHNet adds useful detections to an existing submission."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from jinsight_track1.data import natural_key
from jinsight_track1.deeppro_adapter import _files, _load_sequence_images, _sequences, component_points
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.mshnet_adapter import MSHNetDetector
from jinsight_track1.postprocess import centroids
from jinsight_track1.submission import parse


def add_metrics(total, row):
    for key in ("tp", "fp", "fn"):
        total[key] += row[key]


def summarize(row):
    tp, fp, fn = (int(row[key]) for key in ("tp", "fp", "fn"))
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
    parser.add_argument("source_root")
    parser.add_argument("weights")
    parser.add_argument("val_root")
    parser.add_argument("baseline_submission")
    parser.add_argument("output")
    parser.add_argument("--device", default="0")
    parser.add_argument("--thresholds", default="0.99999,0.999995,0.999999,0.9999995")
    parser.add_argument("--resolutions")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--dedup-radius", type=float, default=2.0)
    parser.add_argument("--min-area", type=int, default=3)
    parser.add_argument("--max-area", type=int, default=30)
    parser.add_argument("--adaptive-normalization", action="store_true")
    args = parser.parse_args()

    thresholds = [float(value) for value in args.thresholds.split(",")]
    selected_resolutions = (
        set(args.resolutions.split(",")) if args.resolutions else None
    )
    detector = MSHNetDetector(
        args.source_root,
        args.weights,
        device=args.device,
        batch_size=args.batch_size,
        adaptive_normalization=args.adaptive_normalization,
    )
    totals = {
        threshold: {
            "baseline": defaultdict(int),
            "model": defaultdict(int),
            "union": defaultdict(int),
        }
        for threshold in thresholds
    }
    frames_seen = 0
    for sequence in _sequences(Path(args.val_root)):
        stems, frames = _load_sequence_images(sequence, args.max_frames)
        if not len(stems):
            continue
        height, width = frames.shape[1:]
        if (
            selected_resolutions
            and f"{width}x{height}" not in selected_resolutions
        ):
            continue
        prediction = parse(
            (Path(args.baseline_submission) / f"{sequence.name}.txt").read_text(
                encoding="ascii"
            ),
            sequence.name,
        )
        probabilities = detector.predict(frames)
        masks = _files(sequence / "mask")
        ordered_masks = sorted(masks, key=lambda value: natural_key(Path(value)))
        if stems != ordered_masks[: len(stems)]:
            raise ValueError(f"image/mask order differs for {sequence.name}")
        for index, stem in enumerate(stems):
            truth_mask = np.asarray(Image.open(masks[stem]).convert("L"))
            truth = [
                (point.x, point.y)
                for point in centroids(truth_mask, .5, 1)
            ]
            baseline = [
                (point.x, point.y)
                for point in prediction.frames[index + 1]
            ]
            baseline_metrics = point_metrics(baseline, truth, args.radius)
            for threshold in thresholds:
                model = component_points(
                    probabilities[index],
                    threshold,
                    min_area=args.min_area,
                    max_area=args.max_area,
                )
                novel = [
                    point
                    for point in model
                    if all(
                        (point[0] - old[0]) ** 2 + (point[1] - old[1]) ** 2
                        > args.dedup_radius**2
                        for old in baseline
                    )
                ]
                add_metrics(totals[threshold]["baseline"], baseline_metrics)
                add_metrics(
                    totals[threshold]["model"],
                    point_metrics(model, truth, args.radius),
                )
                add_metrics(
                    totals[threshold]["union"],
                    point_metrics(baseline + novel, truth, args.radius),
                )
            frames_seen += 1
        print(
            f"mshnet-complement sequence={sequence.name} frames={frames_seen}",
            flush=True,
        )

    sweep = []
    for threshold in thresholds:
        row = {
            "threshold": threshold,
            **{
                name: summarize(values)
                for name, values in totals[threshold].items()
            },
        }
        baseline = row["baseline"]
        union = row["union"]
        added_tp = union["tp"] - baseline["tp"]
        added_fp = union["fp"] - baseline["fp"]
        row["marginal"] = {
            "tp": added_tp,
            "fp": added_fp,
            "precision": added_tp / max(1, added_tp + added_fp),
        }
        sweep.append(row)
    report = {
        "frames": frames_seen,
        "min_area": args.min_area,
        "max_area": args.max_area,
        "sweep": sweep,
        "best_union": max(sweep, key=lambda row: row["union"]["f1"]),
        "metric_status": "local point-matching proxy; not official scorer",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["best_union"], indent=2))


if __name__ == "__main__":
    main()
