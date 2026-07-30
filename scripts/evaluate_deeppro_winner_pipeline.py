"""Evaluate phase-compensated, bidirectional DeepPro challenge refinements."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from jinsight_track1.deeppro_adapter import (
    DeepProDetector,
    _files,
    _load_sequence_images,
    _sequences,
    component_points,
)
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.postprocess import centroids


MODES = {
    "forward": {"bidirectional": False, "motion_compensation": False},
    "bidirectional": {"bidirectional": True, "motion_compensation": False},
    "pcm_forward": {"bidirectional": False, "motion_compensation": True},
    "pcm_bidirectional": {"bidirectional": True, "motion_compensation": True},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("weights")
    parser.add_argument("val_root")
    parser.add_argument("--output")
    parser.add_argument("--device", default="1")
    parser.add_argument("--resolutions", default="640x512")
    parser.add_argument("--sequences")
    parser.add_argument("--modes", default="pcm_forward,pcm_bidirectional")
    parser.add_argument("--fusion", choices=("max", "mean", "geometric"), default="max")
    parser.add_argument("--thresholds", default="1e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,.03,.1,.3,.5,.7,.9,.99")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-halo", type=int, default=12)
    parser.add_argument("--phase-max-side", type=int, default=512)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int)
    parser.add_argument("--centroid-mode", choices=("binary", "weighted", "peak"), default="binary")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Skip connected components and compare truth-site probabilities with image-wide percentiles.",
    )
    return parser.parse_args()


def summary(values, threshold):
    tp, fp, fn = (int(values[key]) for key in ("tp", "fp", "fn"))
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def main():
    args = parse_args()
    modes = args.modes.split(",")
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    thresholds = [float(value) for value in args.thresholds.split(",")]
    resolutions = set(args.resolutions.split(",")) if args.resolutions else None
    selected_sequences = set(args.sequences.split(",")) if args.sequences else None
    detector = DeepProDetector(
        args.source_root,
        args.weights,
        device=args.device,
        tile_size=args.tile_size,
        tile_halo=args.tile_halo,
    )
    totals = {
        mode: {threshold: defaultdict(int) for threshold in thresholds}
        for mode in modes
    }
    per_sequence = {}
    frames_seen = 0
    for sequence in _sequences(Path(args.val_root)):
        if selected_sequences and sequence.name not in selected_sequences:
            continue
        stems, frames = _load_sequence_images(sequence, args.max_frames)
        if not len(frames):
            continue
        height, width = frames.shape[1:]
        resolution = f"{width}x{height}"
        if resolutions and resolution not in resolutions:
            continue
        masks = _files(sequence / "mask")
        truth_frames = []
        for stem in stems:
            mask = np.asarray(Image.open(masks[stem]).convert("L"))
            truth_frames.append(
                [(point.x, point.y) for point in centroids(mask, 0.5, 1)]
            )
        sequence_rows = {}
        for mode in modes:
            options = dict(MODES[mode])
            probabilities = detector.predict(
                frames,
                fusion=args.fusion,
                phase_max_side=args.phase_max_side,
                **options,
            )
            probability_percentiles = {
                str(level): float(np.percentile(probabilities, level))
                for level in (50, 90, 99, 99.9, 99.99, 100)
            }
            if args.diagnostic_only:
                target_scores = []
                radius = int(np.ceil(args.radius))
                for probability, truth in zip(probabilities, truth_frames):
                    for x, y in truth:
                        ix, iy = int(round(x)), int(round(y))
                        y0, y1 = max(0, iy - radius), min(height, iy + radius + 1)
                        x0, x1 = max(0, ix - radius), min(width, ix + radius + 1)
                        target_scores.append(float(np.max(probability[y0:y1, x0:x1])))
                target_percentiles = {
                    str(level): float(np.percentile(target_scores, level))
                    for level in (0, 10, 25, 50, 75, 90, 100)
                } if target_scores else {}
                sequence_rows[mode] = {
                    "target_points": len(target_scores),
                    "target_probability_percentiles": target_percentiles,
                    "probability_percentiles": probability_percentiles,
                }
                continue
            mode_totals = {threshold: defaultdict(int) for threshold in thresholds}
            for probability, truth in zip(probabilities, truth_frames):
                for threshold in thresholds:
                    predicted = component_points(
                        probability,
                        threshold,
                        min_area=args.min_area,
                        max_area=args.max_area,
                        centroid_mode=args.centroid_mode,
                    )
                    metrics = point_metrics(predicted, truth, args.radius)
                    for key in ("tp", "fp", "fn"):
                        totals[mode][threshold][key] += metrics[key]
                        mode_totals[threshold][key] += metrics[key]
            sweep = [
                summary(mode_totals[threshold], threshold)
                for threshold in thresholds
            ]
            sequence_rows[mode] = {
                "best": max(sweep, key=lambda row: row["f1"]),
                "sweep": sweep,
                "probability_percentiles": probability_percentiles,
            }
        per_sequence[sequence.name] = {
            "resolution": resolution,
            "frames": len(stems),
            "truth": sum(map(len, truth_frames)),
            "modes": sequence_rows,
        }
        frames_seen += len(stems)
        print(
            f"winner-pipeline {sequence.name} frames={frames_seen} "
            + " ".join(
                (
                    f"{mode}=target-p50:"
                    f"{sequence_rows[mode]['target_probability_percentiles'].get('50', float('nan')):.3g}"
                    if args.diagnostic_only
                    else f"{mode}={sequence_rows[mode]['best']['f1']:.4f}"
                )
                for mode in modes
            ),
            flush=True,
        )

    mode_reports = {}
    for mode in modes:
        if args.diagnostic_only:
            mode_reports[mode] = {"diagnostic_only": True}
            continue
        sweep = [
            summary(totals[mode][threshold], threshold) for threshold in thresholds
        ]
        mode_reports[mode] = {
            "best": max(sweep, key=lambda row: row["f1"]),
            "sweep": sweep,
        }
    report = {
        "settings": vars(args),
        "frames": frames_seen,
        "modes": mode_reports,
        "sequences": per_sequence,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
