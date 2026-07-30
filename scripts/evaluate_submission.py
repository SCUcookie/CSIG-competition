"""Evaluate an assembled Track 1 submission directory with the local point proxy."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from jinsight_track1.data import natural_key
from jinsight_track1.deeppro_adapter import _files, _sequences
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.postprocess import centroids
from jinsight_track1.submission import parse


def summary(values: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = values["tp"], values["fp"], values["fn"]
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
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--coordinate-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--resolutions", help="optional comma-separated WxH filter")
    parser.add_argument("--output")
    args = parser.parse_args()

    totals: dict[str, int] = defaultdict(int)
    by_resolution: dict[str, dict[str, int]] = {}
    by_sequence: dict[str, dict[str, int]] = {}
    sequences = _sequences(Path(args.val_root))
    selected_resolutions = (
        set(args.resolutions.split(",")) if args.resolutions else None
    )
    submission = Path(args.submission)
    evaluated_sequences = 0
    for sequence_index, sequence in enumerate(sequences, 1):
        masks = _files(sequence / "mask")
        first_mask = next(iter(masks.values()))
        first_shape = np.asarray(Image.open(first_mask)).shape
        resolution = f"{first_shape[1]}x{first_shape[0]}"
        if selected_resolutions and resolution not in selected_resolutions:
            continue
        prediction = parse(
            (submission / f"{sequence.name}.txt").read_text(encoding="ascii"),
            sequence.name,
            args.coordinate_order,
        )
        stems = sorted(masks, key=lambda value: natural_key(Path(value)))
        expected_frames = set(range(1, len(stems) + 1))
        if set(prediction.frames) != expected_frames:
            raise ValueError(f"frame mismatch for sequence {sequence.name}")
        sequence_totals: dict[str, int] = defaultdict(int)
        for frame_index, stem in enumerate(stems, 1):
            mask = np.asarray(Image.open(masks[stem]).convert("L"))
            height, width = mask.shape
            predicted = [
                (point.x, point.y) for point in prediction.frames[frame_index]
            ]
            if any(not (0 <= x < width and 0 <= y < height) for x, y in predicted):
                raise ValueError(
                    f"out-of-bounds point in {sequence.name} frame {frame_index}"
                )
            truth = [(point.x, point.y) for point in centroids(mask, 0.5, 1)]
            metrics = point_metrics(predicted, truth, args.radius)
            bucket = by_resolution.setdefault(
                f"{width}x{height}", defaultdict(int)
            )
            for key in ("tp", "fp", "fn"):
                totals[key] += metrics[key]
                bucket[key] += metrics[key]
                sequence_totals[key] += metrics[key]
        by_sequence[sequence.name] = sequence_totals
        evaluated_sequences += 1
        print(
            f"submission-eval {sequence_index}/{len(sequences)} "
            f"sequence={sequence.name} frames={len(stems)}",
            flush=True,
        )

    report = {
        "sequences": evaluated_sequences,
        "radius_pixels": args.radius,
        **summary(totals),
        "resolution_breakdown": {
            resolution: summary(values)
            for resolution, values in sorted(by_resolution.items())
        },
        "sequence_breakdown": {
            sequence: summary(values)
            for sequence, values in sorted(by_sequence.items())
        },
        "metric_status": "local point-matching proxy; not official scorer",
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
