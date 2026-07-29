"""Evaluate many submission directories while loading validation masks once."""
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
    parser.add_argument("output_dir")
    parser.add_argument("candidates", nargs="+", help="LABEL=SUBMISSION_DIR")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--coordinate-order", choices=["xy", "yx"], default="xy")
    args = parser.parse_args()

    candidate_dirs = {
        label: Path(path)
        for label, path in (value.split("=", 1) for value in args.candidates)
    }
    totals = {
        label: defaultdict(int) for label in candidate_dirs
    }
    by_sequence = {
        label: {} for label in candidate_dirs
    }
    sequences = _sequences(Path(args.val_root))
    for sequence_index, sequence in enumerate(sequences, 1):
        masks = _files(sequence / "mask")
        stems = sorted(masks, key=lambda value: natural_key(Path(value)))
        expected_frames = set(range(1, len(stems) + 1))
        predictions = {
            label: parse(
                (directory / f"{sequence.name}.txt").read_text(encoding="ascii"),
                sequence.name,
                args.coordinate_order,
            )
            for label, directory in candidate_dirs.items()
        }
        for label, prediction in predictions.items():
            if set(prediction.frames) != expected_frames:
                raise ValueError(f"frame mismatch for {label}/{sequence.name}")
        sequence_totals = {
            label: defaultdict(int) for label in candidate_dirs
        }
        for frame_index, stem in enumerate(stems, 1):
            mask = np.asarray(Image.open(masks[stem]).convert("L"))
            height, width = mask.shape
            truth = [(point.x, point.y) for point in centroids(mask, 0.5, 1)]
            for label, prediction in predictions.items():
                predicted = [
                    (point.x, point.y)
                    for point in prediction.frames[frame_index]
                ]
                if any(
                    not (0 <= x < width and 0 <= y < height)
                    for x, y in predicted
                ):
                    raise ValueError(
                        f"out-of-bounds point in {label}/{sequence.name} "
                        f"frame {frame_index}"
                    )
                metrics = point_metrics(predicted, truth, args.radius)
                for key in ("tp", "fp", "fn"):
                    totals[label][key] += metrics[key]
                    sequence_totals[label][key] += metrics[key]
        for label in candidate_dirs:
            by_sequence[label][sequence.name] = sequence_totals[label]
        print(
            f"candidate-eval {sequence_index}/{len(sequences)} "
            f"sequence={sequence.name} frames={len(stems)}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, directory in candidate_dirs.items():
        report = {
            "candidate": label,
            "submission": str(directory),
            "sequences": len(sequences),
            "radius_pixels": args.radius,
            **summary(totals[label]),
            "sequence_breakdown": {
                sequence: summary(values)
                for sequence, values in sorted(by_sequence[label].items())
            },
            "metric_status": "local point-matching proxy; not official scorer",
        }
        (output_dir / f"{label}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                label: summary(values)
                for label, values in sorted(totals.items())
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
