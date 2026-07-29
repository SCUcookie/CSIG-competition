"""Evaluate local-contrast point candidates inside named image regions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from jinsight_track1.evaluation import point_metrics
from jinsight_track1.local_contrast import contrast_responses, topk_points
from jinsight_track1.postprocess import centroids


def parse_roi(value: str) -> tuple[str, tuple[int, int, int, int]]:
    name, coordinates = value.split(":", 1)
    values = tuple(int(part) for part in coordinates.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI must be NAME:X0,Y0,X1,Y1")
    return name, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("output")
    parser.add_argument("--roi", action="append", type=parse_roi, required=True)
    parser.add_argument(
        "--methods",
        default=(
            "z_dark,blackhat_5,blackhat_9,blackhat_15,"
            "temporal_dark,temporal_z_dark"
        ),
    )
    parser.add_argument("--top-k", default="1,2,3,5")
    parser.add_argument("--radius", type=float, default=2.0)
    args = parser.parse_args()

    methods = args.methods.split(",")
    counts = sorted({int(value) for value in args.top_k.split(",")})
    totals: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    sequences = sorted(Path(args.root).iterdir())
    frames_seen = 0
    for sequence in sequences:
        images = [
            cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            for path in sorted((sequence / "img").glob("*"))
        ]
        masks = sorted((sequence / "mask").glob("*"))
        stride = max(1, len(images) // 80)
        sample = np.stack(images[::stride]).astype(np.float32)
        background = np.median(sample, axis=0)
        deviation = np.maximum(sample.std(axis=0), 2.0)
        for image, mask_path in zip(images, masks):
            truth = [
                (point.x, point.y)
                for point in centroids(
                    cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE), 0.5, 1
                )
            ]
            responses = contrast_responses(image, background, deviation)
            for roi_name, (x0, y0, x1, y1) in args.roi:
                for method in methods:
                    selected = topk_points(
                        responses[method][y0:y1, x0:x1], max(counts)
                    )
                    selected = [(x + x0, y + y0) for x, y in selected]
                    for count in counts:
                        metrics = point_metrics(
                            selected[:count], truth, args.radius
                        )
                        bucket = totals[(roi_name, method, count)]
                        for key in ("tp", "fp", "fn"):
                            bucket[key] += metrics[key]
            frames_seen += 1
        print(
            f"roi-contrast {sequence.name} frames={frames_seen}",
            flush=True,
        )

    sweep = []
    for (roi, method, count), values in totals.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        sweep.append(
            {
                "roi": roi,
                "method": method,
                "top_k": count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            }
        )
    sweep.sort(
        key=lambda item: (item["f1"], item["recall"], item["precision"]),
        reverse=True,
    )
    report = {
        "sequences": len(sequences),
        "frames": frames_seen,
        "best": sweep[0],
        "sweep": sweep,
        "metric_status": "local point-matching proxy; not official scorer",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["best"], indent=2))


if __name__ == "__main__":
    main()
