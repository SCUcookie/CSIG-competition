"""Evaluate temporal centering of dense supervised patch-response maps."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from jinsight_track1.deeppro_adapter import _files
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.postprocess import centroids
from jinsight_track1.tbd_patch import (
    TBDPatchClassifier,
    extract_patch_channels,
    spatial_score,
)
from evaluate_tbd_shift_stack import evaluate, oracle, search, select_candidates


def dense_response(model, device, frame, roi, patch_size, score_mode, batch_size):
    x0, y0, x1, y1 = roi
    xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    points = np.column_stack((xs.ravel(), ys.ravel()))
    score = spatial_score(frame, score_mode)
    features = extract_patch_channels(frame, score, points, patch_size)
    values = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            values.append(model(batch).cpu().numpy())
    return np.concatenate(values).reshape(y1 - y0, x1 - x0)


def top_points(score, count, roi):
    maximum = cv2.dilate(score, np.ones((3, 3), dtype=np.uint8))
    ys, xs = np.where(score >= maximum - 1e-6)
    values = score[ys, xs]
    order = np.argsort(values)[::-1][:count]
    return [(float(xs[index] + roi[0]), float(ys[index] + roi[1])) for index in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    parser.add_argument("weights")
    parser.add_argument("output")
    parser.add_argument("--roi", default="380,220,510,410")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--shift-stack", action="store_true")
    parser.add_argument("--block-size", type=int, default=8)
    args = parser.parse_args()

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    patch_size = int(checkpoint["patch_size"])
    score_mode = checkpoint.get("settings", {}).get("score_mode", "dark")
    device = torch.device(args.device)
    model = TBDPatchClassifier().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    roi = tuple(map(int, args.roi.split(",")))

    sequence = Path(args.sequence)
    images, masks = _files(sequence / "img"), _files(sequence / "mask")
    stems = sorted(set(images) & set(masks), key=lambda value: int(value))
    responses = []
    truths = []
    for index, stem in enumerate(stems, 1):
        frame = np.asarray(Image.open(images[stem]).convert("L"), dtype=np.uint8)
        responses.append(
            dense_response(
                model, device, frame, roi, patch_size, score_mode, args.batch_size
            ).astype(np.float16)
        )
        mask = np.asarray(Image.open(masks[stem]).convert("L"))
        truths.append([(point.x, point.y) for point in centroids(mask)])
        if index % 25 == 0:
            print(f"dense-response {index}/{len(stems)}", flush=True)
    stack = np.asarray(responses, dtype=np.float32)
    median = np.median(stack, axis=0)
    mean = np.mean(stack, axis=0)
    modes = {
        "raw": stack,
        "median_centered": stack - median,
        "mean_centered": stack - mean,
    }
    limits = (1, 2, 5, 10, 20, 50, 100, 200)
    report = {}
    for mode, values in modes.items():
        rows = []
        for limit in limits:
            total = defaultdict(int)
            for score, truth in zip(values, truths):
                metrics = point_metrics(
                    top_points(score, limit, roi), truth, args.radius
                )
                for key in ("tp", "fp", "fn"):
                    total[key] += metrics[key]
            tp, fp, fn = total["tp"], total["fp"], total["fn"]
            rows.append(
                {
                    "top_k": limit,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": tp / max(1, tp + fp),
                    "recall": tp / max(1, tp + fn),
                    "f1": 2 * tp / max(1, 2 * tp + fp + fn),
                }
            )
        report[mode] = rows
        print(mode, max(rows, key=lambda row: row["f1"]), flush=True)
    shift_report = {}
    if args.shift_stack:
        local_truths = [
            [(x - roi[0], y - roi[1]) for x, y in frame] for frame in truths
        ]
        for mode in ("median_centered", "mean_centered"):
            values = modes[mode]
            block_maps = []
            block_times = []
            for start in range(0, len(values), args.block_size):
                end = min(len(values), start + args.block_size)
                block_maps.append(values[start:end].mean(axis=0))
                block_times.append((start + end - 1) / 2)
            raw_candidates = search(
                block_maps,
                np.asarray(block_times),
                (0, 0, roi[2] - roi[0], roi[3] - roi[1]),
                np.arange(-0.03, 0.0201, 0.004),
                np.arange(-0.02, 0.0501, 0.004),
                0.003,
                5,
                3,
                "mean",
            )
            selected = select_candidates(raw_candidates, 20, 2.0, len(stems))
            evaluation = evaluate(selected, local_truths, args.radius)
            shift_report[mode] = {
                "oracle": oracle(local_truths, block_maps, np.asarray(block_times)),
                "candidates": [
                    {
                        "rank": index,
                        "midpoint": candidate["midpoint"].tolist(),
                        "velocity": candidate["velocity"].tolist(),
                        "score": candidate["score"],
                    }
                    for index, candidate in enumerate(selected, 1)
                ],
                "evaluation": evaluation,
            }
            print(
                f"shift-stack {mode}",
                max(evaluation, key=lambda row: row["f1"]),
                flush=True,
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "settings": vars(args),
                "frames": len(stems),
                "truth_points": sum(map(len, truths)),
                "modes": report,
                "shift_stack": shift_report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
