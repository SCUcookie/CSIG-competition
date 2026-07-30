#!/usr/bin/env python3
"""Evaluate the TDC pixel heatmap route with point-radius F1."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from train_tdc_heatmap_csig import TDCHeatmap
from train_tdcnet_csig import CSIGTDCSequenceDataset, collate


def point_counts(predicted, truth, radius):
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 2)
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    matched = 0
    if len(predicted) and len(truth):
        distance = np.linalg.norm(predicted[:, None] - truth[None, :], axis=2)
        rows, columns = linear_sum_assignment(distance)
        matched = int(sum(
            distance[row, column] <= radius
            for row, column in zip(rows, columns)
        ))
    return matched, len(predicted) - matched, len(truth) - matched


def heatmap_points(logits, max_det):
    probability = logits.sigmoid()
    maxima = probability.eq(F.max_pool2d(probability, 3, stride=1, padding=1))
    score = (probability * maxima).flatten(1)
    values, indices = score.topk(min(max_det, score.shape[1]), dim=1)
    width = logits.shape[-1]
    x = (indices % width).float()
    y = torch.div(indices, width, rounding_mode="floor").float()
    return torch.stack((x, y, values), dim=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("weights")
    parser.add_argument("val_root")
    parser.add_argument("output")
    parser.add_argument("--resolution", default="256x256")
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument(
        "--thresholds", default="0.01,0.03,0.1,0.2,0.3,0.5,0.7,0.9"
    )
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--max-sequences", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    width, height = (int(value) for value in args.resolution.split("x"))
    thresholds = [float(value) for value in args.thresholds.split(",")]
    dataset = CSIGTDCSequenceDataset(
        args.val_root, (width, height), args.context, 12.0, False,
        args.max_sequences,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate,
        persistent_workers=args.workers > 0,
    )
    device = torch.device(f"cuda:{args.device}")
    model = TDCHeatmap(args.source_root, args.context)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.to(device).eval()
    totals = {threshold: defaultdict(int) for threshold in thresholds}
    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader, 1):
            points = heatmap_points(
                model(images.to(device, non_blocking=True)), args.max_det
            ).cpu().numpy()
            for scored, target in zip(points, targets):
                truth = target[:, :2].numpy()
                for threshold in thresholds:
                    predicted = scored[scored[:, 2] >= threshold, :2]
                    tp, fp, fn = point_counts(predicted, truth, args.radius)
                    totals[threshold]["tp"] += int(tp)
                    totals[threshold]["fp"] += int(fp)
                    totals[threshold]["fn"] += int(fn)
            if batch_index % 100 == 0:
                print(f"batches={batch_index}/{len(loader)}", flush=True)
    sweep = []
    for threshold in thresholds:
        tp, fp, fn = (totals[threshold][key] for key in ("tp", "fp", "fn"))
        sweep.append({
            "threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        })
    report = {
        "weights": args.weights,
        "frames": len(dataset),
        "sequences": len(dataset.sequences),
        "radius": args.radius,
        "sweep": sweep,
        "best": max(sweep, key=lambda row: (row["f1"], row["recall"])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
