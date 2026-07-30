#!/usr/bin/env python3
"""Evaluate a CSIG-adapted TDCNet checkpoint with point-radius F1."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torchvision.ops import nms

from train_tdcnet_csig import (
    CSIGTDCSequenceDataset,
    collate,
)


def metrics(predicted, truth, radius):
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 2)
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    matches = 0
    if len(predicted) and len(truth):
        distances = np.linalg.norm(predicted[:, None] - truth[None, :], axis=2)
        rows, columns = linear_sum_assignment(distances)
        matches = sum(
            distances[row, column] <= radius
            for row, column in zip(rows, columns)
        )
    return matches, len(predicted) - matches, len(truth) - matches


def decode(output, stride=8, min_conf=1e-4, max_det=50):
    batch, _, height, width = output.shape
    prediction = output.flatten(2).permute(0, 2, 1)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=output.device),
        torch.arange(width, device=output.device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2)
    centres = (prediction[..., :2] + grid) * stride
    wh = prediction[..., 2:4].exp() * stride
    scores = prediction[..., 4].sigmoid() * prediction[..., 5].sigmoid()
    result = []
    for sample in range(batch):
        keep = scores[sample] >= min_conf
        xy = centres[sample][keep]
        size = wh[sample][keep]
        sample_scores = scores[sample][keep]
        boxes = torch.cat((xy - size / 2, xy + size / 2), dim=1)
        selected = nms(boxes, sample_scores, 0.3)[:max_det]
        result.append(torch.cat((xy[selected], sample_scores[selected, None]), 1))
    return result


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
        "--thresholds",
        default="0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,0.5",
    )
    parser.add_argument("--max-det", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    width, height = (int(value) for value in args.resolution.split("x"))
    thresholds = [float(value) for value in args.thresholds.split(",")]
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    from model.TDCNet.TDCNetwork import TDCNetwork

    dataset = CSIGTDCSequenceDataset(
        args.val_root, (width, height), args.context, 12.0, False
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate,
        persistent_workers=args.workers > 0,
    )
    device = torch.device(f"cuda:{args.device}")
    model = TDCNetwork(1, num_frame=args.context)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.to(device).eval()
    totals = {threshold: defaultdict(int) for threshold in thresholds}
    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader, 1):
            output = model(images.to(device, non_blocking=True))[0]
            decoded = decode(
                output, min_conf=min(thresholds), max_det=args.max_det
            )
            for points, target in zip(decoded, targets):
                scored = points.detach().cpu().numpy()
                truth = target[:, :2].numpy()
                for threshold in thresholds:
                    predicted = scored[scored[:, 2] >= threshold, :2]
                    tp, fp, fn = metrics(predicted, truth, args.radius)
                    totals[threshold]["tp"] += tp
                    totals[threshold]["fp"] += fp
                    totals[threshold]["fn"] += fn
            if batch_index % 100 == 0:
                print(f"batches={batch_index}/{len(loader)}", flush=True)
    sweep = []
    for threshold in thresholds:
        tp, fp, fn = (totals[threshold][key] for key in ("tp", "fp", "fn"))
        sweep.append({
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
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
