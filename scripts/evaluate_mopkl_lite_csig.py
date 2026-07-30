#!/usr/bin/env python3
"""Evaluate a detection-supervised MoPKL visual checkpoint."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_tdcnet_csig import decode, metrics
from train_mopkl_lite_csig import MoPKLLite, prepare_inputs
from train_tdcnet_csig import CSIGTDCSequenceDataset, collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("weights")
    parser.add_argument("val_root")
    parser.add_argument("output")
    parser.add_argument("--resolution", default="256x256")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument(
        "--thresholds",
        default="0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,0.5",
    )
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--max-sequences", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    width, height = (int(value) for value in args.resolution.split("x"))
    thresholds = [float(value) for value in args.thresholds.split(",")]
    dataset = CSIGTDCSequenceDataset(
        args.val_root, (width, height), context=2, box_size=12.0,
        augment=False, max_sequences=args.max_sequences,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate,
        persistent_workers=args.workers > 0,
    )
    device = torch.device(f"cuda:{args.device}")
    model = MoPKLLite(args.source_root)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.to(device).eval()
    scale = args.input_size / width
    totals = {threshold: defaultdict(int) for threshold in thresholds}
    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader, 1):
            images = prepare_inputs(
                images.to(device, non_blocking=True), args.input_size
            )
            output = model(images)[0]
            decoded = decode(
                output, stride=8, min_conf=min(thresholds),
                max_det=args.max_det,
            )
            for points, target in zip(decoded, targets):
                scored = points.detach().cpu().numpy()
                scored[:, :2] /= scale
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
