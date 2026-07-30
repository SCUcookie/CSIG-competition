"""Evaluate a temporal patch classifier on a dense stride grid."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from jinsight_track1.deeppro_adapter import _files
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.postprocess import centroids
from jinsight_track1.tbd_patch import TBDPatchClassifier, spatial_score
from train_tbd_temporal_patch_classifier import temporal_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    parser.add_argument("weights")
    parser.add_argument("output")
    parser.add_argument("--roi", default="380,220,510,410")
    parser.add_argument("--grid-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radius", type=float, default=2)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    patch_size = int(checkpoint["patch_size"])
    temporal_radius = int(checkpoint["temporal_radius"])
    channels = 2 * (2 * temporal_radius + 1)
    device = torch.device(args.device)
    model = TBDPatchClassifier(channels=channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    roi = tuple(map(int, args.roi.split(",")))
    sequence = Path(args.sequence)
    images, masks = _files(sequence / "img"), _files(sequence / "mask")
    stems = sorted(set(images) & set(masks), key=lambda value: int(value))
    if args.max_frames:
        stems = stems[: args.max_frames]
    x_values = np.arange(roi[0], roi[2], args.grid_stride)
    y_values = np.arange(roi[1], roi[3], args.grid_stride)
    xs, ys = np.meshgrid(x_values, y_values)
    points = np.column_stack((xs.ravel(), ys.ravel()))
    limits = (1, 2, 5, 10, 20, 50, 100)
    totals = {limit: defaultdict(int) for limit in limits}
    truth_total = covered = 0
    cache = {}
    with torch.inference_mode():
        for center, stem in enumerate(stems):
            indices = np.clip(
                np.arange(center - temporal_radius, center + temporal_radius + 1),
                0,
                len(stems) - 1,
            )
            frames, scores = [], []
            for index in indices:
                index = int(index)
                if index not in cache:
                    frame = np.asarray(
                        Image.open(images[stems[index]]).convert("L"), dtype=np.uint8
                    )
                    cache[index] = (frame, spatial_score(frame, "absolute"))
                frame, score = cache[index]
                frames.append(frame)
                scores.append(score)
            features = temporal_features(frames, scores, points, patch_size)
            logits = []
            for start in range(0, len(features), args.batch_size):
                batch = torch.from_numpy(features[start : start + args.batch_size]).to(device)
                logits.append(model(batch).cpu().numpy())
            logits = np.concatenate(logits)
            truth_mask = np.asarray(Image.open(masks[stem]).convert("L"))
            truth = [(point.x, point.y) for point in centroids(truth_mask)]
            truth_total += len(truth)
            if truth:
                distance = np.linalg.norm(points[:, None] - np.asarray(truth)[None], axis=2)
                covered += int((distance.min(axis=0) <= args.radius).sum())
            order = np.argsort(logits)[::-1]
            for limit in limits:
                metrics = point_metrics(points[order[:limit]], truth, args.radius)
                for key in ("tp", "fp", "fn"):
                    totals[limit][key] += metrics[key]
            if (center + 1) % 25 == 0:
                print(f"temporal-patch-eval {center + 1}/{len(stems)}", flush=True)
    rows = []
    for limit, values in totals.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
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
    report = {
        "settings": vars(args),
        "frames": len(stems),
        "truth": truth_total,
        "covered": covered,
        "coverage": covered / max(1, truth_total),
        "top": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
