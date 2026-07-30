"""Evaluate patch-candidate coverage and ranking on labelled sequences."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from jinsight_track1.deeppro_adapter import _files, _sequences
from jinsight_track1.postprocess import centroids
from jinsight_track1.tbd_patch import (
    TBDPatchClassifier,
    candidate_peaks,
    extract_patch_channels,
    spatial_dark_score,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("weights")
    parser.add_argument("--resolutions", default="640x512")
    parser.add_argument("--sequences")
    parser.add_argument("--roi", default="380,220,510,410")
    parser.add_argument("--candidates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--output")
    return parser.parse_args()


def point_counts(predicted, truth, radius):
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 2)
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    matched = 0
    if len(predicted) and len(truth):
        distances = np.linalg.norm(predicted[:, None] - truth[None, :], axis=2)
        rows, columns = linear_sum_assignment(distances)
        matched = int(
            sum(
                distances[row, column] <= radius
                for row, column in zip(rows, columns)
            )
        )
    return matched, len(predicted) - matched, len(truth) - matched


def main():
    args = parse_args()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    patch_size = int(checkpoint["patch_size"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TBDPatchClassifier().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    resolutions = set(args.resolutions.split(",")) if args.resolutions else None
    selected_sequences = set(args.sequences.split(",")) if args.sequences else None
    roi = tuple(map(int, args.roi.split(","))) if args.roi else None
    top_limits = (1, 2, 3, 5, 10, 20, 50)
    threshold_values = (-4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8)
    totals_top = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    totals_threshold = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    truth_ranks = []
    covered_truth = 0
    truth_total = 0
    sequence_reports = []

    with torch.inference_mode():
        for sequence in _sequences(Path(args.root)):
            if selected_sequences and sequence.name not in selected_sequences:
                continue
            images = _files(sequence / "img")
            masks = _files(sequence / "mask")
            stems = sorted(set(images) & set(masks))
            if not stems:
                continue
            width, height = Image.open(images[stems[0]]).size
            resolution = f"{width}x{height}"
            if resolutions and resolution not in resolutions:
                continue
            frame_records = []
            for stem in stems:
                frame = np.asarray(Image.open(images[stem]).convert("L"), dtype=np.uint8)
                mask = np.asarray(Image.open(masks[stem]).convert("L"))
                truth = [(point.x, point.y) for point in centroids(mask, 0.5, 1)]
                score = spatial_dark_score(frame)
                points = candidate_peaks(score, args.candidates, roi=roi)
                features = extract_patch_channels(frame, score, points, patch_size)
                frame_logits = []
                for start in range(0, len(features), args.batch_size):
                    batch = torch.from_numpy(
                        features[start : start + args.batch_size]
                    ).to(device)
                    frame_logits.append(model(batch).cpu().numpy())
                frame_records.append(
                    {
                        "points": points,
                        "truth": truth,
                        "logits": (
                            np.concatenate(frame_logits)
                            if frame_logits
                            else np.zeros(0)
                        ),
                    }
                )
            sequence_top = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            sequence_covered = sequence_truth = 0
            for record in frame_records:
                points = record["points"]
                frame_logits = record["logits"]
                order = np.argsort(frame_logits)[::-1]
                truth = record["truth"]
                truth_total += len(truth)
                sequence_truth += len(truth)
                if len(points) and len(truth):
                    distance = np.linalg.norm(
                        points[:, None] - np.asarray(truth)[None, :, :], axis=2
                    )
                    for truth_index in range(len(truth)):
                        matching = np.where(distance[:, truth_index] <= args.radius)[0]
                        if not len(matching):
                            continue
                        covered_truth += 1
                        sequence_covered += 1
                        best_index = matching[np.argmax(frame_logits[matching])]
                        rank = int(np.where(order == best_index)[0][0]) + 1
                        truth_ranks.append(rank)
                for limit in top_limits:
                    tp, fp, fn = point_counts(points[order[:limit]], truth, args.radius)
                    for bucket in (totals_top[limit], sequence_top[limit]):
                        bucket["tp"] += tp
                        bucket["fp"] += fp
                        bucket["fn"] += fn
                for threshold in threshold_values:
                    selected = points[frame_logits >= threshold]
                    tp, fp, fn = point_counts(selected, truth, args.radius)
                    bucket = totals_threshold[threshold]
                    bucket["tp"] += tp
                    bucket["fp"] += fp
                    bucket["fn"] += fn
            sequence_reports.append(
                {
                    "sequence": sequence.name,
                    "frames": len(stems),
                    "truth": sequence_truth,
                    "covered": sequence_covered,
                    "coverage": sequence_covered / max(1, sequence_truth),
                    "top": {
                        str(limit): values for limit, values in sequence_top.items()
                    },
                }
            )
            print(
                f"patch-eval {sequence.name}: coverage="
                f"{sequence_covered / max(1, sequence_truth):.4f}",
                flush=True,
            )

    def metrics(rows):
        output = []
        for setting, values in rows.items():
            tp, fp, fn = values["tp"], values["fp"], values["fn"]
            output.append(
                {
                    "setting": setting,
                    **values,
                    "precision": tp / max(1, tp + fp),
                    "recall": tp / max(1, tp + fn),
                    "f1": 2 * tp / max(1, 2 * tp + fp + fn),
                }
            )
        return output

    report = {
        "settings": vars(args),
        "truth": truth_total,
        "covered": covered_truth,
        "coverage": covered_truth / max(1, truth_total),
        "truth_rank": {
            "p50": float(np.quantile(truth_ranks, 0.5)) if truth_ranks else None,
            "p75": float(np.quantile(truth_ranks, 0.75)) if truth_ranks else None,
            "p90": float(np.quantile(truth_ranks, 0.9)) if truth_ranks else None,
            "p95": float(np.quantile(truth_ranks, 0.95)) if truth_ranks else None,
        },
        "top": metrics(totals_top),
        "threshold": metrics(totals_threshold),
        "sequences": sequence_reports,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
