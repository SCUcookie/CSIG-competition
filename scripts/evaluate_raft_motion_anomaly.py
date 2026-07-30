#!/usr/bin/env python3
"""Evaluate dense-flow-compensated temporal anomalies as a detector/complement.

This is deliberately a diagnostic route: it measures whether a pretrained
optical-flow prior exposes targets that the current high-precision submission
misses before any trainable fusion model is built.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

from jinsight_track1.data import natural_key
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.postprocess import centroids
from jinsight_track1.submission import parse


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("val_root")
    parser.add_argument("output_json")
    parser.add_argument("--baseline-dir")
    parser.add_argument("--device", default="0")
    parser.add_argument("--resolution", default="256x256")
    parser.add_argument("--lags", default="2,4,8")
    parser.add_argument("--topk", default="1,2,3,5,10,20")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--suppression-radius", type=float, default=3.0)
    parser.add_argument("--peak-distance", type=int, default=3)
    parser.add_argument("--blur-sigma", type=float, default=0.7)
    return parser.parse_args()


def files(directory: Path) -> dict[str, Path]:
    return {item.stem: item for item in directory.iterdir() if item.is_file()}


def load_sequence(sequence: Path, max_frames: int | None):
    images = files(sequence / "img")
    stems = sorted(images, key=lambda value: natural_key(Path(value)))
    if max_frames:
        stems = stems[:max_frames]
    frames = np.stack(
        [np.asarray(Image.open(images[stem]).convert("L"), dtype=np.uint8) for stem in stems]
    )
    return stems, frames


def warp_neighbor(neighbor: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample a neighboring frame at positions reached from the current frame."""
    batch, _, height, width = neighbor.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=neighbor.device),
        torch.arange(width, device=neighbor.device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).float()[None].expand(batch, -1, -1, -1)
    grid = grid + flow.permute(0, 2, 3, 1)
    grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    return F.grid_sample(
        neighbor, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def motion_response(
    model: torch.nn.Module,
    frames: np.ndarray,
    lags: list[int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    count, height, width = frames.shape
    pairs = []
    for index in range(count):
        for lag in lags:
            if index - lag >= 0:
                pairs.append((index, index - lag, lag, -1))
            if index + lag < count:
                pairs.append((index, index + lag, lag, 1))

    differences: dict[tuple[int, int, int], np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            group = pairs[start : start + batch_size]
            current = np.stack([frames[a] for a, _, _, _ in group])
            neighbor = np.stack([frames[b] for _, b, _, _ in group])
            current_tensor = torch.from_numpy(current).float().to(device)[:, None] / 255.0
            neighbor_tensor = torch.from_numpy(neighbor).float().to(device)[:, None] / 255.0
            current_rgb = current_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
            neighbor_rgb = neighbor_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
            flow = model(current_rgb, neighbor_rgb)[-1]
            reconstructed = warp_neighbor(neighbor_tensor, flow)
            delta = (current_tensor - reconstructed)[:, 0].float().cpu().numpy()
            for key, value in zip(group, delta):
                index, _, lag, direction = key
                differences[(index, lag, direction)] = value

    response = np.zeros((count, height, width), dtype=np.float32)
    for index in range(count):
        lag_responses = []
        for lag in lags:
            previous = differences.get((index, lag, -1))
            following = differences.get((index, lag, 1))
            if previous is not None and following is not None:
                same_polarity = previous * following > 0
                agreed = np.minimum(np.abs(previous), np.abs(following))
                lag_responses.append(np.where(same_polarity, agreed, 0.0))
            elif previous is not None:
                lag_responses.append(np.abs(previous))
            elif following is not None:
                lag_responses.append(np.abs(following))
        if lag_responses:
            response[index] = np.maximum.reduce(lag_responses)
    return response


def standardized_peaks(
    response: np.ndarray, distance: int, blur_sigma: float, limit: int
) -> tuple[list[tuple[float, float]], np.ndarray]:
    score = gaussian_filter(response.astype(np.float32), blur_sigma)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    score = (score - median) / max(1.4826 * mad, 1e-6)
    local = score == maximum_filter(score, size=2 * distance + 1, mode="nearest")
    ys, xs = np.where(local)
    if not len(xs):
        return [], score
    order = np.argsort(score[ys, xs])[::-1][:limit]
    return [(float(xs[i]), float(ys[i])) for i in order], score


def add_counts(destination: dict, values: dict) -> None:
    for key in ("tp", "fp", "fn"):
        destination[key] += int(values[key])


def summarize(values: dict) -> dict:
    tp, fp, fn = (int(values[key]) for key in ("tp", "fp", "fn"))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def main() -> None:
    args = arguments()
    width, height = (int(value) for value in args.resolution.split("x"))
    lags = sorted({int(value) for value in args.lags.split(",") if int(value) > 0})
    topk = sorted({int(value) for value in args.topk.split(",") if int(value) > 0})
    root = Path(args.val_root)
    sequences = []
    for sequence in sorted((p for p in root.iterdir() if p.is_dir()), key=natural_key):
        image_paths = sorted((sequence / "img").glob("*"), key=natural_key)
        if image_paths and Image.open(image_paths[0]).size == (width, height):
            sequences.append(sequence)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device).eval()
    totals = {value: defaultdict(int) for value in topk}
    union_totals = {value: defaultdict(int) for value in topk}
    baseline_totals = defaultdict(int)
    target_ranks = []
    missed_target_ranks = []
    sequence_rows = []
    baseline_root = Path(args.baseline_dir) if args.baseline_dir else None

    for sequence_index, sequence in enumerate(sequences, 1):
        stems, frames = load_sequence(sequence, args.max_frames)
        masks = files(sequence / "mask")
        response = motion_response(model, frames, lags, args.batch_size, device)
        baseline = None
        if baseline_root:
            baseline = parse(
                (baseline_root / f"{sequence.name}.txt").read_text(encoding="ascii"),
                sequence.name,
            )
        sequence_counts = {value: defaultdict(int) for value in topk}
        sequence_union = {value: defaultdict(int) for value in topk}
        for frame_index, stem in enumerate(stems):
            truth = [
                (point.x, point.y)
                for point in centroids(
                    np.asarray(Image.open(masks[stem]).convert("L")), 0.5, 1
                )
            ]
            peaks, score = standardized_peaks(
                response[frame_index],
                args.peak_distance,
                args.blur_sigma,
                max(topk),
            )
            existing = []
            if baseline is not None:
                existing = [
                    (point.x, point.y)
                    for point in baseline.frames.get(frame_index + 1, [])
                ]
                add_counts(baseline_totals, point_metrics(existing, truth, args.radius))
            for x, y in truth:
                distances = [np.hypot(px - x, py - y) for px, py in peaks]
                matching = [index + 1 for index, value in enumerate(distances) if value <= args.radius]
                rank = min(matching) if matching else None
                target_ranks.append(rank)
                if baseline is not None and all(
                    np.hypot(old_x - x, old_y - y) > args.radius
                    for old_x, old_y in existing
                ):
                    missed_target_ranks.append(rank)
            for value in topk:
                predicted = peaks[:value]
                metrics = point_metrics(predicted, truth, args.radius)
                add_counts(totals[value], metrics)
                add_counts(sequence_counts[value], metrics)
                if baseline is not None:
                    additions = [
                        point
                        for point in predicted
                        if all(
                            np.hypot(point[0] - old[0], point[1] - old[1])
                            > args.suppression_radius
                            for old in existing
                        )
                    ]
                    union = point_metrics(existing + additions, truth, args.radius)
                    add_counts(union_totals[value], union)
                    add_counts(sequence_union[value], union)
        row = {
            "sequence": sequence.name,
            "frames": len(stems),
            "standalone": {str(k): summarize(v) for k, v in sequence_counts.items()},
        }
        if baseline is not None:
            row["union"] = {str(k): summarize(v) for k, v in sequence_union.items()}
        sequence_rows.append(row)
        partial = {str(k): summarize(v)["f1"] for k, v in totals.items()}
        print(
            f"raft-motion {sequence_index}/{len(sequences)} {sequence.name} "
            f"standalone_f1={partial}",
            flush=True,
        )

    finite_ranks = [value for value in target_ranks if value is not None]
    report = {
        "method": "RAFT-small bidirectional flow-compensated temporal anomaly",
        "resolution": args.resolution,
        "lags": lags,
        "sequences": len(sequences),
        "frames": sum(row["frames"] for row in sequence_rows),
        "standalone": {str(k): summarize(v) for k, v in totals.items()},
        "baseline": summarize(baseline_totals) if baseline_root else None,
        "union": (
            {str(k): summarize(v) for k, v in union_totals.items()}
            if baseline_root
            else None
        ),
        "target_peak_rank": {
            "targets": len(target_ranks),
            "covered_topk": len(finite_ranks),
            "median": float(np.median(finite_ranks)) if finite_ranks else None,
            "p90": float(np.percentile(finite_ranks, 90)) if finite_ranks else None,
        },
        "missed_target_peak_rank": None,
        "by_sequence": sequence_rows,
    }
    if baseline_root:
        finite_missed = [value for value in missed_target_ranks if value is not None]
        report["missed_target_peak_rank"] = {
            "targets": len(missed_target_ranks),
            "covered_topk": len(finite_missed),
            "median": float(np.median(finite_missed)) if finite_missed else None,
            "p90": (
                float(np.percentile(finite_missed, 90)) if finite_missed else None
            ),
        }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "by_sequence"}))


if __name__ == "__main__":
    main()
