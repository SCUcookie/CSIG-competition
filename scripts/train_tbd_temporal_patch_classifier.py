"""Train a multi-frame hard-negative patch classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jinsight_track1.deeppro_adapter import _files, _sequences
from jinsight_track1.postprocess import centroids
from jinsight_track1.tbd_patch import (
    TBDPatchClassifier,
    candidate_peaks,
    extract_patch_channels,
    spatial_score,
)


def temporal_features(frames, scores, points, patch_size):
    return np.concatenate(
        [
            extract_patch_channels(frame, score, points, patch_size)
            for frame, score in zip(frames, scores)
        ],
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root")
    parser.add_argument("output")
    parser.add_argument("--resolution", default="640x512")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--temporal-radius", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=21)
    parser.add_argument("--negative-candidates", type=int, default=200)
    parser.add_argument("--negatives-per-frame", type=int, default=24)
    parser.add_argument("--positive-jitter", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    positives, negatives = [], []
    for sequence in _sequences(Path(args.train_root)):
        images, masks = _files(sequence / "img"), _files(sequence / "mask")
        stems = sorted(set(images) & set(masks), key=lambda value: int(value))
        if not stems:
            continue
        width, height = Image.open(images[stems[0]]).size
        if f"{width}x{height}" != args.resolution:
            continue
        cache = {}
        for center in range(0, len(stems), args.stride):
            indices = np.clip(
                np.arange(
                    center - args.temporal_radius,
                    center + args.temporal_radius + 1,
                ),
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
            mask = np.asarray(Image.open(masks[stems[center]]).convert("L"))
            truth = np.asarray(
                [(point.x, point.y) for point in centroids(mask)],
                dtype=float,
            ).reshape(-1, 2)
            positive_points = []
            for x, y in truth:
                for dy in range(-args.positive_jitter, args.positive_jitter + 1):
                    for dx in range(-args.positive_jitter, args.positive_jitter + 1):
                        positive_points.append(
                            (
                                int(np.clip(round(x) + dx, 0, width - 1)),
                                int(np.clip(round(y) + dy, 0, height - 1)),
                            )
                        )
            if positive_points:
                positives.append(
                    temporal_features(
                        frames, scores, np.asarray(positive_points), args.patch_size
                    )
                )
            candidates = candidate_peaks(scores[args.temporal_radius], args.negative_candidates)
            if len(truth) and len(candidates):
                distance = np.linalg.norm(
                    candidates[:, None] - truth[None, :, :], axis=2
                )
                candidates = candidates[np.min(distance, axis=1) >= 6]
            if len(candidates):
                hard = candidates[: min(args.negatives_per_frame, len(candidates))]
                negatives.append(
                    temporal_features(frames, scores, hard, args.patch_size)
                )
            for old in list(cache):
                if old < center - args.temporal_radius - 2:
                    del cache[old]
        print(
            f"temporal-patch-data {sequence.name} "
            f"positive_chunks={len(positives)} negative_chunks={len(negatives)}",
            flush=True,
        )
    positive, negative = np.concatenate(positives), np.concatenate(negatives)
    print(f"samples positive={len(positive)} negative={len(negative)}", flush=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    channels = 2 * (2 * args.temporal_radius + 1)
    model = TBDPatchClassifier(channels=channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    positive, negative = torch.from_numpy(positive), torch.from_numpy(negative)
    history = []
    for epoch in range(1, args.epochs + 1):
        count = min(len(positive), len(negative))
        features = torch.cat(
            (
                positive[torch.randperm(len(positive))[:count]],
                negative[torch.randperm(len(negative))[:count]],
            )
        )
        labels = torch.cat((torch.ones(count), torch.zeros(count)))
        order = torch.randperm(len(labels))
        loader = DataLoader(
            TensorDataset(features[order], labels[order]),
            batch_size=args.batch_size,
            pin_memory=True,
        )
        model.train()
        loss_total = correct = seen = 0
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            loss_total += float(loss) * len(batch)
            correct += int(((logits >= 0) == (target >= 0.5)).sum())
            seen += len(batch)
        row = {"epoch": epoch, "loss": loss_total / seen, "accuracy": correct / seen}
        history.append(row)
        print(json.dumps(row), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "patch_size": args.patch_size,
            "temporal_radius": args.temporal_radius,
            "settings": vars(args),
            "history": history,
        },
        output,
    )


if __name__ == "__main__":
    main()
