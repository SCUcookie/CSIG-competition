"""Train a balanced hard-negative patch classifier on labelled 640 imagery."""
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root")
    parser.add_argument("output")
    parser.add_argument("--resolution", default="640x512")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=21)
    parser.add_argument("--negative-candidates", type=int, default=80)
    parser.add_argument("--negatives-per-frame", type=int, default=12)
    parser.add_argument("--min-positive-score", type=float, default=0.0)
    parser.add_argument("--positive-jitter", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--score-mode", choices=("dark", "absolute"), default="dark")
    parser.add_argument("--polarity-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    positives = []
    negatives = []
    frames_seen = 0
    for sequence in _sequences(Path(args.train_root)):
        images = _files(sequence / "img")
        masks = _files(sequence / "mask")
        stems = sorted(set(images) & set(masks))
        if not stems:
            continue
        width, height = Image.open(images[stems[0]]).size
        if f"{width}x{height}" != args.resolution:
            continue
        for stem in stems[:: args.stride]:
            frame = np.asarray(Image.open(images[stem]).convert("L"), dtype=np.uint8)
            mask = np.asarray(Image.open(masks[stem]).convert("L"))
            truth = np.asarray(
                [(point.x, point.y) for point in centroids(mask, 0.5, 1)],
                dtype=float,
            ).reshape(-1, 2)
            score = spatial_score(frame, args.score_mode)
            candidates = candidate_peaks(score, args.negative_candidates)
            dark_truth = []
            for x, y in truth:
                ix, iy = int(round(x)), int(round(y))
                if score[iy, ix] >= args.min_positive_score:
                    aligned = np.asarray([ix, iy])
                    if len(candidates):
                        distances = np.linalg.norm(candidates - aligned, axis=1)
                        nearby = np.where(distances <= 2.0)[0]
                        if len(nearby):
                            aligned = candidates[nearby[0]]
                    for offset_y in range(
                        -args.positive_jitter, args.positive_jitter + 1
                    ):
                        for offset_x in range(
                            -args.positive_jitter, args.positive_jitter + 1
                        ):
                            dark_truth.append(
                                (
                                    int(np.clip(aligned[0] + offset_x, 0, width - 1)),
                                    int(np.clip(aligned[1] + offset_y, 0, height - 1)),
                                )
                            )
            if dark_truth:
                positives.append(
                    extract_patch_channels(
                        frame, score, np.asarray(dark_truth), args.patch_size
                    )
                )
            if len(truth) and len(candidates):
                distance = np.linalg.norm(
                    candidates[:, None] - truth[None, :, :], axis=2
                )
                candidates = candidates[np.min(distance, axis=1) >= 6]
            if len(candidates):
                count = min(args.negatives_per_frame, len(candidates))
                # Keep most hard negatives, with a little random diversity.
                hard_count = min(count * 3 // 4, len(candidates))
                selected = list(range(hard_count))
                if count > hard_count and len(candidates) > hard_count:
                    selected.extend(
                        rng.choice(
                            np.arange(hard_count, len(candidates)),
                            size=min(count - hard_count, len(candidates) - hard_count),
                            replace=False,
                        ).tolist()
                    )
                negatives.append(
                    extract_patch_channels(
                        frame, score, candidates[selected], args.patch_size
                    )
                )
            frames_seen += 1
        print(
            f"patch-data {sequence.name}: frames={frames_seen} "
            f"positive_chunks={len(positives)} negative_chunks={len(negatives)}",
            flush=True,
        )

    positive = np.concatenate(positives)
    negative = np.concatenate(negatives)
    if args.polarity_augment:
        positive_flipped = positive.copy()
        negative_flipped = negative.copy()
        positive_flipped[:, 0] *= -1
        negative_flipped[:, 0] *= -1
        positive = np.concatenate((positive, positive_flipped))
        negative = np.concatenate((negative, negative_flipped))
    print(f"patch-data positives={len(positive)} negatives={len(negative)}", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = TBDPatchClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    positive_tensor = torch.from_numpy(positive)
    negative_tensor = torch.from_numpy(negative)

    history = []
    for epoch in range(1, args.epochs + 1):
        sample_count = min(len(positive_tensor), len(negative_tensor))
        positive_ids = torch.randperm(len(positive_tensor))[:sample_count]
        negative_ids = torch.randperm(len(negative_tensor))[:sample_count]
        features = torch.cat(
            (positive_tensor[positive_ids], negative_tensor[negative_ids])
        )
        labels = torch.cat((torch.ones(sample_count), torch.zeros(sample_count)))
        permutation = torch.randperm(len(labels))
        loader = DataLoader(
            TensorDataset(features[permutation], labels[permutation]),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        model.train()
        total_loss = correct = seen = 0
        for batch, target in loader:
            batch = batch.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(batch)
            correct += int(((logits >= 0) == (target >= 0.5)).sum())
            seen += len(batch)
        row = {
            "epoch": epoch,
            "loss": total_loss / seen,
            "accuracy": correct / seen,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "patch_size": args.patch_size,
            "settings": vars(args),
            "history": history,
            "positive_samples": len(positive),
            "negative_samples": len(negative),
        },
        output,
    )
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
