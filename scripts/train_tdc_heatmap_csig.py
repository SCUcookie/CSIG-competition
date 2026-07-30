#!/usr/bin/env python3
"""Train a pixel-level heatmap head on the official TDCNet temporal backbone."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from train_tdcnet_csig import CSIGTDCSequenceDataset, collate


class TDCHeatmap(nn.Module):
    """Replace TDCNet's stride-8 box head with a full-resolution point head."""

    def __init__(self, source_root: str | Path, context: int = 5,
                 backbone_weights: str | Path | None = None):
        super().__init__()
        sys.path.insert(0, str(Path(source_root).resolve()))
        from model.TDCNet.TDCNetwork import TDCNetwork

        self.temporal = TDCNetwork(1, num_frame=context)
        if backbone_weights:
            self.temporal.load_state_dict(
                torch.load(backbone_weights, map_location="cpu")
            )
        self.temporal.head = nn.Identity()
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        nn.init.constant_(self.decoder[-1].bias, -4.6)

    def forward(self, images):
        features = self.temporal(images)[0]
        return self.decoder(features)


def gaussian_targets(targets, height, width, device, sigma=1.5):
    result = torch.zeros(
        (len(targets), 1, height, width), device=device, dtype=torch.float32
    )
    radius = int(np.ceil(3 * sigma))
    axis = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * sigma**2))
    for batch_index, points in enumerate(targets):
        for point in points:
            x, y = (int(round(float(value))) for value in point[:2])
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            kx0, ky0 = x0 - (x - radius), y0 - (y - radius)
            patch = kernel[ky0:ky0 + y1 - y0, kx0:kx0 + x1 - x0]
            result[batch_index, 0, y0:y1, x0:x1] = torch.maximum(
                result[batch_index, 0, y0:y1, x0:x1], patch
            )
    return result


def centernet_focal_loss(logits, target):
    prediction = logits.sigmoid().clamp(1e-5, 1 - 1e-5)
    positive = target.eq(1).float()
    negative = target.lt(1).float()
    negative_weight = (1 - target).pow(4)
    positive_loss = torch.log(prediction) * (1 - prediction).pow(2) * positive
    negative_loss = (
        torch.log(1 - prediction)
        * prediction.pow(2)
        * negative_weight
        * negative
    )
    count = positive.sum().clamp_min(1)
    return -(positive_loss.sum() + negative_loss.sum()) / count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("train_root")
    parser.add_argument("output")
    parser.add_argument("--backbone-weights")
    parser.add_argument("--resolution", default="256x256")
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main():
    args = parse_args()
    width, height = (int(value) for value in args.resolution.split("x"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = CSIGTDCSequenceDataset(
        args.train_root, (width, height), args.context, 12.0, True
    )
    sampler = RandomSampler(
        dataset, replacement=True,
        num_samples=args.steps_per_epoch * args.batch_size,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, collate_fn=collate,
        persistent_workers=args.workers > 0,
    )
    device = torch.device(f"cuda:{args.device}")
    model = TDCHeatmap(
        args.source_root, args.context, args.backbone_weights
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.temporal.parameters(), "lr": args.backbone_lr},
            {"params": model.decoder.parameters(), "lr": args.head_lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.backbone_lr * 0.03
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "samples": len(dataset),
        "sequences": len(dataset.sequences),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "train_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for step, (images, targets) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            targets = [target.to(device, non_blocking=True) for target in targets]
            truth = gaussian_targets(targets, height, width, device)
            optimizer.zero_grad(set_to_none=True)
            loss = centernet_focal_loss(model(images), truth)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += float(loss)
            if step % 100 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/"
                    f"{args.steps_per_epoch} loss={total / step:.5f}",
                    flush=True,
                )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "loss": total / args.steps_per_epoch,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        torch.save(model.state_dict(), output / f"epoch_{epoch + 1:02d}.pth")
        (output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
