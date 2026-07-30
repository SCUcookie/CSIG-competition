"""Train the published LVNet architecture on challenge-format sequences."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from jinsight_track1.lvnet_adapter import ScaleLocationLoss, load_lvnet_model
from train_rfr import ChallengeRFRDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("train_root")
    parser.add_argument("output_dir")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-epoch", type=int, default=10000)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mean", type=float, default=111.47)
    parser.add_argument("--std", type=float, default=22.43)
    parser.add_argument("--positive-probability", type=float, default=.9)
    parser.add_argument("--polarity-flip-probability", type=float, default=0)
    parser.add_argument("--mask-dilation", type=int, default=1)
    parser.add_argument("--resolutions")
    parser.add_argument("--balance-resolutions", action="store_true")
    parser.add_argument("--warm-epochs", type=int, default=0)
    parser.add_argument("--focal-weight", type=float, default=.25)
    parser.add_argument("--location-weight", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.sequence_length != 4:
        raise ValueError("the published LVNet configuration requires 4 frames")
    if args.mask_dilation < 1 or args.mask_dilation % 2 != 1:
        raise ValueError("mask-dilation must be a positive odd integer")

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch_device = torch.device(
        "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    )
    model = load_lvnet_model(
        args.source_root, num_frames=args.sequence_length
    ).to(torch_device)
    dataset = ChallengeRFRDataset(
        torch,
        Path(args.train_root),
        args.samples_per_epoch,
        args.sequence_length,
        args.patch_size,
        args.mean,
        args.std,
        args.positive_probability,
        args.polarity_flip_probability,
        set(args.resolutions.split(",")) if args.resolutions else None,
        args.balance_resolutions,
    )
    # Upstream LVNet hard-codes b=1 in two einops operations.
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    loss_function = ScaleLocationLoss(
        torch,
        focal_weight=args.focal_weight,
        location_weight=args.location_weight,
    )
    amp_enabled = not args.no_amp and torch_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for iteration, (images, masks) in enumerate(loader, 1):
            images = images.to(torch_device, non_blocking=True).permute(0, 2, 1, 3, 4)
            masks = masks.to(torch_device, non_blocking=True).permute(0, 2, 1, 3, 4)
            if args.mask_dilation > 1:
                batch, channels, depth, height, width = masks.shape
                masks = torch.nn.functional.max_pool2d(
                    masks.permute(0, 2, 1, 3, 4).reshape(
                        batch * depth, channels, height, width
                    ),
                    kernel_size=args.mask_dilation,
                    stride=1,
                    padding=args.mask_dilation // 2,
                ).reshape(batch, depth, channels, height, width).permute(
                    0, 2, 1, 3, 4
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = loss_function(
                    logits, masks, use_location=epoch > args.warm_epochs
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            if iteration % 100 == 0:
                progress = {
                    "status": "running",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "iteration": iteration,
                    "iterations": len(loader),
                    "mean_loss": float(np.mean(losses)),
                    "elapsed_seconds": time.time() - started,
                }
                (output / "progress.json").write_text(
                    json.dumps(progress, indent=2), encoding="utf-8"
                )
                print(json.dumps(progress), flush=True)
        scheduler.step()
        epoch_row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_row)
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "mean": args.mean,
                "std": args.std,
                "num_frames": args.sequence_length,
            },
            output / f"epoch_{epoch:02d}.pth.tar",
        )
        (output / "train_metrics.json").write_text(
            json.dumps(
                {
                    "model": "LVNet",
                    "resolutions": sorted(dataset.group_names),
                    "balanced_resolutions": args.balance_resolutions,
                    "samples_per_epoch": args.samples_per_epoch,
                    "polarity_flip_probability": args.polarity_flip_probability,
                    "mask_dilation": args.mask_dilation,
                    "warm_epochs": args.warm_epochs,
                    "focal_weight": args.focal_weight,
                    "location_weight": args.location_weight,
                    "history": history,
                    "elapsed_seconds": time.time() - started,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(epoch_row), flush=True)
    (output / "progress.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "epochs": args.epochs,
                "history": history,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
