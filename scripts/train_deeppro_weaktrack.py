#!/usr/bin/env python3
"""Fine-tune DeepPro with a weak-target contrast curriculum.

The curriculum edits only labelled target neighborhoods: it attenuates or
reverses their contrast against a local background estimate while preserving
the real sensor background. This targets whole weak trajectories missed by the
existing high-precision model rather than relearning generic segmentation.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import types
from pathlib import Path

import numpy as np

from jinsight_track1.deeppro_adapter import DeepProDetector, _sequences
from jinsight_track1.deeppro_train import (
    _dataset_config,
    _fast_sample_sequence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("initial_weights")
    parser.add_argument("train_root")
    parser.add_argument("output_dir")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--sample-rate", type=float, default=0.05)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-contrast", type=float, default=0.1)
    parser.add_argument("--max-contrast", type=float, default=0.8)
    parser.add_argument("--target-polarity-probability", type=float, default=0.35)
    parser.add_argument("--global-inversion-probability", type=float, default=0.25)
    parser.add_argument("--mask-dilation", type=int, default=3)
    parser.add_argument("--positive-weight", type=float, default=0.5)
    parser.add_argument("--negative-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_root = _dataset_config(Path(args.train_root), output)
    detector = DeepProDetector(
        args.source_root,
        args.initial_weights,
        device=args.device,
        sequence_length=args.sequence_length,
    )
    model = detector.model

    from data_utils.TrainDataLoader import TrainIRSeqDataLoader

    dataset = TrainIRSeqDataLoader(
        "SatVideoIRSDT",
        data_root=str(config_root),
        seq_len=args.sequence_length,
        sample_rate=args.sample_rate,
        patch_size=args.patch_size,
        transform=None,
    )
    dataset._sample_cdf = np.cumsum(np.asarray(dataset.sample_p, dtype=np.float64))
    dataset._sample_cdf[-1] = 1.0
    dataset.sample_sequence = types.MethodType(_fast_sample_sequence, dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config = {
        **vars(args),
        "training_samples_per_epoch": len(dataset),
        "steps_per_epoch": len(loader),
        "train_sequences": len(_sequences(Path(args.train_root))),
        "augmentation": "local-background target contrast curriculum",
    }
    (output / "train_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, (images, targets) in enumerate(loader, 1):
            images = images.float().to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True)
            target_mask = F.max_pool3d(
                targets[:, None],
                kernel_size=(1, args.mask_dilation, args.mask_dilation),
                stride=1,
                padding=(0, args.mask_dilation // 2, args.mask_dilation // 2),
            )
            local_background = F.avg_pool3d(
                images, kernel_size=(1, 7, 7), stride=1, padding=(0, 3, 3)
            )
            contrast = torch.empty(
                (len(images), 1, 1, 1, 1), device=device
            ).uniform_(args.min_contrast, args.max_contrast)
            residual = images - local_background
            flip_target = (
                torch.rand((len(images), 1, 1, 1, 1), device=device)
                < args.target_polarity_probability
            )
            signed_contrast = torch.where(flip_target, -contrast, contrast)
            augmented_target = local_background + signed_contrast * residual
            images = images * (1.0 - target_mask) + augmented_target * target_mask
            invert = (
                torch.rand((len(images), 1, 1, 1, 1), device=device)
                < args.global_inversion_probability
            )
            images = torch.where(invert, -images, images)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _, logits = model(images)
                probability = torch.sigmoid(logits)
                intersection = (probability * target_mask[:, 0]).sum(
                    dim=(1, 2, 3)
                )
                union = (
                    probability.sum(dim=(1, 2, 3))
                    + target_mask[:, 0].sum(dim=(1, 2, 3))
                    - intersection
                )
                soft_iou = 1.0 - ((intersection + 1e-6) / (union + 1e-6)).mean()
                positive = target_mask[:, 0] > 0
                negative = ~positive
                positive_loss = (
                    F.softplus(-logits)[positive].mean()
                    if positive.any()
                    else logits.new_tensor(0.0)
                )
                negative_loss = F.softplus(logits)[negative].mean()
                loss = (
                    soft_iou
                    + args.positive_weight * positive_loss
                    + args.negative_weight * negative_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            if step % 50 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "steps": len(loader),
                            "loss": float(np.mean(losses)),
                        }
                    ),
                    flush=True,
                )
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "training_config": config,
            "history": history,
        }
        torch.save(checkpoint, output / f"epoch_{epoch:02d}_model.pth")
        (output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        (output / "progress.json").write_text(
            json.dumps(
                {
                    "status": "complete" if epoch == args.epochs else "running",
                    **row,
                    "elapsed_seconds": time.time() - started,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
