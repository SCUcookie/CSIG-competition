#!/usr/bin/env python3
"""Train HDNet on spatial frames plus forward/backward temporal differences."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from jinsight_track1.hdnet_adapter import load_hdnet_model
from jinsight_track1.lvnet_adapter import ScaleLocationLoss
from train_mshnet import ChallengeFrameDataset


class DifferenceDataset:
    def __init__(self, base):
        self.base = base
        self.group_names = base.group_names

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        images, mask = self.base[index]
        previous, center, following = images
        return (
            self.base.torch.stack(
                (center, center - previous, center - following), dim=0
            ),
            mask,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("train_root")
    parser.add_argument("output_dir")
    parser.add_argument("--initial-weights")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=5000)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--positive-probability", type=float, default=0.9)
    parser.add_argument("--polarity-flip-probability", type=float, default=0.5)
    parser.add_argument("--mask-dilation", type=int, default=3)
    parser.add_argument("--resolutions", default="256x256")
    parser.add_argument("--focal-weight", type=float, default=0.25)
    parser.add_argument("--location-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    model = load_hdnet_model(args.source_root, input_channels=3)
    loaded_keys = []
    if args.initial_weights:
        checkpoint = torch.load(
            args.initial_weights, map_location="cpu", weights_only=False
        )
        source_state = checkpoint.get("state_dict", checkpoint)
        target_state = model.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in target_state and value.shape == target_state[key].shape
        }
        model.load_state_dict(compatible, strict=False)
        loaded_keys = sorted(compatible)
    model.to(device)

    base = ChallengeFrameDataset(
        torch,
        Path(args.train_root),
        args.samples_per_epoch,
        args.patch_size,
        111.47,
        22.43,
        args.positive_probability,
        args.polarity_flip_probability,
        True,
        1,
        set(args.resolutions.split(",")) if args.resolutions else None,
        False,
    )
    dataset = DifferenceDataset(base)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
    loss_function = ScaleLocationLoss(
        torch,
        focal_weight=args.focal_weight,
        location_weight=args.location_weight,
    )
    amp_enabled = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    history = []
    config = {
        **vars(args),
        "loaded_compatible_keys": len(loaded_keys),
        "loaded_compatible_parameters": sum(
            model.state_dict()[key].numel() for key in loaded_keys
        ),
        "representation": "center, center-previous, center-following",
    }
    (output / "train_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for iteration, (images, masks) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if args.mask_dilation > 1:
                masks = torch.nn.functional.max_pool2d(
                    masks,
                    args.mask_dilation,
                    stride=1,
                    padding=args.mask_dilation // 2,
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                side_outputs, logits = model(images, True)
                loss = loss_function(
                    logits[:, :, None], masks[:, :, None], use_location=True
                )
                for side in side_outputs[1:]:
                    target = torch.nn.functional.interpolate(
                        masks, size=side.shape[-2:], mode="nearest"
                    )
                    loss = loss + 0.25 * loss_function(
                        side[:, :, None], target[:, :, None], use_location=False
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            if iteration % 50 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "iteration": iteration,
                            "iterations": len(loader),
                            "loss": float(np.mean(losses)),
                        }
                    ),
                    flush=True,
                )
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(row)
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "history": history,
                "adaptive_normalization": True,
                "representation": config["representation"],
            },
            output / f"epoch_{epoch:02d}.pth.tar",
        )
        (output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        (output / "progress.json").write_text(
            json.dumps(
                {
                    "status": "running" if epoch < args.epochs else "complete",
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
