"""Train the official MSHNet with SLS-style supervision on challenge frames."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from jinsight_track1.deeppro_adapter import _files, _sequences
from jinsight_track1.lvnet_adapter import ScaleLocationLoss
from jinsight_track1.mshnet_adapter import load_mshnet_model


def frame_catalog(
    root: Path, resolutions: set[str] | None, temporal_radius: int
):
    groups = defaultdict(list)
    for sequence in _sequences(root):
        images, masks = _files(sequence / "img"), _files(sequence / "mask")
        paired = sorted(set(images) & set(masks))
        if not paired:
            continue
        width, height = Image.open(images[paired[0]]).size
        resolution = f"{width}x{height}"
        if resolutions and resolution not in resolutions:
            continue
        for index, stem in enumerate(paired):
            clip = tuple(
                images[paired[min(max(index + offset, 0), len(paired) - 1)]]
                for offset in range(-temporal_radius, temporal_radius + 1)
            )
            groups[resolution].append((clip, masks[stem]))
    if not groups:
        raise ValueError("no training frames match the requested resolutions")
    return dict(groups)


class ChallengeFrameDataset:
    def __init__(
        self,
        torch,
        root: Path,
        samples_per_epoch: int,
        patch_size: int,
        mean: float,
        std: float,
        positive_probability: float,
        polarity_flip_probability: float,
        adaptive_normalization: bool,
        temporal_radius: int,
        resolutions: set[str] | None,
        balance_resolutions: bool,
    ):
        self.torch = torch
        self.groups = frame_catalog(root, resolutions, temporal_radius)
        self.group_names = sorted(self.groups)
        self.group_weights = np.asarray(
            [len(self.groups[name]) for name in self.group_names], dtype=float
        )
        if balance_resolutions:
            self.group_weights[:] = 1
        self.group_weights /= self.group_weights.sum()
        self.samples_per_epoch = samples_per_epoch
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        self.positive_probability = positive_probability
        self.polarity_flip_probability = polarity_flip_probability
        self.adaptive_normalization = adaptive_normalization

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, _):
        group = np.random.choice(self.group_names, p=self.group_weights)
        pairs = self.groups[group]
        # A few retries make positive-centred sampling reliable without eagerly
        # decoding every one of the roughly 100k masks at dataset construction.
        image = mask = None
        want_positive = random.random() < self.positive_probability
        for _attempt in range(12 if want_positive else 1):
            image_paths, mask_path = random.choice(pairs)
            candidate_mask = (
                np.asarray(Image.open(mask_path).convert("L")) > 0
            ).astype(np.float32)
            image = np.stack(
                [
                    np.asarray(Image.open(path).convert("L"), dtype=np.float32)
                    for path in image_paths
                ]
            )
            mask = candidate_mask
            if not want_positive or mask.any():
                break
        assert image is not None and mask is not None
        _, height, width = image.shape
        patch = self.patch_size
        if height < patch or width < patch:
            pad_h, pad_w = max(0, patch - height), max(0, patch - width)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)))
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)))
            _, height, width = image.shape
        if want_positive and mask.any():
            y, x = random.choice(np.argwhere(mask > 0).tolist())
            y0 = random.randint(max(0, y - patch), min(y, height - patch))
            x0 = random.randint(max(0, x - patch), min(x, width - patch))
        else:
            y0 = random.randint(0, height - patch)
            x0 = random.randint(0, width - patch)
        image = image[:, y0 : y0 + patch, x0 : x0 + patch]
        mask = mask[y0 : y0 + patch, x0 : x0 + patch]
        if self.adaptive_normalization:
            means = image.mean(axis=(1, 2), keepdims=True)
            stds = image.std(axis=(1, 2), keepdims=True).clip(min=1.0)
            image = (image - means) / stds
        else:
            image = (image - self.mean) / self.std
        if random.random() < self.polarity_flip_probability:
            image = -image
        if random.random() < .5:
            image, mask = image[:, ::-1], mask[::-1]
        if random.random() < .5:
            image, mask = image[:, :, ::-1], mask[:, ::-1]
        if random.random() < .5:
            image, mask = image.transpose(0, 2, 1), mask.T
        return (
            self.torch.from_numpy(np.ascontiguousarray(image)),
            self.torch.from_numpy(np.ascontiguousarray(mask[None])),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("train_root")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--initial-weights",
        help="optional MSHNet checkpoint used to initialize a continuation run",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=10000)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument(
        "--temporal-radius",
        type=int,
        default=0,
        help="use 2*radius+1 neighbouring frames as input channels",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", choices=["adagrad", "adamw"], default="adagrad")
    parser.add_argument("--learning-rate", type=float, default=.05)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--mean", type=float, default=111.47)
    parser.add_argument("--std", type=float, default=22.43)
    parser.add_argument("--adaptive-normalization", action="store_true")
    parser.add_argument("--positive-probability", type=float, default=.9)
    parser.add_argument("--polarity-flip-probability", type=float, default=0)
    parser.add_argument("--mask-dilation", type=int, default=1)
    parser.add_argument("--resolutions")
    parser.add_argument("--balance-resolutions", action="store_true")
    parser.add_argument("--focal-weight", type=float, default=.25)
    parser.add_argument("--location-weight", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.mask_dilation < 1 or args.mask_dilation % 2 != 1:
        raise ValueError("mask-dilation must be a positive odd integer")

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    if args.temporal_radius < 0:
        raise ValueError("temporal-radius must be non-negative")
    model = load_mshnet_model(
        args.source_root, input_channels=2 * args.temporal_radius + 1
    ).to(device)
    if args.initial_weights:
        checkpoint = torch.load(
            args.initial_weights, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    dataset = ChallengeFrameDataset(
        torch,
        Path(args.train_root),
        args.samples_per_epoch,
        args.patch_size,
        args.mean,
        args.std,
        args.positive_probability,
        args.polarity_flip_probability,
        args.adaptive_normalization,
        args.temporal_radius,
        set(args.resolutions.split(",")) if args.resolutions else None,
        args.balance_resolutions,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    if args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    loss_function = ScaleLocationLoss(
        torch,
        focal_weight=args.focal_weight,
        location_weight=args.location_weight,
    )
    amp_enabled = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for iteration, (images, masks) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if args.mask_dilation > 1:
                masks = torch.nn.functional.max_pool2d(
                    masks,
                    kernel_size=args.mask_dilation,
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
                    loss = loss + .25 * loss_function(
                        side[:, :, None], target[:, :, None], use_location=False
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            if iteration % 50 == 0:
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
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(row)
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "history": history,
                "mean": args.mean,
                "std": args.std,
                "adaptive_normalization": args.adaptive_normalization,
                "temporal_radius": args.temporal_radius,
            },
            output / f"epoch_{epoch:02d}.pth.tar",
        )
        (output / "train_metrics.json").write_text(
            json.dumps(
                {
                    "model": "MSHNet",
                    "initial_weights": args.initial_weights,
                    "resolutions": sorted(dataset.group_names),
                    "samples_per_epoch": args.samples_per_epoch,
                    "optimizer": args.optimizer,
                    "learning_rate": args.learning_rate,
                    "adaptive_normalization": args.adaptive_normalization,
                    "temporal_radius": args.temporal_radius,
                    "polarity_flip_probability": args.polarity_flip_probability,
                    "mask_dilation": args.mask_dilation,
                    "history": history,
                    "elapsed_seconds": time.time() - started,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(row), flush=True)
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
