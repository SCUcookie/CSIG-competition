#!/usr/bin/env python
"""Train FeedbackSTS-Det directly on CSIG sequence masks.

The upstream project relies on a legacy DCNv2 extension.  This entry point
installs a parameter-compatible torchvision implementation before importing
the official model, and keeps the original five-frame dense segmentation
objective instead of converting point masks to detection boxes.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from jinsight_track1.deeppro_adapter import _files, _sequences


def install_dcn_compat(source_root: Path, torch) -> None:
    """Expose upstream's DeformConv API through torchvision.ops."""
    from torchvision.ops import deform_conv2d

    nn = torch.nn
    pair = nn.modules.utils._pair

    class DeformConv(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=1,
            groups=1,
            deformable_groups=1,
            im2col_step=128,
            bias=True,
            lr_mult=0.1,
        ):
            super().__init__()
            self.kernel_size = pair(kernel_size)
            self.stride = pair(stride)
            self.padding = pair(padding)
            self.dilation = pair(dilation)
            self.groups = groups
            self.deformable_groups = deformable_groups
            self.im2col_step = im2col_step
            self.weight = nn.Parameter(
                torch.empty(
                    out_channels,
                    in_channels // groups,
                    self.kernel_size[0],
                    self.kernel_size[1],
                )
            )
            self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
            offset_channels = (
                deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
            )
            self.conv_offset = nn.Conv2d(
                in_channels,
                offset_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            self.conv_offset.lr_mult = lr_mult
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            nn.init.zeros_(self.conv_offset.weight)
            nn.init.zeros_(self.conv_offset.bias)

        def forward(self, input_tensor, offset_features):
            offset_y, offset_x, mask = torch.chunk(
                self.conv_offset(offset_features), 3, dim=1
            )
            offset = torch.cat((offset_y, offset_x), dim=1)
            return deform_conv2d(
                input_tensor,
                offset,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=torch.sigmoid(mask),
            )

    package_name = "model.dcn_half.modules"
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(source_root / "model" / "dcn_half" / "modules")
    ]
    module_name = f"{package_name}.deform_conv"
    module = types.ModuleType(module_name)
    module.DeformConv = DeformConv
    module.ConvOffset2d = DeformConv
    module.DeformConvPack = DeformConv
    module._DeformConv = None
    package.DeformConv = DeformConv
    sys.modules[package_name] = package
    sys.modules[module_name] = module


def load_feedbacksts(source_root: str | Path, torch):
    source = Path(source_root).resolve()
    if not (source / "model" / "FeedbackSTS.py").is_file():
        raise FileNotFoundError(f"FeedbackSTS source not found: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    install_dcn_compat(source, torch)
    sys.modules.pop("model.FeedbackSTS", None)
    return importlib.import_module("model.FeedbackSTS").FeedbackSTS()


def sequence_catalog(
    root: Path, resolutions: set[str] | None = None
) -> dict[str, list[tuple[list[Path], list[Path]]]]:
    groups = defaultdict(list)
    for sequence in _sequences(root):
        images = _files(sequence / "img")
        masks = _files(sequence / "mask")
        stems = sorted(set(images) & set(masks))
        if len(stems) < 2:
            continue
        width, height = Image.open(images[stems[0]]).size
        resolution = f"{width}x{height}"
        if resolutions and resolution not in resolutions:
            continue
        groups[resolution].append(
            ([images[stem] for stem in stems], [masks[stem] for stem in stems])
        )
    return dict(groups)


class FeedbackSequenceDataset:
    def __init__(
        self,
        torch,
        root: Path,
        samples_per_epoch: int,
        sequence_length: int,
        patch_size: int,
        mean: float,
        std: float,
        positive_probability: float,
        polarity_flip_probability: float,
        max_temporal_stride: int,
        resolutions: set[str] | None,
        balance_resolutions: bool,
    ):
        self.torch = torch
        self.groups = sequence_catalog(root, resolutions)
        if not self.groups:
            raise ValueError("no training sequences match the requested resolutions")
        self.group_names = sorted(self.groups)
        self.group_weights = np.asarray(
            [
                sum(len(images) for images, _ in self.groups[name])
                for name in self.group_names
            ],
            dtype=float,
        )
        if balance_resolutions:
            self.group_weights[:] = 1
        self.group_weights /= self.group_weights.sum()
        self.sequence_weights = {
            name: np.asarray(
                [len(images) for images, _ in sequences], dtype=float
            )
            for name, sequences in self.groups.items()
        }
        for weights in self.sequence_weights.values():
            weights /= weights.sum()
        self.samples_per_epoch = samples_per_epoch
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        self.positive_probability = positive_probability
        self.polarity_flip_probability = polarity_flip_probability
        self.max_temporal_stride = max_temporal_stride

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, _):
        group = str(np.random.choice(self.group_names, p=self.group_weights))
        sequences = self.groups[group]
        seq_index = int(
            np.random.choice(
                len(sequences), p=self.sequence_weights[group]
            )
        )
        image_paths, mask_paths = sequences[seq_index]
        stride = random.randint(1, self.max_temporal_stride)
        start = random.randrange(len(image_paths))
        indices = [
            min(start + offset * stride, len(image_paths) - 1)
            for offset in range(self.sequence_length)
        ]
        images = np.stack(
            [
                np.asarray(Image.open(image_paths[index]).convert("L"), dtype=np.float32)
                for index in indices
            ]
        )
        masks = np.stack(
            [
                np.asarray(Image.open(mask_paths[index]).convert("L")) > 0
                for index in indices
            ]
        ).astype(np.float32)
        images = (images - self.mean) / self.std
        if random.random() < self.polarity_flip_probability:
            images = -images

        _, height, width = images.shape
        patch = min(self.patch_size, height, width)
        if patch < height or patch < width:
            positive = masks.max() and random.random() < self.positive_probability
            if positive:
                locations = np.argwhere(masks > 0)
                _, y, x = locations[random.randrange(len(locations))]
                y0 = random.randint(max(0, int(y) - patch), min(int(y), height - patch))
                x0 = random.randint(max(0, int(x) - patch), min(int(x), width - patch))
            else:
                y0 = random.randint(0, height - patch)
                x0 = random.randint(0, width - patch)
            images = images[:, y0 : y0 + patch, x0 : x0 + patch]
            masks = masks[:, y0 : y0 + patch, x0 : x0 + patch]

        if random.random() < 0.5:
            images, masks = images[:, ::-1], masks[:, ::-1]
        if random.random() < 0.5:
            images, masks = images[:, :, ::-1], masks[:, :, ::-1]
        if random.random() < 0.5:
            images, masks = images[::-1], masks[::-1]
        if random.random() < 0.5:
            images = images.transpose(0, 2, 1)
            masks = masks.transpose(0, 2, 1)
        return (
            self.torch.from_numpy(np.ascontiguousarray(images[None])),
            self.torch.from_numpy(np.ascontiguousarray(masks[None])),
        )


def segmentation_loss(
    probability,
    target,
    positive_weight: float = 1.0,
    negative_weight: float = 0.25,
):
    import torch

    probability = probability.float().clamp(1e-5, 1 - 1e-5)
    target = target.float()
    dimensions = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dimensions)
    union = (
        probability.sum(dim=dimensions)
        + target.sum(dim=dimensions)
        - intersection
    )
    soft_iou = 1 - ((intersection + 1) / (union + 1)).mean()
    positive = target > 0.5
    negative = ~positive
    positive_focal = (
        -((1 - probability[positive]).square())
        * probability[positive].log()
    ).mean() if positive.any() else probability.sum() * 0
    negative_focal = (
        -(probability[negative].square())
        * (1 - probability[negative]).log()
    ).mean() if negative.any() else probability.sum() * 0
    balanced_focal = (
        positive_weight * positive_focal
        + negative_weight * negative_focal
    )
    return (
        soft_iou + balanced_focal,
        soft_iou.detach(),
        balanced_focal.detach(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("train_root")
    parser.add_argument("output_dir")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=2000)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mean", type=float, default=111.47)
    parser.add_argument("--std", type=float, default=22.43)
    parser.add_argument("--positive-probability", type=float, default=0.9)
    parser.add_argument("--polarity-flip-probability", type=float, default=0.5)
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=5,
        help="odd target-mask dilation used only for the training objective",
    )
    parser.add_argument("--positive-loss-weight", type=float, default=1.0)
    parser.add_argument("--negative-loss-weight", type=float, default=0.25)
    parser.add_argument("--max-temporal-stride", type=int, default=3)
    parser.add_argument("--resolutions", default="256x256")
    parser.add_argument("--balance-resolutions", action="store_true")
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
    device = torch.device(f"cuda:{args.device}")
    model = load_feedbacksts(args.source_root, torch).to(device)
    dataset = FeedbackSequenceDataset(
        torch,
        Path(args.train_root),
        args.samples_per_epoch,
        args.sequence_length,
        args.patch_size,
        args.mean,
        args.std,
        args.positive_probability,
        args.polarity_flip_probability,
        args.max_temporal_stride,
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
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.learning_rate * 0.1
    )
    amp_enabled = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "source_root": str(Path(args.source_root).resolve()),
        "train_root": str(Path(args.train_root).resolve()),
        "resolutions_found": dataset.group_names,
        "sequences": {
            name: len(dataset.groups[name]) for name in dataset.group_names
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "train_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, indent=2), flush=True)

    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(3, dtype=float)
        for step, (images, masks) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if args.mask_dilation > 1:
                batch, channels, length, height, width = masks.shape
                masks = torch.nn.functional.max_pool2d(
                    masks.permute(0, 2, 1, 3, 4).reshape(
                        batch * length, channels, height, width
                    ),
                    kernel_size=args.mask_dilation,
                    stride=1,
                    padding=args.mask_dilation // 2,
                ).reshape(batch, length, channels, height, width).permute(
                    0, 2, 1, 3, 4
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                probability = model(images)
                loss, soft_iou, focal = segmentation_loss(
                    probability,
                    masks,
                    args.positive_loss_weight,
                    args.negative_loss_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals += (float(loss.detach()), float(soft_iou), float(focal))
            if step % 50 == 0:
                print(
                    f"epoch={epoch}/{args.epochs} step={step}/{len(loader)} "
                    f"loss={totals[0] / step:.6f} "
                    f"iou={totals[1] / step:.6f} "
                    f"focal={totals[2] / step:.6f}",
                    flush=True,
                )
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": totals[0] / len(loader),
            "soft_iou_loss": totals[1] / len(loader),
            "focal_loss": totals[2] / len(loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        torch.save(model.state_dict(), output / f"epoch_{epoch:02d}.pth")
        (output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
