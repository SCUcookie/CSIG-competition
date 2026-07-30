#!/usr/bin/env python3
"""Train a detection-supervised MoPKL motion branch on CSIG sequences.

This route keeps the official MoPKL feature extractor, visual motion decoder,
multi-scale fusion module, and stride-8 YOLOX head. It deliberately removes
the language/relation losses so CSIG can train from its point masks alone.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from train_tdcnet_csig import CSIGTDCSequenceDataset, collate


class MoPKLLite(nn.Module):
    """Official MoPKL visual path without language-conditioned training."""

    def __init__(self, source_root: str | Path):
        super().__init__()
        sys.path.insert(0, str(Path(source_root).resolve()))
        # The official module imports CuPy only for its language-supervised
        # comet-mask generator. This visual-only route never calls that code.
        try:
            import cupy  # noqa: F401
        except ModuleNotFoundError:
            sys.modules["cupy"] = np
        from nets.MoPKL import (
            BaseConv,
            Feature_Extractor,
            Fusion_Module,
            MotionModel,
            YOLOXHead,
        )

        self.backbone = Feature_Extractor(0.33, 0.50)
        self.conv_vl = nn.Sequential(
            BaseConv(128 * 2, 256, 3, 1),
            BaseConv(256, 256, 3, 1),
            BaseConv(256, 256, 1, 1),
        )
        self.motion = MotionModel(
            text_input_dim=130 * 300, latent_dim=128, hidden_dim=1024
        )
        self.conv_m = nn.Sequential(
            BaseConv(1, 64, 3, 2),
            BaseConv(64, 128, 3, 2),
            BaseConv(128, 256, 3, 2),
            BaseConv(256, 256, 1, 1),
        )
        self.fusion = Fusion_Module(channels=[128], num_frame=2)
        self.head = YOLOXHead(
            num_classes=1, width=1.0, in_channels=[256], act="silu"
        )

    def forward(self, inputs):
        if inputs.ndim != 5 or inputs.shape[2] != 2:
            raise ValueError("MoPKLLite expects [B, 3, 2, 512, 512]")
        features = [
            self.backbone(inputs[:, :, frame]) for frame in range(2)
        ]
        joint = self.conv_vl(torch.cat(features, dim=1))
        motion = self.motion.inference_forward(joint)
        motion = self.conv_m(motion.unsqueeze(1))
        fused = self.fusion(motion, features[-1])
        return self.head(fused)


def prepare_inputs(images: torch.Tensor, input_size: int) -> torch.Tensor:
    """Select raw history, resize, and ImageNet-normalize repeated gray RGB."""
    context = images.shape[2] // 2
    raw = images[:, :, context:]
    if raw.shape[2] != 2:
        raise ValueError("dataset context must be 2")
    batch, channels, frames, height, width = raw.shape
    raw = raw.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    raw = F.interpolate(
        raw, size=(input_size, input_size), mode="bilinear",
        align_corners=False,
    )
    mean = raw.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = raw.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    raw = (raw - mean) / std
    return raw.reshape(batch, frames, channels, input_size, input_size).permute(
        0, 2, 1, 3, 4
    )


def load_backbone_initialization(model, path: str | Path):
    state = torch.load(path, map_location="cpu")
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    current.update(compatible)
    model.load_state_dict(current)
    return sorted(compatible)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", help="official MoPKL checkout")
    parser.add_argument("train_root")
    parser.add_argument("output")
    parser.add_argument("--resolution", default="256x256")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--box-size", type=float, default=12.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume")
    return parser.parse_args()


def main():
    args = parse_args()
    width, height = (int(value) for value in args.resolution.split("x"))
    if width != height or args.input_size != 512:
        raise ValueError("the official MoPKL visual decoder requires 512 input")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = CSIGTDCSequenceDataset(
        args.train_root, (width, height), context=2,
        box_size=args.box_size, augment=True,
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
    model = MoPKLLite(args.source_root)
    from nets.training import YOLOLoss, weights_init

    weights_init(model)
    loaded = load_backbone_initialization(
        model, Path(args.source_root) / "model_data/pre_trained_backbone.pth"
    )
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    model.to(device)
    criterion = YOLOLoss(1, fp16=False, strides=[8])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs),
        eta_min=args.learning_rate * 0.03,
    )
    scale = args.input_size / width
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "samples": len(dataset),
        "sequences": len(dataset.sequences),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "pretrained_keys_loaded": len(loaded),
        "official_source_commit": "dfcfebf",
        "route": "visual-only MoPKL with direct detection supervision",
    }
    (output / "train_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for step, (images, targets) in enumerate(loader, 1):
            images = prepare_inputs(
                images.to(device, non_blocking=True), args.input_size
            )
            scaled_targets = []
            for target in targets:
                target = target.to(device, non_blocking=True).clone()
                target[:, :4] *= scale
                scaled_targets.append(target)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), scaled_targets)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch + 1}, step {step}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += float(loss)
            if step % 100 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={step}/{args.steps_per_epoch} "
                    f"loss={total / step:.5f}",
                    flush=True,
                )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "loss": total / max(1, args.steps_per_epoch),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        torch.save(model.state_dict(), output / f"epoch_{epoch + 1:02d}.pth")
        (output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
