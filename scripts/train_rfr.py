"""Fine-tune a published RFR checkpoint on challenge-format sequences."""
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
from jinsight_track1.rfr_adapter import RFRDetector


def sequence_catalog(root: Path) -> dict[str, list[tuple[list[Path], list[Path]]]]:
    groups: dict[str, list[tuple[list[Path], list[Path]]]] = defaultdict(list)
    for sequence in _sequences(root):
        images = _files(sequence / "img")
        masks = _files(sequence / "mask")
        stems = sorted(images)
        paired = [stem for stem in stems if stem in masks]
        if not paired:
            continue
        width, height = Image.open(images[paired[0]]).size
        groups[f"{width}x{height}"].append(
            ([images[stem] for stem in paired], [masks[stem] for stem in paired])
        )
    return dict(groups)


class ChallengeRFRDataset:
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
        resolutions: set[str] | None,
        balance_resolutions: bool,
    ):
        self.torch = torch
        groups = sequence_catalog(root)
        if resolutions:
            groups = {key: value for key, value in groups.items() if key in resolutions}
        if not groups:
            raise ValueError("no training sequences match the requested resolutions")
        self.groups = groups
        self.group_names = sorted(groups)
        self.group_weights = np.asarray(
            [sum(len(images) for images, _ in groups[key]) for key in self.group_names],
            dtype=float,
        )
        if balance_resolutions:
            self.group_weights[:] = 1
        self.group_weights /= self.group_weights.sum()
        self.sequence_weights = {
            key: np.asarray([len(images) for images, _ in sequences], dtype=float)
            for key, sequences in groups.items()
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

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, _):
        group = np.random.choice(self.group_names, p=self.group_weights)
        sequences = self.groups[group]
        seq_index = int(
            np.random.choice(len(sequences), p=self.sequence_weights[group])
        )
        image_paths, mask_paths = sequences[seq_index]
        start = random.randrange(len(image_paths))
        indices = [
            min(start + offset, len(image_paths) - 1)
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
        patch = self.patch_size
        if height < patch or width < patch:
            pad_h, pad_w = max(0, patch - height), max(0, patch - width)
            images = np.pad(images, ((0, 0), (0, pad_h), (0, pad_w)))
            masks = np.pad(masks, ((0, 0), (0, pad_h), (0, pad_w)))
            _, height, width = images.shape
        positive = (
            masks.max() > 0 and random.random() < self.positive_probability
        )
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

        if random.random() < .5:
            images, masks = images[:, ::-1], masks[:, ::-1]
        if random.random() < .5:
            images, masks = images[:, :, ::-1], masks[:, :, ::-1]
        if random.random() < .5:
            images, masks = images[::-1], masks[::-1]
        if random.random() < .5:
            images = images.transpose(0, 2, 1)
            masks = masks.transpose(0, 2, 1)
        images = np.ascontiguousarray(images[:, None])
        masks = np.ascontiguousarray(masks[:, None])
        return self.torch.from_numpy(images), self.torch.from_numpy(masks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("weights")
    parser.add_argument("train_root")
    parser.add_argument("output_dir")
    parser.add_argument("--model-name", default="ResUNet_RFR")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-epoch", type=int, default=10000)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--mean", type=float, default=111.47)
    parser.add_argument("--std", type=float, default=22.43)
    parser.add_argument("--positive-probability", type=float, default=.9)
    parser.add_argument("--polarity-flip-probability", type=float, default=0)
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=1,
        help="odd training-mask dilation width; centroids remain unchanged",
    )
    parser.add_argument("--resolutions")
    parser.add_argument("--balance-resolutions", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()
    if args.mask_dilation < 1 or args.mask_dilation % 2 != 1:
        raise ValueError("mask-dilation must be a positive odd integer")

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    detector = RFRDetector(
        args.source_root,
        args.weights,
        model_name=args.model_name,
        device=args.device,
        mean=args.mean,
        std=args.std,
    )
    model = detector.model.train()
    if args.from_scratch:
        model.apply(
            lambda module: module.reset_parameters()
            if hasattr(module, "reset_parameters")
            else None
        )
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, args.epochs // 2), max(2, 3 * args.epochs // 4)],
        gamma=.5,
    )
    amp_enabled = not args.no_amp and detector.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for iteration, (images, masks) in enumerate(loader, 1):
            images = images.to(detector.device, non_blocking=True)
            masks = masks.to(detector.device, non_blocking=True)
            if args.mask_dilation > 1:
                batch, length, channels, height, width = masks.shape
                masks = torch.nn.functional.max_pool2d(
                    masks.view(batch * length, channels, height, width),
                    kernel_size=args.mask_dilation,
                    stride=1,
                    padding=args.mask_dilation // 2,
                ).view(batch, length, channels, height, width)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = model.forward_train(images, masks)
            scaler.scale(loss).backward()
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
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "model_name": args.model_name,
            "mean": args.mean,
            "std": args.std,
        }
        torch.save(checkpoint, output / f"epoch_{epoch:02d}.pth.tar")
        (output / "train_metrics.json").write_text(
            json.dumps(
                {
                    "model": args.model_name,
                    "source_weights": str(Path(args.weights).resolve()),
                    "from_scratch": args.from_scratch,
                    "resolutions": sorted(dataset.group_names),
                    "balanced_resolutions": args.balance_resolutions,
                    "samples_per_epoch": args.samples_per_epoch,
                    "polarity_flip_probability": args.polarity_flip_probability,
                    "mask_dilation": args.mask_dilation,
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
