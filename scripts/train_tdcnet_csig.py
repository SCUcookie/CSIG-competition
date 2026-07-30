#!/usr/bin/env python3
"""Train the official TDCNet architecture on CSIG sequence frames.

The first temporal branch receives phase-aligned history and the second branch
receives the raw history. Tiny mask components are supervised as deliberately
expanded boxes so the stride-8 YOLOX head has a valid assignment region.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler


def natural_key(path: Path):
    import re

    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def image_files(directory: Path) -> list[Path]:
    return sorted((path for path in directory.iterdir() if path.is_file()),
                  key=natural_key)


def sequence_samples(
    root: Path,
    resolution: tuple[int, int],
    max_sequences: int | None = None,
):
    samples = []
    sequences = []
    for sequence in sorted(
        (path for path in root.iterdir() if (path / "img").is_dir()),
        key=natural_key,
    ):
        frames = image_files(sequence / "img")
        if not frames or Image.open(frames[0]).size != resolution:
            continue
        masks = {path.stem: path for path in image_files(sequence / "mask")}
        sequences.append(sequence.name)
        for index, frame in enumerate(frames):
            samples.append((frames, index, masks.get(frame.stem)))
        if max_sequences and len(sequences) >= max_sequences:
            break
    return samples, sequences


def point_targets(mask_path: Path | None, width: int, height: int,
                  box_size: float) -> np.ndarray:
    if mask_path is None:
        return np.zeros((0, 5), dtype=np.float32)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"could not read {mask_path}")
    count, _, stats, centres = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    targets = []
    for index in range(1, count):
        x, y = (float(value) for value in centres[index])
        target_width = max(box_size, float(stats[index, cv2.CC_STAT_WIDTH]))
        target_height = max(box_size, float(stats[index, cv2.CC_STAT_HEIGHT]))
        targets.append((x, y, min(width, target_width),
                        min(height, target_height), 0.0))
    return np.asarray(targets, dtype=np.float32).reshape(-1, 5)


def phase_align(history: list[np.ndarray]) -> list[np.ndarray]:
    current = history[-1].astype(np.float32)
    height, width = current.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    aligned = []
    for frame in history:
        shift, response = cv2.phaseCorrelate(
            frame.astype(np.float32), current, window
        )
        if not np.isfinite(shift).all() or response < 0.01:
            shift = (0.0, 0.0)
        matrix = np.asarray([[1.0, 0.0, shift[0]],
                             [0.0, 1.0, shift[1]]], dtype=np.float32)
        aligned.append(
            cv2.warpAffine(
                frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    return aligned


class CSIGTDCSequenceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        resolution: tuple[int, int] = (256, 256),
        context: int = 5,
        box_size: float = 12.0,
        augment: bool = False,
        max_sequences: int | None = None,
    ):
        self.width, self.height = resolution
        self.context = int(context)
        self.box_size = float(box_size)
        self.augment = augment
        self.samples, self.sequences = sequence_samples(
            Path(root), resolution, max_sequences
        )
        if not self.samples:
            raise RuntimeError(f"no {self.width}x{self.height} samples in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, sample_index):
        frames, index, mask_path = self.samples[sample_index]
        indices = [max(0, index - offset)
                   for offset in reversed(range(self.context))]
        history = [
            cv2.imread(str(frames[frame_index]), cv2.IMREAD_GRAYSCALE)
            for frame_index in indices
        ]
        if any(frame is None for frame in history):
            raise RuntimeError(f"could not read history ending at {frames[index]}")
        aligned = phase_align(history)
        stack = np.stack(aligned + history).astype(np.float32) / 255.0
        targets = point_targets(
            mask_path, self.width, self.height, self.box_size
        )

        if self.augment:
            if random.random() < 0.5:
                stack = stack[:, :, ::-1]
                targets[:, 0] = self.width - 1 - targets[:, 0]
            if random.random() < 0.5:
                stack = stack[:, ::-1, :]
                targets[:, 1] = self.height - 1 - targets[:, 1]
            if random.random() < 0.5:
                stack = 1.0 - stack
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-0.08, 0.08)
            stack = np.clip(stack * gain + bias, 0.0, 1.0)

        # TDCNet expects [C, 2 * context, H, W].
        stack = np.ascontiguousarray(np.repeat(stack[:, None], 3, axis=1))
        tensor = torch.from_numpy(stack.transpose(1, 0, 2, 3))
        return tensor, torch.from_numpy(targets.copy())


def collate(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", help="official TDCNet checkout")
    parser.add_argument("train_root")
    parser.add_argument("output")
    parser.add_argument("--resolution", default="256x256")
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--box-size", type=float, default=12.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume")
    return parser.parse_args()


def main():
    args = parse_args()
    width, height = (int(value) for value in args.resolution.split("x"))
    if width != height or width % 32:
        raise ValueError("the official TDCNet adapter requires square /32 input")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    from model.TDCNet.TDCNetwork import TDCNetwork
    from model.nets.yolo_training import YOLOLoss, weights_init

    dataset = CSIGTDCSequenceDataset(
        args.train_root, (width, height), args.context, args.box_size, True
    )
    sample_count = args.steps_per_epoch * args.batch_size
    sampler = RandomSampler(dataset, replacement=True, num_samples=sample_count)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, collate_fn=collate,
        persistent_workers=args.workers > 0,
    )
    device = torch.device(f"cuda:{args.device}")
    model = TDCNetwork(1, fp16=False, num_frame=args.context)
    weights_init(model)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    model.to(device)
    criterion = YOLOLoss(1, fp16=False, strides=[8])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.learning_rate * 0.03
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "samples": len(dataset),
        "sequences": len(dataset.sequences),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "official_source_commit": "3ba17c7b377a6fc85a95395bb7e2fa83ad2ce1e8",
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
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
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
