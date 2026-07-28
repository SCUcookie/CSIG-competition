"""Fine-tuning entry point for a pinned official DeepPro checkout."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from .deeppro_adapter import DeepProDetector, _sequences


def _dataset_config(train_root: Path, output: Path) -> Path:
    config = output / "dataset"
    config.mkdir(parents=True, exist_ok=True)
    link = config / "train"
    if link.exists() or link.is_symlink():
        if link.resolve() != train_root.resolve():
            raise ValueError(f"existing dataset link points elsewhere: {link}")
    else:
        link.symlink_to(train_root.resolve(), target_is_directory=True)
    names = [sequence.name for sequence in _sequences(train_root)]
    (config / "train.txt").write_text("\n".join(names) + "\n", encoding="ascii")
    return config


def _soft_iou_focal_loss(logits, target, focal_weight: float):
    import torch
    import torch.nn.functional as functional

    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    union = (
        probability.sum(dim=(1, 2, 3))
        + target.sum(dim=(1, 2, 3))
        - intersection
    )
    # A small smooth term gives empty background crops a useful gradient;
    # the official zero-smooth loss is constant on all-empty crops.
    soft_iou = 1.0 - ((intersection + 1e-6) / (union + 1e-6)).mean()
    bce = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    point_probability = probability * target + (1.0 - probability) * (1.0 - target)
    alpha = 0.95 * target + 0.05 * (1.0 - target)
    focal = (alpha * (1.0 - point_probability).square() * bce).mean()
    return soft_iou + focal_weight * focal, soft_iou.detach(), focal.detach()


def train_deeppro(
    source_root: str | Path,
    initial_weights: str | Path,
    train_root: str | Path,
    output_dir: str | Path,
    devices: str = "3,4,5,6",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    sample_rate: float = .04,
    patch_size: int = 128,
    sequence_length: int = 40,
    workers: int = 8,
    focal_weight: float = .5,
    seed: int = 46,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    if epochs <= 0 or batch_size <= 0 or not (0 < sample_rate <= 1):
        raise ValueError("epochs and batch_size must be positive; sample_rate in (0,1]")
    device_ids = [int(value.strip()) for value in devices.split(",") if value.strip()]
    if not device_ids:
        raise ValueError("at least one CUDA device is required")
    if any(value >= torch.cuda.device_count() for value in device_ids):
        raise ValueError(
            f"requested devices {device_ids}, but {torch.cuda.device_count()} are visible"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = _dataset_config(Path(train_root), output)

    detector = DeepProDetector(
        source_root,
        initial_weights,
        device=str(device_ids[0]),
        sequence_length=sequence_length,
    )
    # The pinned source is now on sys.path via DeepProDetector.
    from data_utils.TrainDataLoader import TrainIRSeqDataLoader

    dataset = TrainIRSeqDataLoader(
        "SatVideoIRSDT",
        data_root=str(config),
        seq_len=sequence_length,
        sample_rate=sample_rate,
        patch_size=patch_size,
        transform=None,
    )
    # NumPy 2 rejects ``np.random.choice(list_of_variable_length_lists)`` in
    # the official loader. A genuine 1-D object array preserves its intended
    # weighted choice semantics without changing the pinned source checkout.
    sample_array = np.empty(len(dataset.samplelist), dtype=object)
    sample_array[:] = dataset.samplelist
    dataset.samplelist = sample_array
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    model = detector.model
    primary = torch.device(f"cuda:{device_ids[0]}")
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    model.to(primary)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=learning_rate * .1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    start_epoch = 0
    last_checkpoint = output / "last_model.pth"
    history_path = output / "train_history.jsonl"
    if last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
        target = model.module if isinstance(model, torch.nn.DataParallel) else model
        target.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1

    config_report = {
        "source_root": str(Path(source_root).resolve()),
        "initial_weights": str(Path(initial_weights).resolve()),
        "train_root": str(Path(train_root).resolve()),
        "output_dir": str(output.resolve()),
        "devices": device_ids,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "sample_rate": sample_rate,
        "patch_size": patch_size,
        "sequence_length": sequence_length,
        "workers": workers,
        "focal_weight": focal_weight,
        "seed": seed,
        "training_samples_per_epoch": len(dataset),
        "steps_per_epoch": len(loader),
        "train_sequences": len(_sequences(Path(train_root))),
    }
    (output / "train_config.json").write_text(
        json.dumps(config_report, indent=2), encoding="utf-8"
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        totals = {"loss": 0.0, "soft_iou": 0.0, "focal": 0.0}
        for step, (images, targets) in enumerate(loader, 1):
            images = images.float().to(primary, non_blocking=True)
            targets = targets.float().to(primary, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                _, logits = model(images)
                loss, soft_iou, focal = _soft_iou_focal_loss(
                    logits, targets, focal_weight
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            totals["soft_iou"] += float(soft_iou)
            totals["focal"] += float(focal)
            if step % 50 == 0 or step == len(loader):
                print(
                    f"deeppro-train epoch={epoch + 1}/{epochs} "
                    f"step={step}/{len(loader)} loss={totals['loss'] / step:.6f}",
                    flush=True,
                )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "loss": totals["loss"] / max(1, len(loader)),
            "soft_iou_loss": totals["soft_iou"] / max(1, len(loader)),
            "focal_loss": totals["focal"] / max(1, len(loader)),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        target = model.module if isinstance(model, torch.nn.DataParallel) else model
        state = {
            "epoch": epoch,
            "model_state_dict": target.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "training_config": config_report,
        }
        epoch_checkpoint = output / f"epoch_{epoch + 1:02d}_model.pth"
        torch.save(state, epoch_checkpoint)
        torch.save(state, last_checkpoint)
        print(json.dumps(row), flush=True)

    return {
        **config_report,
        "start_epoch": start_epoch,
        "completed_epochs": epochs,
        "last_checkpoint": str(last_checkpoint),
        "history": str(history_path),
    }
