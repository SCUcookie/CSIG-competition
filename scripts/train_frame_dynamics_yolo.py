#!/usr/bin/env python3
"""Prepare and train a P2 YOLO detector on frame-dynamics inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jinsight_track1.frame_dynamics_yolo import prepare_frame_dynamics_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root")
    parser.add_argument("val_root")
    parser.add_argument("work_dir")
    parser.add_argument("--base-weights")
    parser.add_argument("--model", default="yolov8-p2.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--box-size", type=float, default=12.0)
    parser.add_argument("--diff-gain", type=float, default=4.0)
    parser.add_argument("--lags", type=int, nargs=2, default=[1, 2])
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--dataset", help="reuse an existing prepared dataset YAML")
    parser.add_argument("--skip-prepare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset).parent if args.dataset else work / "yolo_dataset"
    if not args.skip_prepare and not args.dataset:
        stats = prepare_frame_dynamics_dataset(
            args.train_root, args.val_root, dataset, args.box_size, args.diff_gain,
            lags=tuple(args.lags), train_stride=args.train_stride,
        )
        print(json.dumps(stats, indent=2), flush=True)
    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.base_weights:
        model.load(args.base_weights)
    dataset_yaml = Path(args.dataset) if args.dataset else dataset / "dataset.yaml"
    config = vars(args) | {"dataset_yaml": str(dataset_yaml)}
    (work / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model.train(
        data=str(dataset_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(work),
        name="ultralytics",
        exist_ok=True,
        amp=True,
        cache=False,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.05,
        warmup_epochs=1.0,
        patience=max(10, args.epochs),
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        fliplr=0.5,
        flipud=0.5,
        erasing=0.0,
        max_det=50,
        seed=3407,
        deterministic=True,
    )
    best = work / "ultralytics" / "weights" / "best.pt"
    report = {"best": str(best), "exists": best.exists(), "config": config}
    (work / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
