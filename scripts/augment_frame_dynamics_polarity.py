#!/usr/bin/env python3
"""Add paired current-channel inversion samples to a prepared YOLO dataset."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--compression", type=int, default=1)
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    root = Path(args.dataset_root)
    source_images = sorted((root / "images" / "train").glob("*.png"))
    image_out = root / "images" / "train_invert"
    label_out = root / "labels" / "train_invert"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    written = 0
    for source in source_images[::args.stride]:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read {source}")
        # cv2 preserves the stored channel order. Channel zero is the current
        # grayscale frame; channels one and two are absolute temporal deltas.
        image[..., 0] = 255 - image[..., 0]
        target = image_out / f"{source.stem}__inv.png"
        if not cv2.imwrite(
            str(target), image,
            [cv2.IMWRITE_PNG_COMPRESSION, args.compression],
        ):
            raise RuntimeError(f"could not write {target}")
        label = root / "labels" / "train" / f"{source.stem}.txt"
        shutil.copyfile(label, label_out / f"{source.stem}__inv.txt")
        written += 1
    yaml = root / "dataset_polarity.yaml"
    yaml.write_text(
        f"path: {root.resolve()}\n"
        "train: [images/train, images/train_invert]\n"
        "val: images/val\nnc: 1\nnames: [target]\n",
        encoding="utf-8",
    )
    report = {
        "source_images": len(source_images),
        "stride": args.stride,
        "inverted_images": written,
        "dataset_yaml": str(yaml),
    }
    (root / "polarity_augmentation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
