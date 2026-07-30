"""Frame-dynamics YOLO detector for extremely small video targets.

The three input channels follow the winning Anti-UAV detector recipe:
the current grayscale frame and its absolute differences to the preceding
one and two frames.  Point-like masks are converted to deliberately enlarged
boxes so a stride-4 P2 detection head receives a usable supervision signal.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .data import natural_key


def sequence_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "img").is_dir()),
        key=natural_key,
    )


def image_files(directory: Path) -> list[Path]:
    return sorted((p for p in directory.iterdir() if p.is_file()), key=natural_key)


def frame_dynamics(current: np.ndarray, previous: np.ndarray,
                   previous2: np.ndarray, diff_gain: float = 4.0) -> np.ndarray:
    """Return uint8 ``[current, |current-prev|, |current-prev2|]`` input."""
    current = np.asarray(current, dtype=np.uint8)
    previous = np.asarray(previous, dtype=np.uint8)
    previous2 = np.asarray(previous2, dtype=np.uint8)
    if current.ndim != 2 or previous.shape != current.shape or previous2.shape != current.shape:
        raise ValueError("frame_dynamics expects three equally-sized grayscale frames")
    diff1 = np.clip(
        np.abs(current.astype(np.int16) - previous.astype(np.int16)) * diff_gain,
        0, 255,
    ).astype(np.uint8)
    diff2 = np.clip(
        np.abs(current.astype(np.int16) - previous2.astype(np.int16)) * diff_gain,
        0, 255,
    ).astype(np.uint8)
    return np.stack((current, diff1, diff2), axis=-1)


def point_box_labels(mask: np.ndarray, box_size: float = 12.0) -> list[str]:
    """Convert connected components to fixed-minimum-size YOLO detection boxes."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    height, width = binary.shape
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    labels: list[str] = []
    for index in range(1, count):
        x, y = (float(v) for v in centroids[index])
        component_width = float(stats[index, cv2.CC_STAT_WIDTH])
        component_height = float(stats[index, cv2.CC_STAT_HEIGHT])
        bw = min(float(width), max(float(box_size), component_width))
        bh = min(float(height), max(float(box_size), component_height))
        labels.append(
            f"0 {x / width:.8f} {y / height:.8f} {bw / width:.8f} {bh / height:.8f}"
        )
    return labels


def prepare_frame_dynamics_dataset(
    train_root: str | Path,
    val_root: str | Path,
    output_root: str | Path,
    box_size: float = 12.0,
    diff_gain: float = 4.0,
    compression: int = 1,
    lags: tuple[int, int] = (1, 2),
    train_stride: int = 1,
) -> dict:
    """Materialize transformed train/validation images and YOLO labels."""
    output = Path(output_root)
    lag1, lag2 = (int(value) for value in lags)
    if lag1 < 1 or lag2 <= lag1:
        raise ValueError("lags must be positive and strictly increasing")
    if train_stride < 1:
        raise ValueError("train_stride must be positive")
    stats = {
        "train_root": str(train_root),
        "val_root": str(val_root),
        "output_root": str(output),
        "box_size": box_size,
        "diff_gain": diff_gain,
        "lags": [lag1, lag2],
        "train_stride": train_stride,
        "splits": {},
    }
    for split, root_value in (("train", train_root), ("val", val_root)):
        image_out = output / "images" / split
        label_out = output / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        split_stats = {"sequences": 0, "frames": 0, "objects": 0, "empty": 0}
        for sequence in sequence_dirs(root_value):
            frames = image_files(sequence / "img")
            if split == "train" and train_stride > 1:
                frames = frames[::train_stride]
            masks = {p.stem: p for p in image_files(sequence / "mask")}
            history: list[np.ndarray] = []
            for frame_path in frames:
                current = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
                if current is None:
                    raise RuntimeError(f"could not read {frame_path}")
                previous = history[-lag1] if len(history) >= lag1 else current
                previous2 = history[-lag2] if len(history) >= lag2 else current
                merged = frame_dynamics(current, previous, previous2, diff_gain)
                name = f"{sequence.name}__{frame_path.stem}"
                target_image = image_out / f"{name}.png"
                if not cv2.imwrite(
                    str(target_image), merged,
                    [cv2.IMWRITE_PNG_COMPRESSION, int(compression)],
                ):
                    raise RuntimeError(f"could not write {target_image}")
                mask_path = masks.get(frame_path.stem)
                mask = (
                    cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask_path is not None else np.zeros_like(current)
                )
                labels = point_box_labels(mask, box_size)
                (label_out / f"{name}.txt").write_text(
                    "\n".join(labels) + ("\n" if labels else ""), encoding="ascii"
                )
                split_stats["frames"] += 1
                split_stats["objects"] += len(labels)
                split_stats["empty"] += int(not labels)
                history.append(current)
                if len(history) > lag2:
                    history.pop(0)
            split_stats["sequences"] += 1
        stats["splits"][split] = split_stats
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(
        f"path: {output.resolve()}\n"
        "train: images/train\nval: images/val\nnc: 1\nnames: [target]\n",
        encoding="utf-8",
    )
    (output / "conversion_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats
