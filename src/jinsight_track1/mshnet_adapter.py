"""Adapter and validation helpers for the official CVPR 2024 MSHNet."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .deeppro_adapter import (
    _files,
    _load_sequence_images,
    _sequences,
    _summary,
    component_points,
)
from .evaluation import point_metrics
from .postprocess import centroids


class MSHNetUnavailable(RuntimeError):
    pass


def load_mshnet_model(source_root: str | Path, *, input_channels: int = 1):
    source = Path(source_root).resolve()
    model_path = source / "model" / "MSHNet.py"
    if not model_path.is_file():
        raise MSHNetUnavailable(f"MSHNet source not found: {model_path}")
    module_name = "_jinsight_external_mshnet"
    module = sys.modules.get(module_name)
    if module is None or Path(module.__file__).resolve() != model_path:
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        if spec is None or spec.loader is None:
            raise MSHNetUnavailable(f"cannot import MSHNet from {model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.MSHNet(input_channels)


class MSHNetDetector:
    def __init__(
        self,
        source_root: str | Path,
        weights: str | Path,
        *,
        device: str = "0",
        batch_size: int = 8,
        temporal_radius: int | None = None,
        mean: float = 111.47,
        std: float = 22.43,
        adaptive_normalization: bool = False,
        invert_normalized: bool = False,
    ):
        try:
            import torch
        except ImportError as exc:
            raise MSHNetUnavailable("PyTorch is required for MSHNet") from exc
        checkpoint_path = Path(weights).resolve()
        if not checkpoint_path.is_file():
            raise MSHNetUnavailable(f"MSHNet weights not found: {checkpoint_path}")
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        if temporal_radius is None:
            temporal_radius = int(checkpoint.get("temporal_radius", 0))
        if temporal_radius < 0:
            raise ValueError("temporal_radius must be non-negative")
        model = load_mshnet_model(
            source_root, input_channels=2 * temporal_radius + 1
        )
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        torch_device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        self.model = model.to(torch_device).eval()
        self.torch = torch
        self.device = torch_device
        self.source_root = Path(source_root).resolve()
        self.weights = checkpoint_path
        self.batch_size = int(batch_size)
        self.temporal_radius = int(temporal_radius)
        self.mean = float(mean)
        self.std = float(std)
        self.adaptive_normalization = bool(adaptive_normalization)
        self.invert_normalized = bool(invert_normalized)

    def predict(self, frames: np.ndarray) -> np.ndarray:
        values = np.asarray(frames)
        if values.ndim != 3:
            raise ValueError("frames must have shape [time, height, width]")
        count, height, width = values.shape
        if not count:
            return np.empty((0, height, width), dtype=np.float32)
        normalized = values.astype(np.float32, copy=False)
        if self.adaptive_normalization:
            means = normalized.mean(axis=(1, 2), keepdims=True)
            stds = normalized.std(axis=(1, 2), keepdims=True).clip(min=1.0)
            normalized = (normalized - means) / stds
        else:
            normalized = (normalized - self.mean) / self.std
        if self.invert_normalized:
            normalized = -normalized
        padded_height = math.ceil(height / 16) * 16
        padded_width = math.ceil(width / 16) * 16
        offsets = range(-self.temporal_radius, self.temporal_radius + 1)
        indices = np.arange(count)
        stacked = np.stack(
            [normalized[np.clip(indices + offset, 0, count - 1)] for offset in offsets],
            axis=1,
        )
        padded = np.zeros(
            (count, len(tuple(offsets)), padded_height, padded_width),
            dtype=np.float32,
        )
        padded[:, :, :height, :width] = stacked
        outputs = np.empty((count, height, width), dtype=np.float32)
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, count, self.batch_size):
                tensor = torch.from_numpy(
                    padded[start : start + self.batch_size]
                ).to(self.device)
                with torch.amp.autocast(
                    "cuda", enabled=self.device.type == "cuda"
                ):
                    _, logits = self.model(tensor, False)
                # Keep the sigmoid in float32. Half precision rounds strong
                # logits to exactly one, making the high-confidence tail
                # impossible to calibrate for tiny-target point detection.
                probability = torch.sigmoid(logits.float())
                outputs[start : start + len(tensor)] = (
                    probability[:, 0, :height, :width].float().cpu().numpy()
                )
        return outputs


def evaluate_mshnet(
    source_root: str | Path,
    weights: str | Path,
    val_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "0",
    thresholds: list[float] | None = None,
    radius: float = 2.0,
    max_sequences: int | None = None,
    max_frames: int | None = None,
    resolutions: set[str] | None = None,
    batch_size: int = 8,
    temporal_radius: int | None = None,
    mean: float = 111.47,
    std: float = 22.43,
    adaptive_normalization: bool = False,
    invert_normalized: bool = False,
    min_area: int = 1,
    max_area: int | None = None,
    centroid_mode: str = "binary",
) -> dict:
    thresholds = sorted(
        set(thresholds or [1e-4, 1e-3, .01, .03, .1, .3, .5, .7, .9])
    )
    detector = MSHNetDetector(
        source_root,
        weights,
        device=device,
        batch_size=batch_size,
        temporal_radius=temporal_radius,
        mean=mean,
        std=std,
        adaptive_normalization=adaptive_normalization,
        invert_normalized=invert_normalized,
    )
    sequences = _sequences(Path(val_root))
    if resolutions:
        selected = []
        for sequence in sequences:
            first = next(iter(_files(sequence / "img").values()))
            width, height = Image.open(first).size
            if f"{width}x{height}" in resolutions:
                selected.append(sequence)
        sequences = selected
    if max_sequences:
        sequences = sequences[:max_sequences]

    totals = {value: defaultdict(float) for value in thresholds}
    by_resolution = {value: {} for value in thresholds}
    frame_count = 0
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for sequence_index, sequence in enumerate(sequences, 1):
        stems, frames = _load_sequence_images(sequence, max_frames)
        probabilities = detector.predict(frames)
        masks = _files(sequence / "mask")
        for index, stem in enumerate(stems):
            truth = []
            if stem in masks:
                mask = np.asarray(Image.open(masks[stem]).convert("L"))
                truth = [(point.x, point.y) for point in centroids(mask, .5, 1)]
            height, width = frames[index].shape
            resolution = f"{width}x{height}"
            for threshold in thresholds:
                predicted = component_points(
                    probabilities[index],
                    threshold,
                    min_area=min_area,
                    max_area=max_area,
                    centroid_mode=centroid_mode,
                )
                metrics = point_metrics(predicted, truth, radius)
                bucket = by_resolution[threshold].setdefault(
                    resolution, defaultdict(float)
                )
                for key in ("tp", "fp", "fn"):
                    totals[threshold][key] += metrics[key]
                    bucket[key] += metrics[key]
            frame_count += 1
        print(
            f"mshnet-eval {sequence_index}/{len(sequences)} "
            f"sequence={sequence.name} frames={frame_count}",
            flush=True,
        )

    sweep = [_summary(totals[value], value) for value in thresholds]
    best = max(sweep, key=lambda row: (row["f1"], row["recall"]))
    resolution_names = sorted(
        {name for rows in by_resolution.values() for name in rows}
    )
    resolution_sweeps = {
        name: [
            _summary(by_resolution[value].get(name, defaultdict(float)), value)
            for value in thresholds
        ]
        for name in resolution_names
    }
    report = {
        "model": "MSHNet",
        "source_root": str(detector.source_root),
        "weights": str(detector.weights),
        "sequences": len(sequences),
        "frames": frame_count,
        **best,
        "best_threshold": best["threshold"],
        "sweep": sweep,
        "best_threshold_by_resolution": {
            name: max(rows, key=lambda row: (row["f1"], row["recall"]))
            for name, rows in resolution_sweeps.items()
        },
        "resolution_sweeps": resolution_sweeps,
        "radius_pixels": radius,
        "temporal_radius": detector.temporal_radius,
        "mean": mean,
        "std": std,
        "adaptive_normalization": adaptive_normalization,
        "invert_normalized": invert_normalized,
        "component_min_area": min_area,
        "component_max_area": max_area,
        "centroid_mode": centroid_mode,
        "metric_status": "local point-matching proxy; not official scorer",
    }
    (output / "val_proxy_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
