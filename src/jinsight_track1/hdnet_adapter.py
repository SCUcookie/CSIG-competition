"""Adapter for the official TGRS 2025 HDNet implementation."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


class HDNetUnavailable(RuntimeError):
    pass


def load_hdnet_model(source_root: str | Path, input_channels: int = 3):
    source = Path(source_root).resolve()
    model_path = source / "model" / "HDNet.py"
    if not model_path.is_file():
        raise HDNetUnavailable(f"HDNet source not found: {model_path}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    name = "_jinsight_external_hdnet"
    module = sys.modules.get(name)
    if module is None or Path(module.__file__).resolve() != model_path:
        spec = importlib.util.spec_from_file_location(name, model_path)
        if spec is None or spec.loader is None:
            raise HDNetUnavailable(f"cannot import HDNet from {model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.HDNet(input_channels)


def temporal_difference_channels(frames: np.ndarray) -> np.ndarray:
    """Convert [T,H,W] frames to [T,3,H,W]: center and two signed differences."""
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("frames must have shape [time, height, width]")
    indices = np.arange(len(values))
    previous = values[np.maximum(indices - 1, 0)]
    following = values[np.minimum(indices + 1, len(values) - 1)]
    return np.stack((values, values - previous, values - following), axis=1)


class HDNetDetector:
    def __init__(
        self,
        source_root: str | Path,
        weights: str | Path,
        *,
        device: str = "0",
        batch_size: int = 8,
        adaptive_normalization: bool = True,
        mean: float = 111.47,
        std: float = 22.43,
    ):
        import torch

        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        model = load_hdnet_model(source_root, input_channels=3)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        self.device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        self.model = model.to(self.device).eval()
        self.torch = torch
        self.batch_size = int(batch_size)
        self.adaptive_normalization = bool(adaptive_normalization)
        self.mean = float(mean)
        self.std = float(std)

    def predict(self, frames: np.ndarray) -> np.ndarray:
        values = np.asarray(frames, dtype=np.float32)
        if self.adaptive_normalization:
            means = values.mean(axis=(1, 2), keepdims=True)
            stds = values.std(axis=(1, 2), keepdims=True).clip(min=1.0)
            values = (values - means) / stds
        else:
            values = (values - self.mean) / self.std
        channels = temporal_difference_channels(values)
        count, _, height, width = channels.shape
        padded_height = math.ceil(height / 16) * 16
        padded_width = math.ceil(width / 16) * 16
        padded = np.zeros(
            (count, 3, padded_height, padded_width), dtype=np.float32
        )
        padded[:, :, :height, :width] = channels
        result = np.empty((count, height, width), dtype=np.float32)
        with self.torch.inference_mode():
            for start in range(0, count, self.batch_size):
                tensor = self.torch.from_numpy(
                    padded[start : start + self.batch_size]
                ).to(self.device)
                with self.torch.amp.autocast(
                    "cuda", enabled=self.device.type == "cuda"
                ):
                    _, logits = self.model(tensor, True)
                result[start : start + len(tensor)] = (
                    self.torch.sigmoid(logits.float())[:, 0, :height, :width]
                    .cpu()
                    .numpy()
                )
        return result
