"""Adapter for the official Recurrent Feature Refinement (RFR) repository.

The upstream project vendors a legacy DCNv2 extension that does not build on
current PyTorch releases.  This adapter supplies the same parameter layout
with torchvision's maintained modulated deformable convolution operator, so
the published checkpoints can be evaluated without rewriting their weights.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
import types
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


class RFRUnavailable(RuntimeError):
    pass


def _install_torchvision_dcn_compat(source: Path, torch) -> None:
    """Install an import-compatible replacement for upstream's old DCNv2."""
    try:
        from torchvision.ops import deform_conv2d
    except (ImportError, RuntimeError) as exc:
        raise RFRUnavailable(
            "RFR requires a torchvision build matching the installed PyTorch/CUDA"
        ) from exc

    nn = torch.nn
    pair = torch.nn.modules.utils._pair

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
            if in_channels % groups or out_channels % groups:
                raise ValueError("channels must be divisible by groups")
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
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            offset_channels = (
                deformable_groups
                * 3
                * self.kernel_size[0]
                * self.kernel_size[1]
            )
            self.conv_offset = nn.Conv2d(
                in_channels,
                offset_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=True,
            )
            self.conv_offset.lr_mult = lr_mult
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            if hasattr(self, "conv_offset"):
                nn.init.zeros_(self.conv_offset.weight)
                nn.init.zeros_(self.conv_offset.bias)

        def forward(self, input, offset_features):
            offset_and_mask = self.conv_offset(offset_features)
            offset_y, offset_x, mask = torch.chunk(offset_and_mask, 3, dim=1)
            offset = torch.cat((offset_y, offset_x), dim=1)
            return deform_conv2d(
                input,
                offset,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=torch.sigmoid(mask),
            )

    package_name = "model.dcn.modules"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source / "model" / "dcn" / "modules")]
    compat_name = f"{package_name}.deform_conv"
    compat = types.ModuleType(compat_name)
    compat.DeformConv = DeformConv
    compat.ConvOffset2d = DeformConv
    compat.DeformConvPack = DeformConv
    compat._DeformConv = None
    package.DeformConv = DeformConv
    package.ConvOffset2d = DeformConv
    package.DeformConvPack = DeformConv
    sys.modules[package_name] = package
    sys.modules[compat_name] = compat


class RFRDetector:
    """Run a published RFR checkpoint frame by frame at native resolution."""

    def __init__(
        self,
        source_root: str | Path,
        weights: str | Path,
        model_name: str = "ResUNet_RFR",
        device: str = "0",
        mean: float = 111.47,
        std: float = 22.43,
        adaptive_normalization: bool = False,
        invert_normalized: bool = False,
    ):
        source = Path(source_root).resolve()
        checkpoint_path = Path(weights).resolve()
        if (source / "codes").is_dir():
            source = source / "codes"
        if not (source / "model" / "RFR_framework.py").is_file():
            raise RFRUnavailable(f"RFR source not found: {source}")
        if not checkpoint_path.is_file():
            raise RFRUnavailable(f"RFR weights not found: {checkpoint_path}")
        try:
            import torch
        except ImportError as exc:
            raise RFRUnavailable("PyTorch is required for RFR") from exc
        if not torch.cuda.is_available() and device != "cpu":
            raise RFRUnavailable("CUDA is unavailable")

        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        _install_torchvision_dcn_compat(source, torch)
        # Avoid accidentally retaining a differently pinned RFR checkout.
        sys.modules.pop("net", None)
        net_module = importlib.import_module("net")
        model = net_module.Net(model_name=model_name)
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state)

        torch_device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        self.model = model.to(torch_device).eval()
        self.torch = torch
        self.device = torch_device
        self.source_root = source
        self.weights = checkpoint_path
        self.model_name = model_name
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
        if self.adaptive_normalization:
            mean = float(values.mean())
            std = max(float(values.std()), 1.0)
        else:
            mean, std = self.mean, self.std
        padded_height = math.ceil(height / 32) * 32
        padded_width = math.ceil(width / 32) * 32
        outputs = np.empty((count, height, width), dtype=np.float32)
        feature = None
        torch = self.torch
        with torch.inference_mode():
            for index, frame in enumerate(values):
                normalized = (frame.astype(np.float32, copy=False) - mean) / std
                if self.invert_normalized:
                    normalized = -normalized
                padded = np.zeros((padded_height, padded_width), dtype=np.float32)
                padded[:height, :width] = normalized
                tensor = torch.from_numpy(padded[None, None]).to(self.device)
                probability, feature = self.model.forward_test(tensor, feature)
                outputs[index] = (
                    probability[0, 0, :height, :width].float().cpu().numpy()
                )
        return outputs


def evaluate_rfr(
    source_root: str | Path,
    weights: str | Path,
    val_root: str | Path,
    output_dir: str | Path,
    model_name: str = "ResUNet_RFR",
    device: str = "0",
    thresholds: list[float] | None = None,
    radius: float = 2.0,
    max_sequences: int | None = None,
    max_frames: int | None = None,
    resolutions: set[str] | None = None,
    mean: float = 111.47,
    std: float = 22.43,
    adaptive_normalization: bool = False,
    invert_normalized: bool = False,
    min_area: int = 1,
    max_area: int | None = None,
    centroid_mode: str = "binary",
) -> dict:
    thresholds = sorted(set(thresholds or [1e-4, 1e-3, 1e-2, .1, .3, .5, .7, .9]))
    detector = RFRDetector(
        source_root,
        weights,
        model_name=model_name,
        device=device,
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
        progress = {
            "status": "running",
            "completed_sequences": sequence_index,
            "total_sequences": len(sequences),
            "last_sequence": sequence.name,
            "frames": frame_count,
            "sweep": [_summary(totals[value], value) for value in thresholds],
        }
        (output / "progress.json").write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )
        print(
            f"rfr-eval {sequence_index}/{len(sequences)} "
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
        "model": detector.model_name,
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
