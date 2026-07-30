"""Adapter and validation helpers for the published LVNet architecture."""
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


class LVNetUnavailable(RuntimeError):
    pass


def load_lvnet_model(source_root: str | Path, *, num_frames: int = 4):
    """Load LVNet from its pinned source checkout without modifying sys.path."""
    source = Path(source_root).resolve()
    model_path = source / "model.py"
    if not model_path.is_file():
        raise LVNetUnavailable(f"LVNet model.py not found: {model_path}")
    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
    except ImportError as exc:
        raise LVNetUnavailable("LVNet requires PyTorch, timm, and einops") from exc

    module_name = "_jinsight_external_lvnet"
    module = sys.modules.get(module_name)
    if module is None or Path(module.__file__).resolve() != model_path:
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        if spec is None or spec.loader is None:
            raise LVNetUnavailable(f"cannot import LVNet from {model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.LVNet(num_frame=num_frames)


class ScaleLocationLoss:
    """Small-target loss inspired by MSHNet's published SLS-IoU objective.

    The published formulation is extended to video logits and made safe for
    target-free frames. A balanced focal term supplies useful gradients before
    the predicted foreground mass has converged.
    """

    def __init__(
        self,
        torch,
        focal_weight: float = 0.25,
        location_weight: float = 0.5,
    ):
        self.torch = torch
        self.focal_weight = float(focal_weight)
        self.location_weight = float(location_weight)

    def __call__(self, logits, target, *, use_location: bool = True):
        torch = self.torch
        probability = torch.sigmoid(logits)
        batch, channels, depth, height, width = probability.shape
        pred = probability.permute(0, 2, 1, 3, 4).reshape(
            batch * depth, channels, height, width
        )
        truth = target.permute(0, 2, 1, 3, 4).reshape(
            batch * depth, channels, height, width
        )
        dims = (1, 2, 3)
        intersection = (pred * truth).sum(dims)
        pred_mass = pred.sum(dims)
        truth_mass = truth.sum(dims)
        union = pred_mass + truth_mass - intersection
        iou = (intersection + 1.0) / (union + 1.0)

        # MSHNet's scale-sensitive multiplier: poor foreground-size agreement
        # lowers the rewarded IoU even when the masks overlap.
        distance = ((pred_mass - truth_mass) / 2.0).square()
        scale = (torch.minimum(pred_mass, truth_mass) + distance + 1e-6) / (
            torch.maximum(pred_mass, truth_mass) + distance + 1e-6
        )
        sls_iou = 1.0 - (scale * iou).mean()

        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        pt = probability * target + (1.0 - probability) * (1.0 - target)
        alpha = 0.95 * target + 0.05 * (1.0 - target)
        focal = (alpha * (1.0 - pt).square() * bce).mean()

        location = logits.new_zeros(())
        if use_location:
            positive = truth_mass > 0
            if positive.any():
                x_grid = torch.linspace(0, 1, width, device=logits.device).view(
                    1, 1, 1, width
                )
                y_grid = torch.linspace(0, 1, height, device=logits.device).view(
                    1, 1, height, 1
                )
                pred_den = pred.sum(dims).clamp_min(1e-6)
                truth_den = truth.sum(dims).clamp_min(1e-6)
                pred_x = (pred * x_grid).sum(dims) / pred_den
                pred_y = (pred * y_grid).sum(dims) / pred_den
                truth_x = (truth * x_grid).sum(dims) / truth_den
                truth_y = (truth * y_grid).sum(dims) / truth_den
                location = torch.sqrt(
                    (pred_x[positive] - truth_x[positive]).square()
                    + (pred_y[positive] - truth_y[positive]).square()
                    + 1e-8
                ).mean()
        return sls_iou + self.focal_weight * focal + self.location_weight * location


class LVNetDetector:
    def __init__(
        self,
        source_root: str | Path,
        weights: str | Path,
        *,
        device: str = "0",
        num_frames: int = 4,
        temporal_stride: int = 4,
        mean: float = 111.47,
        std: float = 22.43,
        adaptive_normalization: bool = False,
    ):
        try:
            import torch
        except ImportError as exc:
            raise LVNetUnavailable("PyTorch is required for LVNet") from exc
        if not torch.cuda.is_available() and device != "cpu":
            raise LVNetUnavailable("CUDA is unavailable")
        checkpoint_path = Path(weights).resolve()
        if not checkpoint_path.is_file():
            raise LVNetUnavailable(f"LVNet weights not found: {checkpoint_path}")
        model = load_lvnet_model(source_root, num_frames=num_frames)
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state)
        torch_device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        self.model = model.to(torch_device).eval()
        self.torch = torch
        self.device = torch_device
        self.source_root = Path(source_root).resolve()
        self.weights = checkpoint_path
        self.num_frames = int(num_frames)
        self.temporal_stride = int(temporal_stride)
        self.mean = float(mean)
        self.std = float(std)
        self.adaptive_normalization = bool(adaptive_normalization)

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
        normalized = (values.astype(np.float32, copy=False) - mean) / std
        padded_height = math.ceil(height / 32) * 32
        padded_width = math.ceil(width / 32) * 32
        sums = np.zeros((count, height, width), dtype=np.float32)
        counts = np.zeros(count, dtype=np.float32)
        starts = list(range(0, count, self.temporal_stride))
        if starts[-1] + self.num_frames < count:
            starts.append(count - self.num_frames)

        torch = self.torch
        amp_enabled = self.device.type == "cuda"
        with torch.inference_mode():
            for start in starts:
                indices = [
                    min(start + offset, count - 1)
                    for offset in range(self.num_frames)
                ]
                clip = np.zeros(
                    (self.num_frames, padded_height, padded_width), dtype=np.float32
                )
                clip[:, :height, :width] = normalized[indices]
                tensor = torch.from_numpy(clip[None, None]).to(self.device)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    probability = torch.sigmoid(self.model(tensor))
                prediction = (
                    probability[0, 0, :, :height, :width].float().cpu().numpy()
                )
                for offset, index in enumerate(indices):
                    # Repeated padding frames must not receive extra weight.
                    if start + offset < count:
                        sums[index] += prediction[offset]
                        counts[index] += 1
        return sums / counts[:, None, None].clip(min=1)


def evaluate_lvnet(
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
    num_frames: int = 4,
    temporal_stride: int = 4,
    mean: float = 111.47,
    std: float = 22.43,
    adaptive_normalization: bool = False,
    min_area: int = 1,
    max_area: int | None = None,
    centroid_mode: str = "binary",
) -> dict:
    thresholds = sorted(
        set(thresholds or [1e-4, 1e-3, .01, .03, .1, .3, .5, .7, .9])
    )
    detector = LVNetDetector(
        source_root,
        weights,
        device=device,
        num_frames=num_frames,
        temporal_stride=temporal_stride,
        mean=mean,
        std=std,
        adaptive_normalization=adaptive_normalization,
    )
    sequences = _sequences(Path(val_root))
    if resolutions:
        sequences = [
            sequence
            for sequence in sequences
            if (
                lambda size: f"{size[0]}x{size[1]}" in resolutions
            )(Image.open(next(iter(_files(sequence / "img").values()))).size)
        ]
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
        }
        (output / "progress.json").write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )
        print(
            f"lvnet-eval {sequence_index}/{len(sequences)} "
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
        "model": "LVNet",
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
        "num_frames": num_frames,
        "temporal_stride": temporal_stride,
        "mean": mean,
        "std": std,
        "adaptive_normalization": adaptive_normalization,
        "component_min_area": min_area,
        "component_max_area": max_area,
        "centroid_mode": centroid_mode,
        "metric_status": "local point-matching proxy; not official scorer",
    }
    (output / "val_proxy_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
