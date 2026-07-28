"""Pinned-source adapter for the official DeepPro temporal detector.

The challenge repository intentionally does not vendor DeepPro.  This module
loads a user-supplied checkout and checkpoint, runs temporal/spatial tiled
inference, and converts probability maps to point detections.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .data import natural_key
from .evaluation import point_metrics
from .postprocess import centroids
from .submission import package, write_txt
from .types import SequencePrediction, TrackPoint

try:
    import cv2
except ImportError:  # pragma: no cover - scipy fallback is covered
    cv2 = None


class DeepProUnavailable(RuntimeError):
    pass


def _sequences(root: Path) -> list[Path]:
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "img").is_dir()),
        key=natural_key,
    )


def _files(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in directory.iterdir() if p.is_file()}


def temporal_windows(frame_count: int, length: int = 40, overlap: int = 4) -> list[tuple[int, int]]:
    """Match the official loader's end-aligned, overlapping temporal windows."""
    if frame_count <= 0:
        return []
    if not (0 <= overlap < length):
        raise ValueError("overlap must satisfy 0 <= overlap < length")
    if frame_count <= length:
        return [(0, frame_count)]
    step = length - overlap
    count = math.ceil((frame_count - overlap) / step)
    result = []
    for index in range(count):
        end = min(frame_count, (index + 1) * step + overlap)
        start = max(0, end - length)
        if not result or result[-1] != (start, end):
            result.append((start, end))
    return result


def component_points(probability: np.ndarray, threshold: float,
                     min_area: int = 1, max_area: int | None = None) -> list[tuple[float, float]]:
    """Return (x, y) centroids of 8-connected thresholded components."""
    binary = np.asarray(probability) >= threshold
    if cv2 is not None:
        count, _, stats, centres = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
        )
        found = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area or (max_area is not None and area > max_area):
                continue
            x, y = centres[label]
            if np.isfinite((x, y)).all():
                found.append((float(x), float(y)))
        return found
    labels, count = ndimage.label(binary, np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return []
    ids = np.arange(1, count + 1)
    areas = np.asarray(ndimage.sum(binary, labels, ids))
    centres = ndimage.center_of_mass(binary, labels, ids)
    found = []
    for area, centre in zip(areas, centres):
        if area < min_area or (max_area is not None and area > max_area):
            continue
        y, x = centre
        if np.isfinite((x, y)).all():
            found.append((float(x), float(y)))
    return found


class DeepProDetector:
    """Load and run an official DeepPro/DeepPro-Plus checkpoint."""

    def __init__(
        self,
        source_root: str | Path,
        weights: str | Path,
        device: str = "0",
        model_name: str = "DeepPro-Plus",
        sequence_length: int = 40,
        temporal_overlap: int = 4,
        tile_size: int = 256,
        tile_halo: int = 12,
        mean: float = 111.47,
        std: float = 22.43,
    ):
        source = Path(source_root).resolve()
        checkpoint_path = Path(weights).resolve()
        model_file = source / "networks" / "models" / f"{model_name}.py"
        if not model_file.is_file():
            raise DeepProUnavailable(f"DeepPro model source not found: {model_file}")
        if not checkpoint_path.is_file():
            raise DeepProUnavailable(f"DeepPro weights not found: {checkpoint_path}")
        if tile_size <= 2 * tile_halo:
            raise ValueError("tile_size must be greater than twice tile_halo")

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on GPU host
            raise DeepProUnavailable("PyTorch is required for DeepPro") from exc
        if not torch.cuda.is_available() and device != "cpu":
            raise DeepProUnavailable("CUDA is unavailable; pass device='cpu' only for a smoke run")

        # The official model imports ``networks.*`` from its checkout.  Pin the
        # explicit checkout at the front instead of discovering arbitrary code.
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location(
            f"_jinsight_deeppro_{model_name.replace('-', '_')}", model_file
        )
        if spec is None or spec.loader is None:
            raise DeepProUnavailable(f"cannot import DeepPro model: {model_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        torch_device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        model = module.detector(1, sequence_length, sequence_length)
        try:
            checkpoint = torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=True
            )
        except Exception:
            # Official checkpoints include a NumPy scalar in optimizer state,
            # which older safe unpicklers reject. The path is an explicitly
            # pinned, user-supplied official checkpoint rather than discovery.
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        if state and all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        model.load_state_dict(state)
        self.model = model.to(torch_device).eval()
        self.torch = torch
        self.device = torch_device
        self.source_root = source
        self.weights = checkpoint_path
        self.model_name = model_name
        self.sequence_length = sequence_length
        self.temporal_overlap = temporal_overlap
        self.tile_size = tile_size
        self.tile_halo = tile_halo
        self.mean = float(mean)
        self.std = float(std)

    def _spatial_tiles(self, height: int, width: int):
        core = self.tile_size - 2 * self.tile_halo
        for y0 in range(0, height, core):
            y1 = min(height, y0 + core)
            iy0, iy1 = max(0, y0 - self.tile_halo), min(height, y1 + self.tile_halo)
            for x0 in range(0, width, core):
                x1 = min(width, x0 + core)
                ix0, ix1 = max(0, x0 - self.tile_halo), min(width, x1 + self.tile_halo)
                yield (y0, y1, x0, x1), (iy0, iy1, ix0, ix1)

    def predict(self, frames: np.ndarray) -> np.ndarray:
        """Predict a full sequence as float32 probabilities in original pixels."""
        values = np.asarray(frames)
        if values.ndim != 3:
            raise ValueError("frames must have shape [time, height, width]")
        count, height, width = values.shape
        merged = np.zeros((count, height, width), dtype=np.float32)
        coverage = np.zeros(count, dtype=np.uint16)
        torch = self.torch

        with torch.inference_mode():
            for start, end in temporal_windows(
                count, self.sequence_length, self.temporal_overlap
            ):
                valid = end - start
                chunk = values[start:end].astype(np.float32, copy=False)
                if valid < self.sequence_length:
                    # Training pads *after* normalization, so padded temporal
                    # slots must equal the training mean in raw-image space.
                    pad = np.full(
                        (self.sequence_length - valid, height, width),
                        self.mean,
                        dtype=np.float32,
                    )
                    chunk = np.concatenate((chunk, pad), axis=0)
                window = np.zeros((valid, height, width), dtype=np.float32)
                for core, incoming in self._spatial_tiles(height, width):
                    y0, y1, x0, x1 = core
                    iy0, iy1, ix0, ix1 = incoming
                    tile = (chunk[:, iy0:iy1, ix0:ix1] - self.mean) / self.std
                    tensor = torch.from_numpy(tile[None, None]).to(self.device)
                    _, logits = self.model(tensor)
                    probability = torch.sigmoid(logits)[0, :valid].float().cpu().numpy()
                    window[:, y0:y1, x0:x1] = probability[
                        :, y0 - iy0:y1 - iy0, x0 - ix0:x1 - ix0
                    ]
                merged[start:end] = np.maximum(merged[start:end], window)
                coverage[start:end] += 1
        if count and np.any(coverage == 0):
            raise RuntimeError("DeepPro temporal windows did not cover every frame")
        return merged


def _load_sequence_images(sequence: Path, max_frames: int | None = None):
    images = _files(sequence / "img")
    stems = sorted(images, key=lambda value: natural_key(Path(value)))
    if max_frames:
        stems = stems[:max_frames]
    arrays = []
    for stem in stems:
        image = np.asarray(Image.open(images[stem]).convert("L"), dtype=np.uint8)
        arrays.append(image)
    if arrays and any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError(f"mixed frame shapes within sequence {sequence.name}")
    return stems, np.stack(arrays) if arrays else np.empty((0, 0, 0), dtype=np.uint8)


def _summary(totals: dict[str, float], threshold: float) -> dict:
    tp, fp, fn = (int(totals[key]) for key in ("tp", "fp", "fn"))
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def evaluate_deeppro(
    source_root: str | Path,
    weights: str | Path,
    val_root: str | Path,
    output_dir: str | Path,
    device: str = "0",
    thresholds: list[float] | None = None,
    radius: float = 2.0,
    max_sequences: int | None = None,
    max_frames: int | None = None,
    tile_size: int = 256,
    tile_halo: int = 12,
    min_area: int = 1,
    max_area: int | None = None,
) -> dict:
    thresholds = sorted(set(thresholds or [1e-5, 1e-4, 1e-3, 1e-2, .1, .5]))
    if not thresholds or any(value <= 0 or value >= 1 for value in thresholds):
        raise ValueError("thresholds must be between 0 and 1")
    detector = DeepProDetector(
        source_root, weights, device=device, tile_size=tile_size, tile_halo=tile_halo
    )
    root = Path(val_root)
    sequences = _sequences(root)
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
                    probabilities[index], threshold, min_area=min_area, max_area=max_area
                )
                metrics = point_metrics(predicted, truth, radius)
                bucket = by_resolution[threshold].setdefault(
                    resolution, defaultdict(float)
                )
                for key in ("tp", "fp", "fn"):
                    totals[threshold][key] += metrics[key]
                    bucket[key] += metrics[key]
            frame_count += 1
        partial_sweep = [_summary(totals[value], value) for value in thresholds]
        progress = {
            "status": "running",
            "completed_sequences": sequence_index,
            "total_sequences": len(sequences),
            "last_sequence": sequence.name,
            "frames": frame_count,
            "sweep": partial_sweep,
        }
        (output / "progress.json").write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )
        print(
            f"deeppro-eval {sequence_index}/{len(sequences)} "
            f"sequence={sequence.name} frames={frame_count}",
            flush=True,
        )

    sweep = [_summary(totals[value], value) for value in thresholds]
    best = max(sweep, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    resolution_names = sorted(
        {name for threshold_rows in by_resolution.values() for name in threshold_rows}
    )
    resolution_sweeps = {
        name: [
            _summary(
                by_resolution[threshold].get(name, defaultdict(float)), threshold
            )
            for threshold in thresholds
        ]
        for name in resolution_names
    }
    best_threshold_by_resolution = {
        name: max(
            rows, key=lambda row: (row["f1"], row["recall"], row["precision"])
        )
        for name, rows in resolution_sweeps.items()
    }
    best_resolution = {
        name: _summary(values, best["threshold"])
        for name, values in sorted(by_resolution[best["threshold"]].items())
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
        "best_resolution_breakdown": best_resolution,
        "resolution_sweeps": resolution_sweeps,
        "best_threshold_by_resolution": best_threshold_by_resolution,
        "radius_pixels": radius,
        "tile_size": tile_size,
        "tile_halo": tile_halo,
        "component_min_area": min_area,
        "component_max_area": max_area,
        "metric_status": "local point-matching proxy; not official scorer",
    }
    (output / "val_proxy_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def infer_deeppro(
    source_root: str | Path,
    weights: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    threshold: float,
    device: str = "0",
    coordinate_order: str = "xy",
    max_sequences: int | None = None,
    max_frames: int | None = None,
    tile_size: int = 256,
    tile_halo: int = 12,
    min_area: int = 1,
    max_area: int | None = None,
    make_zip: bool = True,
    threshold_by_resolution: dict[str, float] | None = None,
) -> dict:
    detector = DeepProDetector(
        source_root, weights, device=device, tile_size=tile_size, tile_halo=tile_halo
    )
    root, output = Path(data_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sequences = _sequences(root)
    if max_sequences:
        sequences = sequences[:max_sequences]
    detections = 0
    frame_count = 0
    for sequence in sequences:
        stems, frames = _load_sequence_images(sequence, max_frames)
        probabilities = detector.predict(frames)
        prediction_frames = {}
        height, width = frames.shape[1:] if len(frames) else (0, 0)
        resolution = f"{width}x{height}"
        sequence_threshold = (
            threshold_by_resolution.get(resolution, threshold)
            if threshold_by_resolution
            else threshold
        )
        for index, _ in enumerate(stems, 1):
            points = component_points(
                probabilities[index - 1],
                sequence_threshold,
                min_area=min_area,
                max_area=max_area,
            )
            prediction_frames[index] = [
                TrackPoint(index, 0, x, y) for x, y in points
            ]
            detections += len(points)
            frame_count += 1
        write_txt(
            SequencePrediction(sequence.name, prediction_frames),
            output / f"{sequence.name}.txt",
            coordinate_order,
            overwrite=True,
        )
    zip_path = (
        package(
            output,
            output.with_suffix(".zip"),
            expected=len(sequences),
            overwrite=True,
            coordinate_order=coordinate_order,
        )
        if make_zip
        else None
    )
    return {
        "model": detector.model_name,
        "sequences": len(sequences),
        "frames": frame_count,
        "detections": detections,
        "threshold": threshold,
        "threshold_by_resolution": threshold_by_resolution,
        "coordinate_order": coordinate_order,
        "output_dir": str(output),
        "zip": str(zip_path) if zip_path else None,
    }
