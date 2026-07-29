"""Training-free local-contrast inference for out-of-domain resolutions."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .deeppro_adapter import _load_sequence_images, _sequences
from .submission import package, write_txt
from .tracking import assign_track_ids
from .types import SequencePrediction, TrackPoint


def contrast_responses(
    image: np.ndarray,
    temporal_background: np.ndarray | None = None,
    temporal_deviation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    value = image.astype(np.float32)
    dog = cv2.GaussianBlur(value, (0, 0), 0.8) - cv2.GaussianBlur(
        value, (0, 0), 2.4
    )
    mean = cv2.boxFilter(value, cv2.CV_32F, (11, 11), normalize=True)
    mean2 = cv2.boxFilter(value * value, cv2.CV_32F, (11, 11), normalize=True)
    deviation = np.sqrt(np.maximum(mean2 - mean * mean, 4.0))
    local_z = (value - mean) / deviation
    found = {
        "dog_bright": np.maximum(dog, 0),
        "dog_dark": np.maximum(-dog, 0),
        "z_bright": np.maximum(local_z, 0),
        "z_dark": np.maximum(-local_z, 0),
    }
    for size in (5, 9, 15):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        found[f"tophat_{size}"] = cv2.morphologyEx(
            image, cv2.MORPH_TOPHAT, kernel
        ).astype(np.float32)
        found[f"blackhat_{size}"] = cv2.morphologyEx(
            image, cv2.MORPH_BLACKHAT, kernel
        ).astype(np.float32)
    if temporal_background is not None and temporal_deviation is not None:
        delta = value - temporal_background
        found["temporal_bright"] = np.maximum(delta, 0)
        found["temporal_dark"] = np.maximum(-delta, 0)
        found["temporal_z_bright"] = np.maximum(delta / temporal_deviation, 0)
        found["temporal_z_dark"] = np.maximum(-delta / temporal_deviation, 0)
    return found


def topk_points(
    response: np.ndarray, count: int, nms_radius: float = 3.0
) -> list[tuple[float, float]]:
    dilated = cv2.dilate(response, np.ones((3, 3), np.uint8))
    ys, xs = np.where((response >= dilated) & (response > 0))
    if not len(xs):
        return []
    scores = response[ys, xs]
    candidate_limit = min(len(scores), max(200, count * 50))
    if len(scores) > candidate_limit:
        keep = np.argpartition(scores, -candidate_limit)[-candidate_limit:]
        xs, ys, scores = xs[keep], ys[keep], scores[keep]
    order = np.argsort(scores)[::-1]
    selected: list[tuple[float, float]] = []
    radius2 = nms_radius * nms_radius
    for index in order:
        point = (float(xs[index]), float(ys[index]))
        if all(
            (point[0] - old[0]) ** 2 + (point[1] - old[1]) ** 2 > radius2
            for old in selected
        ):
            selected.append(point)
            if len(selected) == count:
                break
    return selected


def infer_local_contrast(
    data_root: str | Path,
    output_dir: str | Path,
    config: dict[str, dict],
    coordinate_order: str = "xy",
    track: bool = False,
    make_zip: bool = True,
) -> dict:
    root, output = Path(data_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sequences = _sequences(root)
    detections = 0
    frames_seen = 0
    used_resolutions = set()
    for sequence in sequences:
        stems, frames = _load_sequence_images(sequence)
        height, width = frames.shape[1:] if len(frames) else (0, 0)
        resolution = f"{width}x{height}"
        setting = config.get(resolution)
        point_frames: list[list[tuple[float, float]]] = []
        if setting:
            method = str(setting["method"])
            top_k = int(setting["top_k"])
            used_resolutions.add(resolution)
            if method.startswith("temporal_"):
                stride = max(1, len(frames) // 80)
                sample = frames[::stride].astype(np.float32)
                background = np.median(sample, axis=0).astype(np.float32)
                temporal_deviation = np.maximum(sample.std(axis=0), 2.0)
            else:
                background = temporal_deviation = None
            for frame in frames:
                response = contrast_responses(
                    frame, background, temporal_deviation
                )[method]
                point_frames.append(topk_points(response, top_k))
        else:
            point_frames = [[] for _ in stems]
        detections += sum(map(len, point_frames))
        frames_seen += len(stems)
        if track:
            prediction_frames = assign_track_ids(point_frames, (height, width))
        else:
            prediction_frames = {
                index: [TrackPoint(index, 0, x, y) for x, y in points]
                for index, points in enumerate(point_frames, 1)
            }
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
        "sequences": len(sequences),
        "frames": frames_seen,
        "detections": detections,
        "configured_resolutions": sorted(config),
        "used_resolutions": sorted(used_resolutions),
        "tracking": track,
        "coordinate_order": coordinate_order,
        "output_dir": str(output),
        "zip": str(zip_path) if zip_path else None,
    }
