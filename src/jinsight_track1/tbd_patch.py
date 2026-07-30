"""Small balanced patch classifier for track-before-detect candidate scoring."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


class TBDPatchClassifier(nn.Module):
    def __init__(self, channels: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(48, 1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(patches).flatten(1)).squeeze(1)


def spatial_dark_score(frame: np.ndarray) -> np.ndarray:
    image = frame.astype(np.float32)
    centre = cv2.GaussianBlur(image, (0, 0), 0.65)
    surround = cv2.GaussianBlur(image, (0, 0), 2.4)
    raw = surround - centre
    sample = raw[::4, ::4]
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    return (raw - median) / max(0.25, 1.4826 * mad)


def spatial_score(frame: np.ndarray, mode: str = "dark") -> np.ndarray:
    """Return a robust small-target response, optionally polarity invariant."""
    score = spatial_dark_score(frame)
    if mode == "dark":
        return score
    if mode == "absolute":
        return np.abs(score)
    raise ValueError(f"unsupported spatial score mode: {mode}")


def candidate_peaks(
    score: np.ndarray,
    count: int,
    *,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    maximum = cv2.dilate(score, np.ones((3, 3), dtype=np.uint8))
    ys, xs = np.where(score >= maximum - 1e-6)
    if roi is not None:
        x0, y0, x1, y1 = roi
        keep = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        xs, ys = xs[keep], ys[keep]
    if not len(xs):
        return np.zeros((0, 2), dtype=int)
    values = score[ys, xs]
    if len(values) > count:
        selected = np.argpartition(values, -count)[-count:]
        selected = selected[np.argsort(values[selected])[::-1]]
        xs, ys = xs[selected], ys[selected]
    else:
        order = np.argsort(values)[::-1]
        xs, ys = xs[order], ys[order]
    return np.column_stack((xs, ys)).astype(int)


def extract_patch_channels(
    frame: np.ndarray,
    score: np.ndarray,
    points: np.ndarray,
    size: int = 21,
) -> np.ndarray:
    radius = size // 2
    value = frame.astype(np.float32)
    mean = cv2.boxFilter(value, cv2.CV_32F, (size, size), normalize=True)
    mean2 = cv2.boxFilter(value * value, cv2.CV_32F, (size, size), normalize=True)
    deviation = np.sqrt(np.maximum(mean2 - mean * mean, 1.0))
    normalized_map = np.clip((value - mean) / deviation, -8, 8)
    padded_frame = cv2.copyMakeBorder(
        normalized_map, radius, radius, radius, radius, cv2.BORDER_REFLECT
    )
    padded_score = cv2.copyMakeBorder(
        score, radius, radius, radius, radius, cv2.BORDER_REFLECT
    )
    points = np.asarray(points, dtype=int).reshape(-1, 2)
    if not len(points):
        return np.zeros((0, 2, size, size), dtype=np.float32)
    frame_windows = np.lib.stride_tricks.sliding_window_view(
        padded_frame, (size, size)
    )
    score_windows = np.lib.stride_tricks.sliding_window_view(
        padded_score, (size, size)
    )
    xs, ys = points[:, 0], points[:, 1]
    return np.stack(
        (
            frame_windows[ys, xs] / 4.0,
            np.clip(score_windows[ys, xs], -8, 8) / 4.0,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
