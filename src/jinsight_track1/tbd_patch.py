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
    padded_frame = cv2.copyMakeBorder(
        frame, radius, radius, radius, radius, cv2.BORDER_REFLECT
    )
    padded_score = cv2.copyMakeBorder(
        score, radius, radius, radius, radius, cv2.BORDER_REFLECT
    )
    patches = []
    for x, y in np.asarray(points, dtype=int).reshape(-1, 2):
        raw = padded_frame[y : y + size, x : x + size].astype(np.float32)
        local_score = padded_score[y : y + size, x : x + size].astype(np.float32)
        border = np.concatenate((raw[0], raw[-1], raw[1:-1, 0], raw[1:-1, -1]))
        centre = float(np.median(border))
        mad = float(np.median(np.abs(border - centre)))
        normalized = np.clip((raw - centre) / max(1.0, 1.4826 * mad), -8, 8)
        patches.append(
            np.stack((normalized / 4.0, np.clip(local_score, -8, 8) / 4.0))
        )
    if not patches:
        return np.zeros((0, 2, size, size), dtype=np.float32)
    return np.asarray(patches, dtype=np.float32)
