from dataclasses import dataclass, field
from typing import Protocol, Sequence
import numpy as np

@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    score: float

@dataclass(frozen=True)
class TrackPoint:
    frame_id: int
    track_id: int
    x: float
    y: float

@dataclass
class SequencePrediction:
    sequence_name: str
    frames: dict[int, list[TrackPoint]] = field(default_factory=dict)

class Detector(Protocol):
    def predict(self, frames: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Return one 2-D float mask per input frame, in [0, 1]."""
