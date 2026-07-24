from dataclasses import dataclass
from pathlib import Path
from typing import Union
import re
from collections import Counter
import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]

@dataclass(frozen=True)
class Sequence:
    name: str
    frames: tuple[Path, ...]

def discover_sequences(root: Union[str, Path], max_sequences=None, max_frames=None) -> list:
    root = Path(root)
    if not root.is_dir(): raise FileNotFoundError(f"data directory not found: {root}")
    found = []
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        if d.name.lower() in {"mask", "masks", "label", "labels"}:
            continue
        fs = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=natural_key)
        if fs:
            name = d.parent.name if d.name.lower() in {"img", "images", "frames"} else d.name
            found.append(Sequence(name, tuple(fs[:max_frames] if max_frames else fs)))
    unique = {str(s.frames[0].parent): s for s in found}
    result = sorted(unique.values(), key=lambda s: natural_key(Path(s.name)))
    return result[:max_sequences] if max_sequences else result

def read_frame(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else Image.open(path))

def inspect(root, max_sequences=None, max_frames=None) -> dict:
    seqs = discover_sequences(root, max_sequences, max_frames)
    shapes, exts, counts, unreadable = Counter(), Counter(), [], []
    total_bytes = 0
    for s in seqs:
        counts.append(len(s.frames))
        for p in s.frames:
            exts[p.suffix.lower()] += 1; total_bytes += p.stat().st_size
            try: shapes[tuple(read_frame(p).shape)] += 1
            except Exception as e: unreadable.append(f"{p}: {e}")
    return {"root": str(root), "sequence_count": len(seqs), "frame_count": sum(counts),
            "extensions": dict(exts), "shapes": {str(k): v for k,v in shapes.items()},
            "frames_per_sequence": {"min": min(counts) if counts else 0, "max": max(counts) if counts else 0},
            "bytes": total_bytes, "unreadable": unreadable}
