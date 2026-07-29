"""Snap submission coordinates to a configurable spatial grid."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from jinsight_track1.deeppro_adapter import _sequences
from jinsight_track1.submission import package, parse, write_txt
from jinsight_track1.types import SequencePrediction, TrackPoint


def quantize(value: float, step: float, mode: str) -> float:
    scaled = value / step
    if mode == "nearest":
        snapped = math.floor(scaled + 0.5)
    elif mode == "floor":
        snapped = math.floor(scaled)
    else:
        snapped = math.ceil(scaled)
    return snapped * step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--mode", choices=["nearest", "floor", "ceil"], default="nearest")
    parser.add_argument("--coordinate-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--no-package", action="store_true")
    args = parser.parse_args()
    if args.step <= 0:
        raise ValueError("step must be positive")

    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    changed = 0
    sequences = _sequences(Path(args.data_root))
    for sequence in sequences:
        prediction = parse(
            (source / f"{sequence.name}.txt").read_text(encoding="ascii"),
            sequence.name,
            args.coordinate_order,
        )
        frames = {}
        for frame_id, points in prediction.frames.items():
            frames[frame_id] = []
            for point in points:
                x = quantize(point.x, args.step, args.mode)
                y = quantize(point.y, args.step, args.mode)
                changed += int(x != point.x or y != point.y)
                frames[frame_id].append(
                    TrackPoint(frame_id, point.track_id, x, y)
                )
        write_txt(
            SequencePrediction(sequence.name, frames),
            output / f"{sequence.name}.txt",
            args.coordinate_order,
            overwrite=True,
        )

    zip_path = (
        None
        if args.no_package
        else package(
            output,
            output.with_suffix(".zip"),
            expected=len(sequences),
            overwrite=True,
            coordinate_order=args.coordinate_order,
        )
    )
    print(
        json.dumps(
            {
                "sequences": len(sequences),
                "changed_points": changed,
                "step": args.step,
                "mode": args.mode,
                "output": str(output),
                "zip": str(zip_path) if zip_path else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
