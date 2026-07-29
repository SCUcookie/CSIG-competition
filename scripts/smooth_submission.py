"""Smooth detection coordinates along short sequence-local motion tracks."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from jinsight_track1.deeppro_adapter import _files, _sequences
from jinsight_track1.submission import package, parse, write_txt
from jinsight_track1.tracking import assign_track_ids
from jinsight_track1.types import SequencePrediction, TrackPoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-track-hits", type=int, default=3)
    parser.add_argument("--gate-fraction", type=float, default=0.02)
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--coordinate-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--no-package", action="store_true")
    args = parser.parse_args()
    if args.window < 3 or args.window % 2 == 0:
        raise ValueError("window must be an odd integer of at least 3")

    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    smoothed = 0
    sequences = _sequences(Path(args.data_root))
    for sequence in sequences:
        images = _files(sequence / "img")
        width, height = Image.open(next(iter(images.values()))).size
        prediction = parse(
            (source / f"{sequence.name}.txt").read_text(encoding="ascii"),
            sequence.name,
            args.coordinate_order,
        )
        frame_ids = range(1, len(images) + 1)
        original = {
            frame_id: [
                (point.x, point.y)
                for point in prediction.frames.get(frame_id, [])
            ]
            for frame_id in frame_ids
        }
        tracked = assign_track_ids(
            [original[frame_id] for frame_id in frame_ids],
            (height, width),
            gate_fraction=args.gate_fraction,
            max_age=args.max_age,
        )
        histories: dict[int, list[TrackPoint]] = defaultdict(list)
        for points in tracked.values():
            for point in points:
                histories[point.track_id].append(point)
        replacement = {}
        half = args.window // 2
        for history in histories.values():
            if len(history) < args.min_track_hits:
                continue
            for point in history:
                neighbors = [
                    other
                    for other in history
                    if abs(other.frame_id - point.frame_id) <= half
                ]
                if len(neighbors) < args.min_track_hits:
                    continue
                times = np.asarray([other.frame_id for other in neighbors], float)
                design = np.column_stack((times, np.ones_like(times)))
                xs = np.asarray([other.x for other in neighbors], float)
                ys = np.asarray([other.y for other in neighbors], float)
                x_slope, x_offset = np.linalg.lstsq(design, xs, rcond=None)[0]
                y_slope, y_offset = np.linalg.lstsq(design, ys, rcond=None)[0]
                replacement[(point.frame_id, point.track_id)] = (
                    x_slope * point.frame_id + x_offset,
                    y_slope * point.frame_id + y_offset,
                )
                smoothed += 1
        frames = {}
        for frame_id, points in tracked.items():
            frames[frame_id] = []
            for point in points:
                x, y = replacement.get(
                    (frame_id, point.track_id), (point.x, point.y)
                )
                frames[frame_id].append(
                    TrackPoint(frame_id, 0, float(x), float(y))
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
                "smoothed_points": smoothed,
                "window": args.window,
                "min_track_hits": args.min_track_hits,
                "gate_fraction": args.gate_fraction,
                "output": str(output),
                "zip": str(zip_path) if zip_path else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
