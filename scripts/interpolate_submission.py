"""Fill short detection gaps by interpolating stable sequence-local tracks."""
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
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--min-track-hits", type=int, default=3)
    parser.add_argument("--gate-fraction", type=float, default=0.02)
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--suppression-radius", type=float, default=2.0)
    parser.add_argument("--resolutions")
    parser.add_argument("--coordinate-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--no-package", action="store_true")
    args = parser.parse_args()
    if args.max_gap < 1 or args.min_track_hits < 2:
        raise ValueError("max-gap must be positive and min-track-hits at least 2")

    selected_resolutions = (
        set(args.resolutions.split(",")) if args.resolutions else None
    )
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    added = 0
    processed = 0
    sequences = _sequences(Path(args.data_root))
    for sequence in sequences:
        images = _files(sequence / "img")
        first_image = Image.open(next(iter(images.values())))
        width, height = first_image.size
        resolution = f"{width}x{height}"
        prediction = parse(
            (source / f"{sequence.name}.txt").read_text(encoding="ascii"),
            sequence.name,
            args.coordinate_order,
        )
        frame_ids = range(1, len(images) + 1)
        points = {
            frame_id: [
                (point.x, point.y)
                for point in prediction.frames.get(frame_id, [])
            ]
            for frame_id in frame_ids
        }
        if selected_resolutions is None or resolution in selected_resolutions:
            tracked = assign_track_ids(
                [points[frame_id] for frame_id in frame_ids],
                (height, width),
                gate_fraction=args.gate_fraction,
                max_age=args.max_age,
            )
            histories: dict[int, list[TrackPoint]] = defaultdict(list)
            for tracked_points in tracked.values():
                for point in tracked_points:
                    histories[point.track_id].append(point)
            for history in histories.values():
                if len(history) < args.min_track_hits:
                    continue
                for left, right in zip(history, history[1:]):
                    missing = right.frame_id - left.frame_id - 1
                    if missing < 1 or missing > args.max_gap:
                        continue
                    for offset in range(1, missing + 1):
                        fraction = offset / (missing + 1)
                        x = left.x + fraction * (right.x - left.x)
                        y = left.y + fraction * (right.y - left.y)
                        frame_id = left.frame_id + offset
                        radius2 = args.suppression_radius**2
                        if any(
                            (x - old_x) ** 2 + (y - old_y) ** 2 <= radius2
                            for old_x, old_y in points[frame_id]
                        ):
                            continue
                        points[frame_id].append((x, y))
                        added += 1
            processed += 1
        frames = {
            frame_id: [
                TrackPoint(frame_id, 0, float(x), float(y))
                for x, y in frame_points
            ]
            for frame_id, frame_points in points.items()
        }
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
                "processed_sequences": processed,
                "added_points": added,
                "max_gap": args.max_gap,
                "min_track_hits": args.min_track_hits,
                "gate_fraction": args.gate_fraction,
                "max_age": args.max_age,
                "resolutions": sorted(selected_resolutions)
                if selected_resolutions
                else None,
                "output": str(output),
                "zip": str(zip_path) if zip_path else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
