"""Remove detections that do not belong to sufficiently persistent tracks."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from jinsight_track1.deeppro_adapter import _files, _sequences
from jinsight_track1.submission import package, parse, write_txt
from jinsight_track1.tracking import assign_track_ids
from jinsight_track1.types import SequencePrediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--min-track-hits", type=int, default=3)
    parser.add_argument("--gate-fraction", type=float, default=0.02)
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--coordinate-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--no-package", action="store_true")
    args = parser.parse_args()
    if args.min_track_hits < 2:
        raise ValueError("min-track-hits must be at least 2")

    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    input_points = 0
    kept_points = 0
    sequences = _sequences(Path(args.data_root))
    for sequence in sequences:
        prediction = parse(
            (source / f"{sequence.name}.txt").read_text(encoding="ascii"),
            sequence.name,
            args.coordinate_order,
        )
        images = _files(sequence / "img")
        width, height = Image.open(next(iter(images.values()))).size
        frame_ids = range(1, len(images) + 1)
        point_frames = [
            [(point.x, point.y) for point in prediction.frames[frame_id]]
            for frame_id in frame_ids
        ]
        tracked = assign_track_ids(
            point_frames,
            (height, width),
            gate_fraction=args.gate_fraction,
            max_age=args.max_age,
        )
        hits = Counter(
            point.track_id
            for points in tracked.values()
            for point in points
        )
        frames = {
            frame_id: [
                point
                for point in tracked[frame_id]
                if hits[point.track_id] >= args.min_track_hits
            ]
            for frame_id in frame_ids
        }
        input_points += sum(map(len, point_frames))
        kept_points += sum(map(len, frames.values()))
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
                "input_points": input_points,
                "kept_points": kept_points,
                "removed_points": input_points - kept_points,
                "min_track_hits": args.min_track_hits,
                "gate_fraction": args.gate_fraction,
                "max_age": args.max_age,
                "output": str(output),
                "zip": str(zip_path) if zip_path else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
