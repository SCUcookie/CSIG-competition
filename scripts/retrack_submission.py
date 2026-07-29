"""Assign stable trajectory IDs to an existing detection-mode submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from jinsight_track1.deeppro_adapter import _files, _sequences
from jinsight_track1.submission import package, parse, write_txt
from jinsight_track1.tracking import assign_track_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--coordinate-order", choices=["xy", "yx"], default="xy")
    args = parser.parse_args()
    sequences = _sequences(Path(args.data_root))
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    points_seen = 0
    for sequence in sequences:
        prediction = parse(
            (source / f"{sequence.name}.txt").read_text(encoding="ascii"),
            sequence.name,
            args.coordinate_order,
        )
        frame_ids = sorted(prediction.frames)
        point_frames = [
            [(point.x, point.y) for point in prediction.frames[frame_id]]
            for frame_id in frame_ids
        ]
        images = _files(sequence / "img")
        first_image = next(iter(images.values()))
        height, width = np.asarray(Image.open(first_image)).shape[:2]
        tracked = assign_track_ids(point_frames, (height, width))
        write_txt(
            prediction.__class__(sequence.name, tracked),
            output / f"{sequence.name}.txt",
            args.coordinate_order,
            overwrite=True,
        )
        points_seen += sum(map(len, point_frames))
    zip_path = package(
        output,
        output.with_suffix(".zip"),
        expected=len(sequences),
        overwrite=True,
        coordinate_order=args.coordinate_order,
    )
    print(
        json.dumps(
            {
                "sequences": len(sequences),
                "points": points_seen,
                "output": str(output),
                "zip": str(zip_path),
                "zip_bytes": zip_path.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
