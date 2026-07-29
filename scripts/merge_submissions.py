"""Fuse detection submissions and spatially deduplicate matching points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jinsight_track1.deeppro_adapter import _files, _sequences
from jinsight_track1.submission import package, parse, write_txt
from jinsight_track1.types import SequencePrediction, TrackPoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("output")
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--dedup-radius", type=float, default=2.0)
    parser.add_argument("--coordinate-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--no-package", action="store_true")
    args = parser.parse_args()
    if args.dedup_radius < 0:
        raise ValueError("dedup-radius must be non-negative")

    sequences = _sequences(Path(args.data_root))
    sources = [Path(value) for value in args.sources]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    input_points = 0
    output_points = 0
    for sequence in sequences:
        frame_count = len(_files(sequence / "img"))
        predictions = [
            parse(
                (source / f"{sequence.name}.txt").read_text(encoding="ascii"),
                sequence.name,
                args.coordinate_order,
            )
            for source in sources
        ]
        frames = {}
        for frame_id in range(1, frame_count + 1):
            clusters: list[dict] = []
            for source_index, prediction in enumerate(predictions):
                for point in prediction.frames.get(frame_id, []):
                    input_points += 1
                    position = np.array([point.x, point.y], dtype=float)
                    eligible = [
                        index
                        for index, cluster in enumerate(clusters)
                        if source_index not in cluster["sources"]
                    ]
                    if eligible:
                        distances = [
                            float(np.linalg.norm(position - clusters[index]["mean"]))
                            for index in eligible
                        ]
                        nearest_offset = int(np.argmin(distances))
                        nearest = eligible[nearest_offset]
                    else:
                        distances, nearest = [], -1
                    if distances and distances[nearest_offset] <= args.dedup_radius:
                        cluster = clusters[nearest]
                        cluster["sum"] += position
                        cluster["count"] += 1
                        cluster["mean"] = cluster["sum"] / cluster["count"]
                        cluster["sources"].add(source_index)
                    else:
                        clusters.append(
                            {
                                "sum": position.copy(),
                                "count": 1,
                                "mean": position.copy(),
                                "sources": {source_index},
                            }
                        )
            frames[frame_id] = [
                TrackPoint(
                    frame_id,
                    0,
                    float(cluster["mean"][0]),
                    float(cluster["mean"][1]),
                )
                for cluster in clusters
            ]
            output_points += len(frames[frame_id])
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
                "sources": args.sources,
                "dedup_radius": args.dedup_radius,
                "input_points": input_points,
                "output_points": output_points,
                "output": str(output),
                "zip": str(zip_path) if zip_path else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
