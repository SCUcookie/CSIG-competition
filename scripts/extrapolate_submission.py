"""Add image-gated one-frame extrapolations from stable endpoint tracks."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_track_extrapolation import evidence
from jinsight_track1.data import natural_key
from jinsight_track1.deeppro_adapter import _files, _load_sequence_images, _sequences
from jinsight_track1.submission import package, parse, write_txt
from jinsight_track1.tracking import assign_track_ids
from jinsight_track1.types import SequencePrediction, TrackPoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--resolutions", default="256x256")
    parser.add_argument("--min-hits", type=int, default=5)
    parser.add_argument("--max-residual", type=float, default=1.0)
    parser.add_argument("--score", choices=["spatial", "temporal", "minimum", "maximum"], default="maximum")
    parser.add_argument("--score-threshold", type=float, default=1.0)
    parser.add_argument("--fit-window", type=int, default=5)
    parser.add_argument("--gate-fraction", type=float, default=.02)
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--suppression-radius", type=float, default=2.0)
    parser.add_argument("--coordinate-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--no-package", action="store_true")
    args = parser.parse_args()
    selected_resolutions = set(args.resolutions.split(","))
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    added = 0

    sequences = _sequences(Path(args.data_root))
    for sequence in sequences:
        image_files = _files(sequence / "img")
        stems = sorted(image_files, key=lambda value: natural_key(Path(value)))
        width, height = Image.open(image_files[stems[0]]).size
        resolution = f"{width}x{height}"
        images = (
            _load_sequence_images(sequence)[1]
            if resolution in selected_resolutions
            else None
        )
        prediction = parse(
            (source / f"{sequence.name}.txt").read_text(encoding="ascii"),
            sequence.name,
            args.coordinate_order,
        )
        frame_ids = range(1, len(stems) + 1)
        original = {
            frame_id: [
                (point.x, point.y)
                for point in prediction.frames.get(frame_id, [])
            ]
            for frame_id in frame_ids
        }
        candidates = defaultdict(list)
        if resolution in selected_resolutions:
            tracked = assign_track_ids(
                [original[frame_id] for frame_id in frame_ids],
                (height, width),
                gate_fraction=args.gate_fraction,
                max_age=args.max_age,
            )
            histories = defaultdict(list)
            for points in tracked.values():
                for point in points:
                    histories[point.track_id].append(point)
            for history in histories.values():
                history.sort(key=lambda point: point.frame_id)
                if len(history) < args.min_hits:
                    continue
                for side in ("start", "end"):
                    sample = (
                        history[: args.fit_window]
                        if side == "start"
                        else history[-args.fit_window :]
                    )
                    if len(sample) < args.fit_window:
                        continue
                    times = np.asarray([point.frame_id for point in sample], float)
                    if not np.all(np.diff(times) == 1):
                        continue
                    design = np.column_stack((times, np.ones_like(times)))
                    xs = np.asarray([point.x for point in sample])
                    ys = np.asarray([point.y for point in sample])
                    x_fit = np.linalg.lstsq(design, xs, rcond=None)[0]
                    y_fit = np.linalg.lstsq(design, ys, rcond=None)[0]
                    fitted_x, fitted_y = design @ x_fit, design @ y_fit
                    residual = float(
                        np.sqrt(
                            np.mean(
                                (xs - fitted_x) ** 2 + (ys - fitted_y) ** 2
                            )
                        )
                    )
                    if residual > args.max_residual:
                        continue
                    frame_id = int(
                        times[0] - 1 if side == "start" else times[-1] + 1
                    )
                    if not (1 <= frame_id <= len(stems)):
                        continue
                    x = float(x_fit[0] * frame_id + x_fit[1])
                    y = float(y_fit[0] * frame_id + y_fit[1])
                    if not (0 <= x < width and 0 <= y < height):
                        continue
                    if any(
                        (x - old_x) ** 2 + (y - old_y) ** 2
                        <= args.suppression_radius**2
                        for old_x, old_y in original[frame_id]
                    ):
                        continue
                    spatial, temporal = evidence(
                        images, frame_id - 1, x, y
                    )
                    if args.score == "minimum":
                        score = min(spatial, temporal)
                    elif args.score == "maximum":
                        score = max(spatial, temporal)
                    else:
                        score = spatial if args.score == "spatial" else temporal
                    if score >= args.score_threshold:
                        candidates[frame_id].append((x, y))

        frames = {}
        for frame_id in frame_ids:
            selected = []
            for point in candidates[frame_id]:
                if all(
                    (point[0] - old[0]) ** 2 + (point[1] - old[1]) ** 2
                    > args.suppression_radius**2
                    for old in selected
                ):
                    selected.append(point)
            added += len(selected)
            frames[frame_id] = [
                TrackPoint(frame_id, point.track_id, point.x, point.y)
                for point in prediction.frames.get(frame_id, [])
            ] + [
                TrackPoint(frame_id, 0, float(x), float(y))
                for x, y in selected
            ]
        write_txt(
            SequencePrediction(sequence.name, frames),
            output / f"{sequence.name}.txt",
            args.coordinate_order,
            overwrite=True,
        )
        print(
            f"track-extrapolate sequence={sequence.name} added={added}",
            flush=True,
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
    report = {
        "sequences": len(sequences),
        "added_points": added,
        "resolutions": sorted(selected_resolutions),
        "min_hits": args.min_hits,
        "max_residual": args.max_residual,
        "score": args.score,
        "score_threshold": args.score_threshold,
        "fit_window": args.fit_window,
        "gate_fraction": args.gate_fraction,
        "max_age": args.max_age,
        "suppression_radius": args.suppression_radius,
        "output": str(output),
        "zip": str(zip_path) if zip_path else None,
    }
    (output.parent / f"{output.name}_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
