"""Measure how many false negatives are recoverable from detected track spans."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from jinsight_track1.deeppro_adapter import _files, _sequences
from jinsight_track1.postprocess import centroids
from jinsight_track1.submission import parse
from jinsight_track1.tracking import assign_track_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("val_root")
    parser.add_argument("submission_dir")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--gate-fraction", type=float, default=.02)
    parser.add_argument("--max-age", type=int, default=5)
    parser.add_argument("--coordinate-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--resolutions")
    parser.add_argument("--output")
    args = parser.parse_args()

    gap_limits = (1, 2, 3, 5, 10, 20, 50)
    selected_resolutions = (
        set(args.resolutions.split(",")) if args.resolutions else None
    )
    buckets: dict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for sequence in _sequences(Path(args.val_root)):
        images = _files(sequence / "img")
        masks = _files(sequence / "mask")
        stems = sorted(images)
        width, height = Image.open(images[stems[0]]).size
        resolution = f"{width}x{height}"
        if selected_resolutions and resolution not in selected_resolutions:
            continue
        prediction = parse(
            (Path(args.submission_dir) / f"{sequence.name}.txt").read_text(
                encoding="ascii"
            ),
            sequence.name,
            args.coordinate_order,
        )
        truths = []
        predictions = []
        for frame_id, stem in enumerate(stems, 1):
            mask = np.asarray(Image.open(masks[stem]).convert("L"))
            truths.append([(p.x, p.y) for p in centroids(mask, .5, 1)])
            predictions.append(
                [(p.x, p.y) for p in prediction.frames.get(frame_id, [])]
            )
        tracked_truth = assign_track_ids(
            truths,
            (height, width),
            gate_fraction=args.gate_fraction,
            max_age=args.max_age,
        )
        track_frames: dict[int, list[tuple[int, bool]]] = defaultdict(list)
        for frame_id, truth_points in tracked_truth.items():
            predicted = np.asarray(predictions[frame_id - 1], dtype=float).reshape(-1, 2)
            truth_xy = np.asarray(
                [(point.x, point.y) for point in truth_points], dtype=float
            ).reshape(-1, 2)
            matched_truth = set()
            if len(predicted) and len(truth_xy):
                distances = np.linalg.norm(
                    truth_xy[:, None, :] - predicted[None, :, :], axis=2
                )
                rows, columns = linear_sum_assignment(distances)
                matched_truth = {
                    int(row)
                    for row, column in zip(rows, columns)
                    if distances[row, column] <= args.radius
                }
            for index, point in enumerate(truth_points):
                track_frames[point.track_id].append(
                    (frame_id, index in matched_truth)
                )

        for bucket_name in ("all", resolution):
            bucket = buckets[bucket_name]
            bucket["sequences"] += 1
            bucket["truth_points"] += sum(map(len, truths))
            bucket["predicted_points"] += sum(map(len, predictions))
            bucket["tracks"] += len(track_frames)
        for history in track_frames.values():
            history.sort()
            flags = [matched for _, matched in history]
            matched_count = sum(flags)
            for bucket_name in ("all", resolution):
                bucket = buckets[bucket_name]
                bucket["matched_truth"] += matched_count
                bucket["missed_truth"] += len(flags) - matched_count
                if not matched_count:
                    bucket["unseen_tracks"] += 1
                    bucket["unseen_truth"] += len(flags)
                    continue
                first = flags.index(True)
                last = len(flags) - 1 - flags[::-1].index(True)
                bucket["partly_seen_tracks"] += 1
                bucket["leading_misses"] += sum(not value for value in flags[:first])
                bucket["trailing_misses"] += sum(
                    not value for value in flags[last + 1 :]
                )
                index = first
                while index <= last:
                    if flags[index]:
                        index += 1
                        continue
                    end = index
                    while end <= last and not flags[end]:
                        end += 1
                    gap = end - index
                    bucket["internal_misses"] += gap
                    for limit in gap_limits:
                        if gap <= limit:
                            bucket[f"recoverable_gap_le_{limit}"] += gap
                    index = end

    report = {}
    for name, values in buckets.items():
        row = dict(values)
        truth = max(1, row["truth_points"])
        row["observed_recall"] = row["matched_truth"] / truth
        for limit in gap_limits:
            recovered = row.get(f"recoverable_gap_le_{limit}", 0)
            row[f"recoverable_gap_le_{limit}"] = recovered
            row[f"oracle_recall_gap_le_{limit}"] = (
                row["matched_truth"] + recovered
            ) / truth
        report[name] = row
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
