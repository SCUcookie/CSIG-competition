"""Link per-shard shift-and-stack candidates across contiguous video shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from jinsight_track1.deeppro_adapter import _files
from jinsight_track1.postprocess import centroids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("candidate_json", nargs="+")
    parser.add_argument("--gate", type=float, default=3.0)
    parser.add_argument("--velocity-gate", type=float, default=0.025)
    parser.add_argument("--top-chains", type=int, default=100)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--output")
    return parser.parse_args()


def load_candidate_file(path: Path):
    report = json.loads(path.read_text())
    sequence = report["sequences"][0]
    frames = int(sequence["frames"])
    candidates = []
    scores = np.asarray([row["score"] for row in sequence["candidates"]])
    median = float(np.median(scores))
    scale = max(1e-6, float(np.median(np.abs(scores - median))) * 1.4826)
    for row in sequence["candidates"]:
        midpoint = np.asarray(row["midpoint"], dtype=float)
        velocity = np.asarray(row["velocity"], dtype=float)
        reference = float(row["reference_time"])
        candidates.append(
            {
                "rank": int(row["rank"]),
                "midpoint": midpoint,
                "velocity": velocity,
                "reference": reference,
                "start": midpoint - reference * velocity,
                "end": midpoint + (frames - 1 - reference) * velocity,
                "score": float(row["score"]),
                "normalized_score": (float(row["score"]) - median) / scale,
            }
        )
    return sequence["sequence"], frames, candidates


def build_chains(shards, gate: float, velocity_gate: float, top_chains: int):
    first_candidates = shards[0][2]
    states = [
        {
            "score": candidate["normalized_score"],
            "indices": [index],
        }
        for index, candidate in enumerate(first_candidates)
    ]
    for shard_index in range(1, len(shards)):
        previous_candidates = shards[shard_index - 1][2]
        current_candidates = shards[shard_index][2]
        previous_end = np.asarray([candidate["end"] for candidate in previous_candidates])
        previous_velocity = np.asarray(
            [candidate["velocity"] for candidate in previous_candidates]
        )
        current_start = np.asarray([candidate["start"] for candidate in current_candidates])
        current_velocity = np.asarray(
            [candidate["velocity"] for candidate in current_candidates]
        )
        position_distance = np.linalg.norm(
            previous_end[:, None] - current_start[None, :], axis=2
        )
        velocity_distance = np.linalg.norm(
            previous_velocity[:, None] - current_velocity[None, :], axis=2
        )
        new_states = []
        for current_index, candidate in enumerate(current_candidates):
            linked = np.where(
                (position_distance[:, current_index] <= gate)
                & (velocity_distance[:, current_index] <= velocity_gate)
            )[0]
            if not len(linked):
                new_states.append(None)
                continue
            best_previous = max(
                linked,
                key=lambda index: states[index]["score"]
                if states[index] is not None
                else -np.inf,
            )
            previous_state = states[int(best_previous)]
            if previous_state is None:
                new_states.append(None)
                continue
            boundary_penalty = position_distance[best_previous, current_index] / gate
            velocity_penalty = (
                velocity_distance[best_previous, current_index] / velocity_gate
            )
            new_states.append(
                {
                    "score": previous_state["score"]
                    + candidate["normalized_score"]
                    - 0.25 * boundary_penalty
                    - 0.1 * velocity_penalty,
                    "indices": previous_state["indices"] + [current_index],
                }
            )
        states = new_states
    complete = [state for state in states if state is not None]
    complete.sort(key=lambda state: state["score"], reverse=True)
    return complete[:top_chains]


def truth_frames(root: Path, sequence: str):
    masks = _files(root / sequence / "mask")
    result = []
    for stem in sorted(masks):
        mask = np.asarray(Image.open(masks[stem]).convert("L"))
        result.append([(point.x, point.y) for point in centroids(mask, 0.5, 1)])
    return result


def point_counts(predicted, truth, radius):
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 2)
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    matched = 0
    if len(predicted) and len(truth):
        distances = np.linalg.norm(predicted[:, None] - truth[None, :], axis=2)
        rows, columns = linear_sum_assignment(distances)
        matched = int(
            sum(
                distances[row, column] <= radius
                for row, column in zip(rows, columns)
            )
        )
    return matched, len(predicted) - matched, len(truth) - matched


def evaluate(chains, shards, root: Path, radius: float):
    truths = {
        sequence: truth_frames(root, sequence) for sequence, _, _ in shards
    }
    rows = []
    for limit in range(1, len(chains) + 1):
        tp = fp = fn = 0
        for shard_index, (sequence, frames, candidates) in enumerate(shards):
            for time, truth in enumerate(truths[sequence]):
                predicted = []
                for chain in chains[:limit]:
                    candidate = candidates[chain["indices"][shard_index]]
                    predicted.append(
                        candidate["midpoint"]
                        + (time - candidate["reference"]) * candidate["velocity"]
                    )
                counts = point_counts(predicted, truth, radius)
                tp += counts[0]
                fp += counts[1]
                fn += counts[2]
        rows.append(
            {
                "chains": limit,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            }
        )
    return rows


def individual_oracle(shards, root: Path, radius: float):
    report = {}
    for sequence, frames, candidates in shards:
        truths = truth_frames(root, sequence)
        rows = []
        for candidate in candidates:
            distances = []
            hits = 0
            for time, truth in enumerate(truths):
                point = candidate["midpoint"] + (
                    time - candidate["reference"]
                ) * candidate["velocity"]
                if truth:
                    distance = float(
                        np.min(np.linalg.norm(np.asarray(truth) - point, axis=1))
                    )
                else:
                    distance = 999.0
                distances.append(distance)
                hits += distance <= radius
            rows.append(
                {
                    "rank": candidate["rank"],
                    "hits": int(hits),
                    "score": candidate["score"],
                    "midpoint": candidate["midpoint"].tolist(),
                    "velocity": candidate["velocity"].tolist(),
                    "start": candidate["start"].tolist(),
                    "end": candidate["end"].tolist(),
                    "distance_p50": float(np.quantile(distances, 0.5)),
                    "distance_p90": float(np.quantile(distances, 0.9)),
                }
            )
        rows.sort(key=lambda row: (-row["hits"], row["rank"]))
        report[sequence] = rows[:20]
    return report


def main():
    args = parse_args()
    shards = [load_candidate_file(Path(path)) for path in args.candidate_json]
    chains = build_chains(shards, args.gate, args.velocity_gate, args.top_chains)
    evaluation = evaluate(chains, shards, Path(args.root), args.radius)
    serializable = []
    for rank, chain in enumerate(chains, 1):
        serializable.append(
            {
                "rank": rank,
                "score": chain["score"],
                "candidate_ranks": [
                    shards[index][2][candidate_index]["rank"]
                    for index, candidate_index in enumerate(chain["indices"])
                ],
                "boundary_distances": [
                    float(
                        np.linalg.norm(
                            shards[index][2][chain["indices"][index]]["end"]
                            - shards[index + 1][2][chain["indices"][index + 1]][
                                "start"
                            ]
                        )
                    )
                    for index in range(len(shards) - 1)
                ],
            }
        )
    report = {
        "settings": vars(args),
        "shards": [shard[0] for shard in shards],
        "chains": serializable,
        "individual_oracle": individual_oracle(
            shards, Path(args.root), args.radius
        ),
        "evaluation": evaluation,
        "best": max(evaluation, key=lambda row: row["f1"]) if evaluation else None,
    }
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
