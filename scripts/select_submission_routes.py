"""Select the best submission candidate independently for every sequence."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("report")
    parser.add_argument(
        "candidates",
        nargs="+",
        help="LABEL=METRICS_JSON=SUBMISSION_DIR",
    )
    parser.add_argument("--fixed-tp", type=int, default=0)
    parser.add_argument("--fixed-fp", type=int, default=0)
    parser.add_argument("--fixed-fn", type=int, default=0)
    args = parser.parse_args()

    candidates: dict[str, tuple[Path, dict[str, dict[str, float | int]]]] = {}
    for value in args.candidates:
        label, metrics_path, submission_dir = value.split("=", 2)
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        candidates[label] = (
            Path(submission_dir),
            metrics["sequence_breakdown"],
        )

    sequence_sets = {
        label: set(metrics) for label, (_, metrics) in candidates.items()
    }
    expected = next(iter(sequence_sets.values()))
    if any(sequences != expected for sequences in sequence_sets.values()):
        raise ValueError(f"candidate sequence sets differ: {sequence_sets}")

    ratio = 0.8
    selected: dict[str, tuple[str, dict[str, float | int]]] = {}
    for _ in range(100):
        selected = {}
        for sequence in sorted(expected):
            options = [
                (label, metrics[sequence])
                for label, (_, metrics) in candidates.items()
            ]
            selected[sequence] = max(
                options,
                key=lambda option: (
                    2 * option[1]["tp"]
                    - ratio
                    * (
                        2 * option[1]["tp"]
                        + option[1]["fp"]
                        + option[1]["fn"]
                    )
                ),
            )
        tp = args.fixed_tp + sum(int(row["tp"]) for _, row in selected.values())
        fp = args.fixed_fp + sum(int(row["fp"]) for _, row in selected.values())
        fn = args.fixed_fn + sum(int(row["fn"]) for _, row in selected.values())
        updated = 2 * tp / max(1, 2 * tp + fp + fn)
        if abs(updated - ratio) < 1e-12:
            ratio = updated
            break
        ratio = updated

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(output_dir.glob("*.txt"))
    if stale:
        raise FileExistsError(f"output contains {len(stale)} text files")
    routes = {}
    for sequence, (label, metrics) in selected.items():
        source_dir = candidates[label][0]
        shutil.copy2(source_dir / f"{sequence}.txt", output_dir / f"{sequence}.txt")
        routes[sequence] = {"candidate": label, **metrics}

    report = {
        "fixed": {
            "tp": args.fixed_tp,
            "fp": args.fixed_fp,
            "fn": args.fixed_fn,
        },
        "total": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "f1": ratio,
        },
        "selected_counts": dict(
            sorted(Counter(label for label, _ in selected.values()).items())
        ),
        "routes": routes,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total", "selected_counts")}, indent=2))


if __name__ == "__main__":
    main()
