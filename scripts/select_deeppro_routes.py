"""Select sequence-level DeepPro checkpoints and thresholds for micro F1."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("output")
    parser.add_argument("reports", nargs="+", help="LABEL=REPORT_JSON")
    parser.add_argument("--fixed-tp", type=int, default=0)
    parser.add_argument("--fixed-fp", type=int, default=0)
    parser.add_argument("--fixed-fn", type=int, default=0)
    parser.add_argument("--group-root")
    args = parser.parse_args()

    options = defaultdict(list)
    metadata = {}
    for value in args.reports:
        label, report_path = value.split("=", 1)
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        metadata[label] = {
            "weights": report["weights"],
            "centroid_mode": report.get("centroid_mode", "binary"),
            "report": report_path,
        }
        for sequence, rows in report["sequence_sweeps"].items():
            for row in rows:
                options[sequence].append((label, row))

    ratio = 0.8
    selected = {}
    for _ in range(100):
        selected = {
            sequence: max(
                rows,
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
            for sequence, rows in options.items()
        }
        tp = args.fixed_tp + sum(row["tp"] for _, row in selected.values())
        fp = args.fixed_fp + sum(row["fp"] for _, row in selected.values())
        fn = args.fixed_fn + sum(row["fn"] for _, row in selected.values())
        updated = 2 * tp / max(1, 2 * tp + fp + fn)
        if abs(updated - ratio) < 1e-12:
            ratio = updated
            break
        ratio = updated

    routes = {
        sequence: {
            "model": label,
            "threshold": row["threshold"],
            "tp": row["tp"],
            "fp": row["fp"],
            "fn": row["fn"],
            "f1": row["f1"],
        }
        for sequence, (label, row) in sorted(selected.items())
    }
    groups = {}
    group_root = Path(args.group_root).resolve() if args.group_root else None
    data_root = Path(args.data_root).resolve()
    for label in sorted(metadata):
        chosen = {
            sequence: route["threshold"]
            for sequence, route in routes.items()
            if route["model"] == label
        }
        if not chosen:
            continue
        group = {**metadata[label], "sequences": len(chosen)}
        if group_root:
            sequence_root = group_root / label
            sequence_root.mkdir(parents=True, exist_ok=True)
            for sequence in chosen:
                link = sequence_root / sequence
                target = (data_root / sequence).resolve()
                if link.exists() or link.is_symlink():
                    if link.resolve() != target:
                        raise ValueError(f"existing route link points elsewhere: {link}")
                else:
                    link.symlink_to(target, target_is_directory=True)
            threshold_path = group_root / f"{label}_thresholds.json"
            threshold_path.write_text(
                json.dumps(chosen, indent=2), encoding="utf-8"
            )
            group["data_root"] = str(sequence_root)
            group["threshold_map"] = str(threshold_path)
        groups[label] = group

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
            sorted(Counter(route["model"] for route in routes.values()).items())
        ),
        "groups": groups,
        "routes": routes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total", "selected_counts", "groups")}, indent=2))


if __name__ == "__main__":
    main()
