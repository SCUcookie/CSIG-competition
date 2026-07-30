"""Evaluate an LVNet checkpoint on challenge-format validation data."""
from __future__ import annotations

import argparse
import json

from jinsight_track1.lvnet_adapter import evaluate_lvnet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("weights")
    parser.add_argument("val_root")
    parser.add_argument("output_dir")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--thresholds",
        default="0.0001,0.001,0.01,0.03,0.1,0.3,0.5,0.7,0.9",
    )
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--resolutions")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--temporal-stride", type=int, default=4)
    parser.add_argument("--mean", type=float, default=111.47)
    parser.add_argument("--std", type=float, default=22.43)
    parser.add_argument("--adaptive-normalization", action="store_true")
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int)
    parser.add_argument(
        "--centroid-mode", choices=["binary", "weighted", "peak"], default="binary"
    )
    args = parser.parse_args()
    report = evaluate_lvnet(
        args.source_root,
        args.weights,
        args.val_root,
        args.output_dir,
        device=args.device,
        thresholds=[float(value) for value in args.thresholds.split(",")],
        radius=args.radius,
        max_sequences=args.max_sequences,
        max_frames=args.max_frames,
        resolutions=set(args.resolutions.split(",")) if args.resolutions else None,
        num_frames=args.num_frames,
        temporal_stride=args.temporal_stride,
        mean=args.mean,
        std=args.std,
        adaptive_normalization=args.adaptive_normalization,
        min_area=args.min_area,
        max_area=args.max_area,
        centroid_mode=args.centroid_mode,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
