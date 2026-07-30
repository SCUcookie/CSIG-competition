#!/usr/bin/env python3
"""Prepare raw whole-frame YOLO detection data with expanded point boxes."""
from __future__ import annotations

import argparse
import json

from jinsight_track1.frame_dynamics_yolo import prepare_raw_point_box_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root")
    parser.add_argument("val_root")
    parser.add_argument("output_root")
    parser.add_argument("--box-size", type=float, default=12.0)
    args = parser.parse_args()
    stats = prepare_raw_point_box_dataset(
        args.train_root, args.val_root, args.output_root, args.box_size
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
