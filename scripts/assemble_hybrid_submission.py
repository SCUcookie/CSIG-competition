"""Assemble disjoint prediction directories into one validated submission ZIP."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from jinsight_track1.deeppro_adapter import _sequences
from jinsight_track1.submission import package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("output")
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--coordinate-order", choices=["xy", "yx"], default="xy")
    args = parser.parse_args()
    expected = {sequence.name for sequence in _sequences(Path(args.data_root))}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    copied = {}
    for source_value in args.sources:
        source = Path(source_value)
        for prediction in sorted(source.glob("*.txt")):
            if prediction.stem in copied:
                raise ValueError(
                    f"duplicate sequence {prediction.stem}: "
                    f"{copied[prediction.stem]} and {prediction}"
                )
            copied[prediction.stem] = str(prediction)
            shutil.copy2(prediction, output / prediction.name)
    missing = sorted(expected - copied.keys())
    extra = sorted(copied.keys() - expected)
    if missing or extra:
        raise ValueError(
            f"sequence mismatch: missing={missing[:10]} ({len(missing)}), "
            f"extra={extra[:10]} ({len(extra)})"
        )
    zip_path = package(
        output,
        output.with_suffix(".zip"),
        expected=len(expected),
        overwrite=True,
        coordinate_order=args.coordinate_order,
    )
    print(
        json.dumps(
            {
                "sequences": len(expected),
                "sources": list(args.sources),
                "output": str(output),
                "zip": str(zip_path),
                "zip_bytes": zip_path.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
