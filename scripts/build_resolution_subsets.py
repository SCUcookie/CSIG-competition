"""Group sequence directories by frame resolution using symlinks."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    examples = {}
    for sequence in sorted(source.iterdir()):
        image_dir = sequence / "img"
        if not sequence.is_dir() or not image_dir.is_dir():
            continue
        first = next((path for path in image_dir.iterdir() if path.is_file()), None)
        if first is None:
            continue
        width, height = Image.open(first).size
        resolution = f"{width}x{height}"
        destination = output / resolution
        destination.mkdir(exist_ok=True)
        link = destination / sequence.name
        if not link.exists():
            link.symlink_to(sequence, target_is_directory=True)
        counts[resolution] += 1
        examples.setdefault(resolution, sequence.name)
    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "sequences": sum(counts.values()),
                "counts": dict(sorted(counts.items())),
                "examples": dict(sorted(examples.items())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
