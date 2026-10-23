#!/usr/bin/env python3
"""Split a golden manifest into two (or more) disjoint episode lists for parallel labeling."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="golden_manifest_balanced_v1 JSON")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for split manifests (default: same directory as --input)",
    )
    p.add_argument(
        "--label-a",
        default="part_a",
        help="Filename suffix / chunk label for the first split (preserves manifest order)",
    )
    p.add_argument(
        "--label-b",
        default="part_b",
        help="Filename suffix / chunk label for the second split",
    )
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing manifest: {args.input}")

    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.input.read_text(encoding="utf-8"))
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise SystemExit("Manifest has no episodes list.")

    n = len(episodes)
    mid = n // 2
    part_a = episodes[:mid]
    part_b = episodes[mid:]
    if n % 2 == 1:
        print(f"Warning: odd episode count ({n}); part_a has {len(part_a)}, part_b has {len(part_b)}.", file=sys.stderr)

    stem = args.input.stem

    def write_chunk(label: str, chunk: list, other_label: str) -> Path:
        out = deepcopy(data)
        out["schema_version"] = str(data.get("schema_version", "unknown")) + "_split"
        out["split_from"] = str(args.input.resolve())
        out["split_chunk"] = label
        out["split_other_chunk"] = other_label
        out["episodes"] = chunk
        out["n_selected"] = len(chunk)
        path = out_dir / f"{stem}__{label}.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return path

    pa = write_chunk(args.label_a, part_a, args.label_b)
    pb = write_chunk(args.label_b, part_b, args.label_a)
    print(f"Wrote {len(part_a)} episodes → {pa}")
    print(f"Wrote {len(part_b)} episodes → {pb}")


if __name__ == "__main__":
    main()
