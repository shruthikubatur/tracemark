#!/usr/bin/env python3
"""Write a subset of normalized.jsonl containing only episodes listed in a golden manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _episode_order(manifest: dict) -> list[str]:
    eps = manifest.get("episodes") or []
    out: list[str] = []
    for row in eps:
        if isinstance(row, dict) and row.get("episode_id"):
            out.append(str(row["episode_id"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Golden manifest JSON (split or full)")
    p.add_argument("--normalized", type=Path, required=True, help="Source normalized.jsonl")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    args = p.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Missing manifest: {args.manifest}")
    if not args.normalized.exists():
        raise SystemExit(f"Missing normalized: {args.normalized}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    order = _episode_order(manifest)
    want = set(order)
    if not want:
        raise SystemExit("Manifest has no episode_id entries.")

    by_id: dict[str, dict] = {}
    with args.normalized.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            eid = row.get("episode_id")
            if eid is None:
                continue
            eid = str(eid)
            if eid in want:
                by_id[eid] = row

    missing = [eid for eid in order if eid not in by_id]
    if missing:
        print(f"Warning: {len(missing)} manifest episode_ids not found in normalized.jsonl.", file=sys.stderr)
        for eid in missing[:12]:
            print(f"  missing: {eid}", file=sys.stderr)
        if len(missing) > 12:
            print(f"  ... and {len(missing) - 12} more", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for eid in order:
            if eid in by_id:
                out.write(json.dumps(by_id[eid], ensure_ascii=False) + "\n")

    print(f"Wrote {sum(1 for eid in order if eid in by_id)} traces → {args.output}")


if __name__ == "__main__":
    main()
