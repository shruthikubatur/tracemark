#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download CRMArena shared results JSONs from GitHub media URLs.")
    p.add_argument("--repo-root", required=True, help="Local clone of SalesforceAIResearch/CRMArena (used to discover result paths).")
    p.add_argument("--output-dir", required=True, help="Where to save downloaded JSON files.")
    p.add_argument("--org", default="b2b", choices=["b2b", "b2c", "original", "all"])
    p.add_argument("--turns", default="multi-turn", choices=["single-turn", "multi-turn", "all"])
    p.add_argument("--model-contains", default="o1-2024-12-17", help="Only download files whose name contains this substring.")
    p.add_argument("--max-files", type=int, default=40)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def is_lfs_pointer(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return text.startswith("version https://git-lfs.github.com/spec/v1")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted((repo_root / "results" / "shared").rglob("results_*.json"))
    selected: list[Path] = []
    for p in candidates:
        rel = p.relative_to(repo_root).as_posix()
        if args.org != "all" and f"/{args.org}/" not in rel:
            continue
        if args.turns != "all" and f"/{args.turns}/" not in rel:
            continue
        if args.model_contains and args.model_contains not in p.name:
            continue
        if not is_lfs_pointer(p):
            # already materialized; copy path is still useful, but we skip here to keep behavior explicit.
            continue
        selected.append(p)
        if len(selected) >= args.max_files:
            break

    preview = [{"path": str(p.relative_to(repo_root)), "name": p.name} for p in selected[:10]]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "selected_files": len(selected),
                    "preview": preview,
                },
                indent=2,
            )
        )
        return

    downloaded = 0
    for p in selected:
        rel = p.relative_to(repo_root).as_posix()
        url = f"https://media.githubusercontent.com/media/SalesforceAIResearch/CRMArena/main/{rel}"
        target = out_dir / p.name
        with urlopen(url) as r:  # nosec - benchmark artifact fetch
            data = r.read()
        target.write_bytes(data)
        downloaded += 1

    print(
        json.dumps(
            {
                "mode": "download",
                "downloaded_files": downloaded,
                "output_dir": str(out_dir),
                "preview": [p.name for p in selected[:10]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
