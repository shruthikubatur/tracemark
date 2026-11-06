#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
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
        with path.open("rb") as f:
            head = f.read(64)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared = repo_root / "results" / "shared"
    if not shared.is_dir():
        sys.exit(
            f"Error: {shared} does not exist. --repo-root must point at your clone of "
            "SalesforceAIResearch/CRMArena (the folder that contains results/)."
        )

    candidates = sorted(shared.rglob("results_*.json"))
    counts = {"found": len(candidates), "after_org": 0, "after_turns": 0, "after_model": 0}
    selected: list[Path] = []
    for p in candidates:
        rel = p.relative_to(repo_root).as_posix()
        if args.org != "all" and f"/{args.org}/" not in rel:
            continue
        counts["after_org"] += 1
        if args.turns != "all" and f"/{args.turns}/" not in rel:
            continue
        counts["after_turns"] += 1
        if args.model_contains and args.model_contains not in p.name:
            continue
        counts["after_model"] += 1
        selected.append(p)
        if len(selected) >= args.max_files:
            break

    if not selected:
        sys.exit(
            "Error: no files matched. Counts at each filter: "
            f"{json.dumps(counts)}. Check --org, --turns and --model-contains."
        )

    preview = [
        {
            "path": str(p.relative_to(repo_root)),
            "source": "download" if is_lfs_pointer(p) else "copy from clone",
        }
        for p in selected[:10]
    ]
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

    # Files are Git LFS objects. If git-lfs is installed, the clone already holds the real
    # JSON and we copy it. Otherwise the clone holds small pointer files and we download.
    downloaded = 0
    copied = 0
    for i, p in enumerate(selected, 1):
        rel = p.relative_to(repo_root).as_posix()
        target = out_dir / p.name
        if is_lfs_pointer(p):
            url = f"https://media.githubusercontent.com/media/SalesforceAIResearch/CRMArena/main/{rel}"
            print(f"[{i}/{len(selected)}] downloading {p.name}", file=sys.stderr)
            with urlopen(url, timeout=120) as r:  # nosec - benchmark artifact fetch
                target.write_bytes(r.read())
            downloaded += 1
        else:
            print(f"[{i}/{len(selected)}] copying {p.name}", file=sys.stderr)
            shutil.copyfile(p, target)
            copied += 1

    print(
        json.dumps(
            {
                "mode": "download",
                "downloaded_files": downloaded,
                "copied_files": copied,
                "output_dir": str(out_dir),
                "preview": [p.name for p in selected[:10]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
