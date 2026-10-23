from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

# Trajectory logs often reference legacy paths (e.g. ./data/spider/dev_spider.json).
# InterCode-compatible task files live under data/sql/*/ic_*.json.
_DATASET_SUFFIX_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("spider", "data/sql/spider/ic_spider_dev.json"),
    ("bird", "data/sql/bird/ic_bird.json"),
    ("wikisql", "data/sql/wikisql/ic_wikisql_dev.json"),
)


def intercode_root_from_env() -> Path | None:
    for key in ("TRACEMARK_INTERCODE_ROOT", "INTERCODE_ROOT", "INTERCODE_REPO_ROOT"):
        raw = os.environ.get(key, "").strip()
        if raw:
            p = Path(raw).expanduser()
            if p.is_dir():
                return p.resolve()
    return None


def resolve_intercode_task_path(dataset_path: str | None, intercode_root: Path | None) -> Path | None:
    """Map a trajectory's `dataset` field to an on-disk InterCode task JSON file."""
    if not dataset_path or intercode_root is None:
        return None
    root = intercode_root.resolve()
    rel = dataset_path.strip().lstrip("./")
    direct = root / rel
    if direct.is_file():
        return direct

    low = dataset_path.lower()
    for needle, rel_path in _DATASET_SUFFIX_CANDIDATES:
        if needle in low:
            candidate = root / rel_path
            if candidate.is_file():
                return candidate

    # Last resort: any ic_*.json under data/sql whose name contains a hint.
    sql_dir = root / "data" / "sql"
    if sql_dir.is_dir():
        for sub in sql_dir.iterdir():
            if not sub.is_dir():
                continue
            for task_file in sub.glob("ic_*.json"):
                if sub.name in low or task_file.stem.replace("ic_", "") in low:
                    return task_file
    return None


@lru_cache(maxsize=8)
def _load_task_list(path_str: str) -> list[dict[str, Any]] | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return None


def lookup_intercode_task(
    dataset_path: str | None,
    task_id: Any,
    intercode_root: Path | None,
) -> dict[str, Any] | None:
    """Return task record (query, gold SQL, db, …) for a trajectory episode."""
    task_path = resolve_intercode_task_path(dataset_path, intercode_root)
    if task_path is None:
        return None
    records = _load_task_list(str(task_path))
    if not records:
        return None
    try:
        idx = int(task_id)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(records):
        return None
    row = dict(records[idx])
    row["_task_file"] = str(task_path)
    row["_task_index"] = idx
    return row


def best_turn_snapshot(episode: dict[str, Any]) -> dict[str, Any] | None:
    """Agent SQL + observation at the turn with highest per-turn reward."""
    history = episode.get("turn_history") or {}
    actions = list(history.get("actions") or [])
    observations = list(history.get("observations") or [])
    rewards = list(history.get("rewards") or [])
    valid = list(history.get("valid_action") or [])
    if not actions:
        return None

    summary = episode.get("summary") if isinstance(episode.get("summary"), dict) else {}
    best_idx = summary.get("max_reward_idx")
    if not isinstance(best_idx, int) or best_idx < 0 or best_idx >= len(actions):
        numeric = []
        for i, r in enumerate(rewards):
            try:
                numeric.append((i, float(r)))
            except (TypeError, ValueError):
                continue
        best_idx = max(numeric, key=lambda x: x[1])[0] if numeric else 0

    out: dict[str, Any] = {
        "turn_index": best_idx,
        "action": actions[best_idx] if best_idx < len(actions) else None,
        "observation": observations[best_idx] if best_idx < len(observations) else None,
    }
    if best_idx < len(rewards):
        out["turn_reward"] = rewards[best_idx]
    if best_idx < len(valid):
        out["valid_action"] = valid[best_idx]
    return out
