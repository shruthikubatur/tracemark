from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tracemark.intercode.task_lookup import (
    intercode_root_from_env,
    lookup_intercode_task,
)

# Defaults match the stock InterCode SQL Docker container. Override with env vars.
DEFAULT_MYSQL_HOST = os.environ.get("TRACEMARK_MYSQL_HOST", "127.0.0.1")
DEFAULT_MYSQL_PORT = int(os.environ.get("TRACEMARK_MYSQL_PORT", "3307"))
DEFAULT_MYSQL_USER = os.environ.get("TRACEMARK_MYSQL_USER", "admin")
DEFAULT_MYSQL_PASSWORD = os.environ.get("TRACEMARK_MYSQL_PASSWORD", "admin")


@dataclass(frozen=True)
class MySQLConfig:
    host: str = DEFAULT_MYSQL_HOST
    port: int = DEFAULT_MYSQL_PORT
    user: str = DEFAULT_MYSQL_USER
    password: str = DEFAULT_MYSQL_PASSWORD

    def connect_kwargs(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
        }


def _require_mysql_connector():
    try:
        import mysql.connector  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "mysql-connector-python is required to run gold SQL against InterCode MySQL. "
            "Install with: pip install mysql-connector-python"
        ) from exc
    return mysql.connector


def check_mysql_connection(config: MySQLConfig | None = None) -> tuple[bool, str]:
    """Return (ok, message) for a quick connectivity probe."""
    config = config or MySQLConfig()
    try:
        connector = _require_mysql_connector()
        cnx = connector.connect(**config.connect_kwargs())
        cur = cnx.cursor()
        cur.execute("SELECT 1")
        cur.fetchall()
        cur.close()
        cnx.close()
        return True, f"Connected to MySQL at {config.host}:{config.port} as {config.user}"
    except Exception as exc:
        return False, f"MySQL connection failed ({config.host}:{config.port}): {exc}"


def serialize_sql_result(rows: Any | None, *, error: str | None = None) -> str:
    """Match InterCode SqlEnv string formatting for observations / eval_obs."""
    if error:
        return f"Error executing query: {error}"
    if rows is None:
        return "None"
    return str(rows)


def execute_gold_sql(db: str, gold_sql: str, config: MySQLConfig | None = None) -> dict[str, Any]:
    """
    Run `USE <db>` then gold SQL the same way InterCode grades submissions.

    Returns dict with keys: eval_obs (str), row_count (int|None), error (str|None), corrupt_gold (bool).
    """
    config = config or MySQLConfig()
    connector = _require_mysql_connector()
    cnx = connector.connect(**config.connect_kwargs())
    cur = cnx.cursor(buffered=True)
    try:
        cur.execute(f"USE {db}")
        cur.execute(gold_sql)
        if cur.description is not None:
            rows = cur.fetchall()
            return {
                "eval_obs": serialize_sql_result(rows),
                "row_count": len(rows),
                "error": None,
                "corrupt_gold": False,
            }
        return {
            "eval_obs": serialize_sql_result([]),
            "row_count": 0,
            "error": None,
            "corrupt_gold": False,
        }
    except Exception as exc:
        msg = getattr(exc, "msg", None) or str(exc)
        return {
            "eval_obs": serialize_sql_result(None, error=msg),
            "row_count": None,
            "error": msg,
            "corrupt_gold": True,
        }
    finally:
        cur.close()
        cnx.close()


def gold_cache_key(db: str, gold_sql: str) -> str:
    digest = hashlib.sha256(f"{db}\n{gold_sql}".encode("utf-8")).hexdigest()
    return digest[:24]


def load_eval_obs_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_eval_obs_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def resolve_gold_for_trace(
    trace: dict[str, Any],
    intercode_root: Path | None,
) -> tuple[str | None, str | None, str | None]:
    """Return (db, gold_sql, error_message)."""
    md = trace.get("metadata") or {}
    gold = md.get("gold_sql") or md.get("ground_truth_sql")
    db = md.get("task_db")
    if gold and db:
        return str(db), str(gold), None

    root = intercode_root or intercode_root_from_env()
    if root is None:
        return None, None, "missing gold_sql/task_db and no intercode root for lookup"

    task = lookup_intercode_task(md.get("dataset_path"), trace.get("task_id"), root)
    if not task:
        return None, None, "could not resolve task file for episode"
    if not task.get("gold"):
        return None, None, "task record has no gold SQL"
    return str(task.get("db", "")), str(task["gold"]), None


def enrich_trace_eval_obs(
    trace: dict[str, Any],
    *,
    config: MySQLConfig | None = None,
    intercode_root: Path | None = None,
    cache: dict[str, dict[str, Any]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Return a copy of the trace with metadata.expected_eval_obs populated from MySQL.
    Uses optional in-memory cache keyed by (db, gold_sql).
    """
    out = dict(trace)
    md = dict(out.get("metadata") or {})

    if md.get("expected_eval_obs") and not force and not md.get("expected_eval_obs_error"):
        out["metadata"] = md
        return out

    db, gold, err = resolve_gold_for_trace(trace, intercode_root)
    if err:
        md["expected_eval_obs_error"] = err
        out["metadata"] = md
        return out

    assert db is not None and gold is not None
    key = gold_cache_key(db, gold)
    if cache is not None and key in cache and not force:
        cached = cache[key]
        md["expected_eval_obs"] = cached.get("eval_obs")
        md["expected_eval_obs_row_count"] = cached.get("row_count")
        md["expected_eval_obs_error"] = cached.get("error")
        md["gold_eval_source"] = "mysql_cache"
        md["gold_eval_cache_key"] = key
        out["metadata"] = md
        return out

    result = execute_gold_sql(db, gold, config=config)
    if cache is not None:
        cache[key] = result

    md["expected_eval_obs"] = result["eval_obs"]
    md["expected_eval_obs_row_count"] = result["row_count"]
    md["expected_eval_obs_error"] = result["error"]
    md["gold_eval_corrupt"] = result["corrupt_gold"]
    md["gold_eval_source"] = "mysql_rerun"
    md["gold_eval_cache_key"] = key
    md["task_db"] = db
    md["gold_sql"] = gold
    out["metadata"] = md
    return out
