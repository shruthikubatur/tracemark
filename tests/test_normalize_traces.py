from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import test_taubench_adapter

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_normalize_script_writes_ui_readable_jsonl(tmp_path: Path) -> None:
    test_taubench_adapter.test_taubench_adapter_normalizes_results_json(tmp_path)
    out = tmp_path / "out" / "normalized.jsonl"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "normalize_traces.py"),
         "--dataset", "taubench", "--input", str(tmp_path / "results.json"), "--output", str(out)],
        check=True,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    for field in ("episode_id", "final_outcome", "events"):
        assert field in row
    assert all("step_index" in ev for ev in row["events"])
