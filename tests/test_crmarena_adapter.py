from __future__ import annotations

import json
from pathlib import Path

from tracemark.datasets.crmarena import CRMArenaAdapter
from tracemark.schema import FinalOutcome


def test_crmarena_adapter_parses_results_row() -> None:
    adapter = CRMArenaAdapter()
    raw = {
        "source_file": "/tmp/results_test.json",
        "row_index": 0,
        "row": {
            "task_id": 42,
            "task_type": "quote_approval",
            "gt_answer": ["X"],
            "reward": 1,
            "agent_info": {"k": "v"},
            "traj": [
                {"role": "user", "content": "Need help"},
                {"role": "assistant", "content": "<thought>t</thought><execute>SELECT Id FROM Account</execute>"},
                {"role": "user", "content": "Salesforce instance output: [{'Id':'001'}]"},
                {"role": "assistant", "content": "<respond>Done</respond>"},
            ],
        },
    }
    trace = adapter.normalize_episode(raw)
    assert trace.dataset_name == "crmarena"
    assert trace.final_outcome == FinalOutcome.SUCCESS
    assert len(trace.events) >= 3
    assert any(e.action_name == "soql_select" for e in trace.events)
    assert trace.metadata.get("task_type") == "quote_approval"


def test_crmarena_adapter_skips_lfs_pointer_file(tmp_path: Path) -> None:
    p = tmp_path / "results_x.json"
    p.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 123\n",
        encoding="utf-8",
    )
    adapter = CRMArenaAdapter()
    rows = list(adapter.load_raw(p))
    assert rows == []


def test_crmarena_adapter_loads_real_json_file(tmp_path: Path) -> None:
    payload = [
        {
            "task_id": 1,
            "task_type": "case_routing",
            "gt_answer": ["a"],
            "reward": 0,
            "agent_info": {},
            "traj": [{"role": "assistant", "content": "<respond>x</respond>"}],
        }
    ]
    p = tmp_path / "results_demo.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    adapter = CRMArenaAdapter()
    rows = list(adapter.load_raw(p))
    assert len(rows) == 1
