from __future__ import annotations

import json
from pathlib import Path

from tracemark.datasets.taubench import TauBenchAdapter
from tracemark.schema import FinalOutcome


def test_taubench_adapter_normalizes_results_json(tmp_path: Path) -> None:
    sample = {
        "tasks": [
            {
                "id": "task-1",
                "description": "Resolve customer issue",
                "evaluation_criteria": {
                    "nl_assertions": ["Agent verifies policy"],
                    "reward_basis": ["DB"],
                },
            }
        ],
        "simulations": [
            {
                "id": "sim-1",
                "task_id": "task-1",
                "trial": 0,
                "seed": 42,
                "termination_reason": "user_stop",
                "info": {"error_type": "RateLimitError", "error": "rate limited", "failed_after_attempts": 3},
                "reward_info": {"reward": 0.0},
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "c1", "name": "lookup_order", "arguments": {"order_id": "O1"}}],
                        "timestamp": "2026-03-01T00:00:00Z",
                    },
                    {
                        "role": "tool",
                        "content": '{"order_id":"O1","status":"cancelled"}',
                        "tool_calls": None,
                        "timestamp": "2026-03-01T00:00:01Z",
                    },
                    {
                        "role": "assistant",
                        "content": "I cannot uncancel this order.",
                        "tool_calls": None,
                        "timestamp": "2026-03-01T00:00:02Z",
                    },
                ],
            }
        ],
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(sample), encoding="utf-8")

    adapter = TauBenchAdapter()
    traces = list(adapter.iter_traces(results_path))

    assert len(traces) == 1
    trace = traces[0]
    assert trace.dataset_name == "taubench"
    assert trace.goal == "Resolve customer issue"
    assert trace.final_outcome == FinalOutcome.FAILURE
    assert len(trace.events) == 2
    assert trace.events[0].action_name == "lookup_order"
    assert trace.events[0].observation == '{"order_id":"O1","status":"cancelled"}'
    assert trace.events[-1].is_terminal is True
    assert trace.metadata.get("reward_basis") is None
    assert trace.metadata.get("reward_breakdown") is None
    assert trace.metadata.get("evaluation_criteria", {}).get("nl_assertions") == ["Agent verifies policy"]
    sim_info = trace.metadata.get("simulation_info_summary") or {}
    assert sim_info.get("error_type") == "RateLimitError"


def test_taubench_adapter_supports_dir_format_results_with_simulation_shards(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_1"
    sims_dir = run_dir / "simulations"
    sims_dir.mkdir(parents=True)

    meta = {"tasks": [{"id": "task-1", "description": "Resolve issue"}], "simulations": []}
    (run_dir / "results.json").write_text(json.dumps(meta), encoding="utf-8")

    shard = {
        "id": "sim-2",
        "task_id": "task-1",
        "reward_info": {"reward": 1.0},
        "messages": [{"role": "assistant", "content": "Done", "tool_calls": None, "timestamp": "2026-03-01T00:00:00Z"}],
    }
    (sims_dir / "sim-2.json").write_text(json.dumps(shard), encoding="utf-8")

    adapter = TauBenchAdapter()
    traces = list(adapter.iter_traces(run_dir))
    assert len(traces) == 1
    assert traces[0].final_outcome == FinalOutcome.SUCCESS


def test_taubench_adapter_splits_parallel_tool_calls_into_one_step_each(tmp_path: Path) -> None:
    """Multiple tool_calls on one assistant turn share thought/payload but each row gets tool + observation."""
    sample = {
        "tasks": [{"id": "task-1", "description": "Do three lookups"}],
        "simulations": [
            {
                "id": "sim-parallel",
                "task_id": "task-1",
                "trial": 0,
                "reward_info": {"reward": 0.0},
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Fetching three orders.",
                        "tool_calls": [
                            {"id": "a", "name": "get_order", "arguments": {"order_id": "1"}},
                            {"id": "b", "name": "get_order", "arguments": {"order_id": "2"}},
                            {"id": "c", "name": "get_order", "arguments": {"order_id": "3"}},
                        ],
                        "timestamp": "2026-03-01T00:00:00Z",
                    },
                    {"role": "tool", "id": "a", "content": r'{"order_id":"1"}', "timestamp": "2026-03-01T00:00:01Z"},
                    {"role": "tool", "id": "b", "content": r'{"order_id":"2"}', "timestamp": "2026-03-01T00:00:02Z"},
                    {"role": "tool", "id": "c", "content": r'{"order_id":"3"}', "timestamp": "2026-03-01T00:00:03Z"},
                    {
                        "role": "assistant",
                        "content": "Summarized.",
                        "tool_calls": None,
                        "timestamp": "2026-03-01T00:00:04Z",
                    },
                ],
            }
        ],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(sample), encoding="utf-8")

    trace = list(TauBenchAdapter().iter_traces(path))[0]
    assert len(trace.events) == 4
    expected_obs = [r'{"order_id":"1"}', r'{"order_id":"2"}', r'{"order_id":"3"}']
    for i in range(3):
        ev = trace.events[i]
        assert ev.action_type == "tool"
        assert ev.tool_name == "get_order"
        assert ev.tool_args == {"order_id": str(i + 1)}
        assert ev.observation == expected_obs[i]
        assert ev.metadata.get("parallel_tool_count") == 3
        assert ev.metadata.get("parallel_tool_index") == i
    assert trace.events[3].action_type == "message"
    assert trace.events[3].agent_thought == "Summarized."


def test_taubench_adapter_preserves_reward_breakdown_metadata(tmp_path: Path) -> None:
    sample = {
        "tasks": [{"id": "task-1", "description": "Resolve customer issue"}],
        "simulations": [
            {
                "id": "sim-3",
                "task_id": "task-1",
                "termination_reason": "user_stop",
                "reward_info": {
                    "reward": 0.0,
                    "reward_basis": ["ACTION", "ENV_ASSERTION"],
                    "reward_breakdown": {"ACTION": 0.5, "ENV_ASSERTION": 0.0},
                    "action_checks": [{"action_match": True}, {"action_match": False}],
                    "env_assertions": [{"passed": False}],
                    "communicate_checks": [],
                    "nl_assertions": [],
                    "db_check": {"db_match": False},
                    "info": {"action": {"note": "bad action"}},
                },
                "messages": [{"role": "assistant", "content": "Done", "tool_calls": None, "timestamp": "2026-03-01T00:00:00Z"}],
            }
        ],
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(sample), encoding="utf-8")

    adapter = TauBenchAdapter()
    trace = list(adapter.iter_traces(results_path))[0]
    assert trace.metadata.get("reward") == 0.0
    assert trace.metadata.get("reward_basis") == ["ACTION", "ENV_ASSERTION"]
    assert trace.metadata.get("reward_breakdown") == {"ACTION": 0.5, "ENV_ASSERTION": 0.0}
    assert trace.metadata.get("db_check") == {"db_match": False}
