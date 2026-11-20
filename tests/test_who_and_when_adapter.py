from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tracemark.datasets.who_and_when import WhoAndWhenAdapter
from tracemark.schema import FinalOutcome


def test_who_and_when_adapter_normalizes_simple_json(tmp_path: Path) -> None:
    sample_data = {
        "id": "episode-1",
        "task_id": "task-a",
        "description": "Test who and when adapter",
        "messages": [
            {"role": "user", "content": "Hello", "timestamp": "2026-01-01T00:00:00Z"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "2026-01-01T00:00:01Z"},
        ],
        "final_outcome": "SUCCESS",
        "reward": 1.0
    }
    
    test_file = tmp_path / "handcrafted.json"
    test_file.write_text(json.dumps(sample_data), encoding="utf-8")

    adapter = WhoAndWhenAdapter()
    traces = list(adapter.iter_traces(test_file))

    assert len(traces) == 1
    trace = traces[0]
    assert trace.dataset_name == "who_and_when"
    assert trace.goal == "Test who and when adapter"
    assert trace.final_outcome == FinalOutcome.SUCCESS
    assert len(trace.events) == 2
    assert trace.events[0].action_name == "user_message"
    assert trace.events[0].observation == "Hello"
    assert trace.events[1].action_name == "assistant_message"
    assert trace.events[1].agent_thought == "Hi there!"
    assert trace.events[-1].is_terminal is True


def test_who_and_when_adapter_normalizes_jsonl_file(tmp_path: Path) -> None:
    sample_data_1 = {
        "id": "episode-2-1",
        "messages": [
            {"role": "user", "content": "First line user", "timestamp": "2026-01-01T00:00:00Z"}
        ]
    }
    sample_data_2 = {
        "id": "episode-2-2",
        "messages": [
            {"role": "assistant", "content": "Second line assistant", "timestamp": "2026-01-01T00:00:01Z"}
        ]
    }

    test_file = tmp_path / "multi.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_data_1) + "\n")
        f.write(json.dumps(sample_data_2) + "\n")

    adapter = WhoAndWhenAdapter()
    traces = list(adapter.iter_traces(test_file))

    assert len(traces) == 2
    assert traces[0].episode_id == "episode-2-1"
    assert traces[1].episode_id == "episode-2-2"
    assert traces[0].events[0].observation == "First line user"
    assert traces[1].events[0].agent_thought == "Second line assistant"


def test_who_and_when_adapter_normalizes_parquet_rows(tmp_path: Path) -> None:
    history = [
        {"role": "assistant", "name": "Verifier", "content": "I checked the prices."},
        {"role": "user", "name": "User", "content": "What is the final answer?"},
    ]
    df = pd.DataFrame([
        {
            "question_ID": "qid-1",
            "question": "How much did I save?",
            "ground_truth": "$120",
            "history": history,
            "is_correct": False,
            "mistake_agent": "Verifier",
            "mistake_step": "2",
            "mistake_reason": "Wrong answer",
            "reward": 0.0,
        }
    ])
    parquet_path = tmp_path / "Algorithm-Generated.parquet"
    df.to_parquet(parquet_path)

    trace = list(WhoAndWhenAdapter().iter_traces(parquet_path))[0]

    assert trace.episode_id == "qid-1"
    assert trace.goal == "How much did I save?"
    assert trace.final_outcome == FinalOutcome.FAILURE
    assert len(trace.events) == 2
    assert trace.events[0].agent_thought == "I checked the prices."
    assert trace.events[1].observation == "What is the final answer?"
