from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from tracemark.datasets.base import BaseDatasetAdapter
from tracemark.schema import FinalOutcome, NormalizedEvent, Trace

_ERR_PREVIEW_CHARS = 4000


def _tool_message_match_id(tool_msg: dict[str, Any]) -> str | None:
    """OpenAI-style tool_call_id or tau-bench message id tying a tool row to an assistant tool_call."""
    for key in ("tool_call_id", "id"):
        v = tool_msg.get(key)
        if v is not None and str(v).strip():
            return str(v)
    return None


def _normalize_tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _collect_following_tool_messages(messages: list[Any], assistant_idx: int) -> tuple[list[dict[str, Any]], int]:
    """Return (consecutive tool-role messages after assistant_idx, index of next unconsumed message)."""
    tool_rows: list[dict[str, Any]] = []
    j = assistant_idx + 1
    while j < len(messages):
        nxt = messages[j] or {}
        if str(nxt.get("role", "")).lower() != "tool":
            break
        if isinstance(nxt, dict):
            tool_rows.append(nxt)
        j += 1
    return tool_rows, j


def _observations_for_tool_calls(
    tool_calls: list[dict[str, Any]], tool_msgs: list[dict[str, Any]]
) -> list[Any | None]:
    """Align tool message content to each tool_call (match by id, else by order)."""
    observations: list[Any | None] = [None] * len(tool_calls)
    used_msg_idx: set[int] = set()

    for ci, call in enumerate(tool_calls):
        cid = call.get("id")
        if cid is not None:
            cid_s = str(cid)
            for mi, tm in enumerate(tool_msgs):
                if mi in used_msg_idx:
                    continue
                mid = _tool_message_match_id(tm)
                if mid == cid_s:
                    observations[ci] = tm.get("content")
                    used_msg_idx.add(mi)
                    break
        if observations[ci] is None:
            for mi, tm in enumerate(tool_msgs):
                if mi in used_msg_idx:
                    continue
                observations[ci] = tm.get("content")
                used_msg_idx.add(mi)
                break

    return observations


def _coerce_observation_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _tool_timestamp_for_call(call: dict[str, Any], tool_msgs: list[dict[str, Any]], fallback_idx: int) -> Any:
    cid = call.get("id")
    if cid is not None:
        cid_s = str(cid)
        for tm in tool_msgs:
            if _tool_message_match_id(tm) == cid_s:
                return tm.get("timestamp")
    if 0 <= fallback_idx < len(tool_msgs):
        return tool_msgs[fallback_idx].get("timestamp")
    return None


def _summarize_simulation_info(info: Any) -> dict[str, Any] | None:
    """Compact infrastructure error snippet for tooling / annotation UIs (full tracebacks are huge)."""
    if not isinstance(info, dict):
        return None
    err = info.get("error")
    if isinstance(err, str) and len(err) > _ERR_PREVIEW_CHARS:
        err = err[:_ERR_PREVIEW_CHARS] + "\n... [truncated]"
    out: dict[str, Any] = {
        "error_type": info.get("error_type"),
        "error": err,
        "failed_after_attempts": info.get("failed_after_attempts"),
    }
    if not any(v not in (None, "") for v in out.values()):
        return None
    return out


def _coerce_messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        value = parsed
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            messages.append(item)
    return messages


def _coerce_final_outcome(raw_episode: dict[str, Any]) -> FinalOutcome:
    is_correct = raw_episode.get("is_correct")
    if isinstance(is_correct, bool):
        return FinalOutcome.SUCCESS if is_correct else FinalOutcome.FAILURE

    outcome = raw_episode.get("final_outcome")
    if isinstance(outcome, FinalOutcome):
        return outcome
    if isinstance(outcome, str):
        normalized = outcome.strip().lower()
        if normalized in {"success", "successful", "correct"}:
            return FinalOutcome.SUCCESS
        if normalized in {"failure", "failed", "incorrect"}:
            return FinalOutcome.FAILURE
    return FinalOutcome.UNKNOWN


class WhoAndWhenAdapter(BaseDatasetAdapter):
    """Adapter for the WhoAndWhen dataset, which stores each row as one problem-solving episode."""

    dataset_name = "who_and_when"

    def discover(self, source: Path) -> list[Path]:
        if source.is_file():
            return [source]

        candidates: list[Path] = []
        for pattern in ("*.parquet", "*.json", "*.jsonl"):
            candidates.extend(sorted(source.rglob(pattern)))
        if candidates:
            return candidates
        return []

    def load_raw(self, source: Path) -> Iterable[dict[str, Any]]:
        for path in self.discover(source):
            suffix = path.suffix.lower()
            try:
                if suffix == ".parquet":
                    import pandas as pd

                    df = pd.read_parquet(path)
                    for row in df.to_dict(orient="records"):
                        record = dict(row)
                        record["messages"] = _coerce_messages(record.get("history"))
                        record["source_file"] = str(path)
                        yield record
                    continue

                if suffix == ".jsonl":
                    with path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            item = json.loads(line)
                            if isinstance(item, dict):
                                item["source_file"] = str(path)
                                yield item
                    continue

                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload["source_file"] = str(path)
                    yield payload
                elif isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, dict):
                            item["source_file"] = str(path)
                            yield item
            except Exception:
                continue

    def normalize_episode(self, raw_episode: dict[str, Any]) -> Trace:
        episode_id = str(raw_episode.get("question_ID") or raw_episode.get("episode_id") or raw_episode.get("id") or "unknown_episode")
        task_id = str(raw_episode.get("question_ID") or raw_episode.get("task_id") or episode_id)
        goal = self._coerce_goal(raw_episode.get("question") or raw_episode.get("description") or raw_episode.get("user_scenario"))

        final_outcome = _coerce_final_outcome(raw_episode)

        messages = _coerce_messages(raw_episode.get("messages") or raw_episode.get("history"))
        events: list[NormalizedEvent] = []

        for idx, message in enumerate(messages):
            role = str(message.get("role", "message")).lower() or "message"
            content = message.get("content")
            if content is None:
                content = message.get("text")
            if content is None:
                content = ""

            if role == "assistant":
                thought = content
                observation = None
            else:
                thought = None
                observation = content

            events.append(
                NormalizedEvent(
                    episode_id=episode_id,
                    dataset_name=self.dataset_name,
                    task_id=task_id,
                    step_index=idx,
                    timestamp=message.get("timestamp"),
                    goal=goal,
                    agent_thought=thought,
                    action_type="message",
                    action_name=f"{role}_message",
                    action_payload={"speaker": message.get("name")} if message.get("name") else {},
                    tool_name=None,
                    tool_args={},
                    observation=observation,
                    state_snapshot={},
                    reward=float(raw_episode.get("reward")) if isinstance(raw_episode.get("reward"), (int, float)) else None,
                    step_outcome=None,
                    is_terminal=(idx == len(messages) - 1),
                    final_outcome=final_outcome,
                    metadata={
                        "speaker": message.get("name"),
                        "role": role,
                        "source_message": message,
                    },
                )
            )

        if not events:
            payload = json.dumps(raw_episode, ensure_ascii=True)
            events.append(
                NormalizedEvent(
                    episode_id=episode_id,
                    dataset_name=self.dataset_name,
                    task_id=task_id,
                    step_index=0,
                    timestamp=None,
                    goal=goal,
                    action_type="raw_data",
                    action_name="raw_episode",
                    action_payload={},
                    observation=payload,
                    state_snapshot={},
                    reward=float(raw_episode.get("reward")) if isinstance(raw_episode.get("reward"), (int, float)) else None,
                    step_outcome=None,
                    is_terminal=True,
                    final_outcome=final_outcome,
                    metadata={"raw_episode": raw_episode},
                )
            )

        return Trace(
            episode_id=episode_id,
            dataset_name=self.dataset_name,
            task_id=task_id,
            goal=goal,
            events=events,
            final_outcome=final_outcome,
            metadata={
                "source": "who_and_when",
                "source_file": raw_episode.get("source_file"),
                "question": raw_episode.get("question"),
                "ground_truth": raw_episode.get("ground_truth"),
                "mistake_agent": raw_episode.get("mistake_agent"),
                "mistake_step": raw_episode.get("mistake_step"),
                "mistake_reason": raw_episode.get("mistake_reason"),
                "is_correct": raw_episode.get("is_correct"),
                "reward": raw_episode.get("reward"),
                "raw_keys": sorted(raw_episode.keys()),
            },
        )

    def _coerce_goal(self, goal_value: Any) -> str | None:
        if goal_value is None:
            return None
        if isinstance(goal_value, str):
            return goal_value
        if isinstance(goal_value, dict):
            parts = []
            for key in ("purpose", "question", "summary", "task"):
                v = goal_value.get(key)
                if v:
                    parts.append(str(v))
            if parts:
                return " | ".join(parts)
            return json.dumps(goal_value, ensure_ascii=True)
        return str(goal_value)