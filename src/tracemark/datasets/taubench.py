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
        err = err[:_ERR_PREVIEW_CHARS] + "\n… [truncated]"
    out: dict[str, Any] = {
        "error_type": info.get("error_type"),
        "error": err,
        "failed_after_attempts": info.get("failed_after_attempts"),
    }
    if not any(v not in (None, "") for v in out.values()):
        return None
    return out


class TauBenchAdapter(BaseDatasetAdapter):
    """Adapter for tau2-bench result artifacts (results.json/results_reviewed.json)."""

    dataset_name = "taubench"

    def discover(self, source: Path) -> list[Path]:
        if source.is_file():
            return [source]
        # Prefer canonical tau2 metadata files to avoid scanning every per-sim shard.
        direct_results = source / "results.json"
        direct_reviewed = source / "results_reviewed.json"
        if direct_reviewed.exists():
            return [direct_reviewed]
        if direct_results.exists():
            return [direct_results]

        meta_files = sorted(source.rglob("results_reviewed.json")) + sorted(source.rglob("results.json"))
        if meta_files:
            return meta_files

        # Fallback for monolithic exported files (e.g., *_4trials.json).
        return sorted(source.rglob("*.json"))

    def load_raw(self, source: Path) -> Iterable[dict[str, Any]]:
        for path in self.discover(source):
            payload = self._load_results_payload(path)
            if payload is None:
                continue

            if not isinstance(payload, dict):
                continue
            simulations = payload.get("simulations")
            if not isinstance(simulations, list):
                continue

            task_meta: dict[str, dict[str, Any]] = {}
            for task in payload.get("tasks", []) or []:
                if isinstance(task, dict) and "id" in task:
                    task_meta[str(task["id"])] = task

            for sim in simulations:
                if not isinstance(sim, dict):
                    continue
                sim_id = str(sim.get("id", "unknown_simulation"))
                task_id = str(sim.get("task_id", "unknown_task"))
                yield {
                    "source_file": str(path),
                    "sim": sim,
                    "task": task_meta.get(task_id, {}),
                    "episode_id": f"{path.stem}__{sim_id}",
                    "task_id": task_id,
                }

    def _load_results_payload(self, path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        # Support tau2 dir-format: metadata in results.json + simulations/*.json siblings.
        sims_dir = path.parent / "simulations"
        sims = payload.get("simulations")
        if isinstance(sims, list) and (sims or not sims_dir.is_dir()):
            return payload

        if sims_dir.is_dir():
            simulations: list[dict[str, Any]] = []
            for sim_file in sorted(sims_dir.glob("*.json")):
                try:
                    sim_obj = json.loads(sim_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(sim_obj, dict):
                    simulations.append(sim_obj)
            payload["simulations"] = simulations
            return payload

        return payload

    def normalize_episode(self, raw_episode: dict[str, Any]) -> Trace:
        sim = raw_episode.get("sim", {})
        task = raw_episode.get("task", {})
        episode_id = str(raw_episode.get("episode_id", "unknown_episode"))
        task_id = str(raw_episode.get("task_id", "unknown_task"))
        goal = self._coerce_goal(task.get("description") or task.get("user_scenario"))

        reward = (sim.get("reward_info") or {}).get("reward")
        if isinstance(reward, (int, float)):
            final_outcome = FinalOutcome.SUCCESS if reward >= 1.0 else FinalOutcome.FAILURE
        else:
            final_outcome = FinalOutcome.UNKNOWN

        events: list[NormalizedEvent] = []
        messages = sim.get("messages") or []
        step_index = 0
        i = 0

        while i < len(messages):
            msg = messages[i] or {}
            role = str(msg.get("role", "")).lower()
            timestamp = msg.get("timestamp")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls") or []
            raw_data = msg.get("raw_data")

            if role == "assistant" and tool_calls:
                tool_msgs, next_i = _collect_following_tool_messages(messages, i)
                obs_per_call = _observations_for_tool_calls(tool_calls, tool_msgs)
                multi = len(tool_calls) > 1

                for tc_idx, call in enumerate(tool_calls):
                    if not isinstance(call, dict):
                        call = {}
                    tool_name = str(call.get("name", "tool_call"))
                    tool_args = _normalize_tool_arguments(call)
                    observation = obs_per_call[tc_idx] if tc_idx < len(obs_per_call) else None
                    tool_ts = _tool_timestamp_for_call(call, tool_msgs, tc_idx)

                    meta = {
                        "message_role": role,
                        "turn_idx": msg.get("turn_idx"),
                        "cost": msg.get("cost"),
                        "usage": msg.get("usage"),
                    }
                    if multi:
                        meta["parallel_tool_index"] = tc_idx
                        meta["parallel_tool_count"] = len(tool_calls)
                    if tool_ts is not None:
                        meta["tool_message_timestamp"] = tool_ts

                    events.append(
                        NormalizedEvent(
                            episode_id=episode_id,
                            dataset_name=self.dataset_name,
                            task_id=task_id,
                            step_index=step_index,
                            timestamp=timestamp,
                            goal=goal,
                            agent_thought=content if isinstance(content, str) else None,
                            action_type="tool",
                            action_name=tool_name,
                            action_payload={
                                "tool_calls": tool_calls,
                                "tool_call_focus_index": tc_idx,
                                "raw_data": raw_data,
                            },
                            tool_name=tool_name,
                            tool_args=tool_args or {},
                            observation=_coerce_observation_str(observation),
                            state_snapshot={},
                            reward=reward if i == len(messages) - 1 else None,
                            step_outcome=None,
                            is_terminal=False,
                            final_outcome=final_outcome,
                            metadata=meta,
                        )
                    )
                    step_index += 1

                i = next_i
                continue

            action_type = "message"
            action_name = f"{role}_message" if role else "message"
            if role == "tool":
                action_type = "tool_result"
                action_name = "tool_result"

            events.append(
                NormalizedEvent(
                    episode_id=episode_id,
                    dataset_name=self.dataset_name,
                    task_id=task_id,
                    step_index=step_index,
                    timestamp=timestamp,
                    goal=goal,
                    agent_thought=content if role == "assistant" and isinstance(content, str) else None,
                    action_type=action_type,
                    action_name=action_name,
                    action_payload={"raw_data": raw_data} if raw_data else {},
                    tool_name=None,
                    tool_args={},
                    observation=content if isinstance(content, str) else None,
                    state_snapshot={},
                    reward=reward if i == len(messages) - 1 else None,
                    step_outcome=None,
                    is_terminal=False,
                    final_outcome=final_outcome,
                    metadata={
                        "message_role": role,
                        "turn_idx": msg.get("turn_idx"),
                        "cost": msg.get("cost"),
                        "usage": msg.get("usage"),
                    },
                )
            )
            step_index += 1
            i += 1

        if events:
            events[-1].is_terminal = True

        return Trace(
            episode_id=episode_id,
            dataset_name=self.dataset_name,
            task_id=task_id,
            goal=goal,
            events=events,
            final_outcome=final_outcome,
            metadata={
                "source": "taubench",
                "source_file": raw_episode.get("source_file"),
                "simulation_id": sim.get("id"),
                "trial": sim.get("trial"),
                "seed": sim.get("seed"),
                "termination_reason": sim.get("termination_reason"),
                "agent_cost": sim.get("agent_cost"),
                "user_cost": sim.get("user_cost"),
                "reward": reward,
                "reward_info": sim.get("reward_info"),
                "reward_basis": (sim.get("reward_info") or {}).get("reward_basis"),
                "reward_breakdown": (sim.get("reward_info") or {}).get("reward_breakdown"),
                "reward_component_info": (sim.get("reward_info") or {}).get("info"),
                "action_checks": (sim.get("reward_info") or {}).get("action_checks"),
                "env_assertions": (sim.get("reward_info") or {}).get("env_assertions"),
                "communicate_checks": (sim.get("reward_info") or {}).get("communicate_checks"),
                "nl_assertions": (sim.get("reward_info") or {}).get("nl_assertions"),
                "db_check": (sim.get("reward_info") or {}).get("db_check"),
                "evaluation_criteria": task.get("evaluation_criteria"),
                "simulation_info_summary": _summarize_simulation_info(sim.get("info")),
                "task_ticket": task.get("ticket"),
                "raw_keys": sorted(sim.keys()) if isinstance(sim, dict) else [],
            },
        )

    def _coerce_goal(self, goal_value: Any) -> str | None:
        if goal_value is None:
            return None
        if isinstance(goal_value, str):
            return goal_value
        if isinstance(goal_value, dict):
            parts = []
            for k in ["purpose", "relevant_data", "personal_issues", "extra_details", "notes"]:
                v = goal_value.get(k)
                if v:
                    parts.append(f"{k}: {v}")
            if not parts:
                # Fall back to compact JSON for uncommon shapes.
                return json.dumps(goal_value, ensure_ascii=True)
            return " | ".join(parts)
        return str(goal_value)
