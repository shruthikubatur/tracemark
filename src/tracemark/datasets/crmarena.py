from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from tracemark.datasets.base import BaseDatasetAdapter
from tracemark.schema import FinalOutcome, NormalizedEvent, Trace

_RE_LFS_POINTER = re.compile(r"^version https://git-lfs.github.com/spec/v1", re.IGNORECASE)
_RE_TAG = {
    "thought": re.compile(r"<thought>(.*?)</thought>", re.IGNORECASE | re.DOTALL),
    "execute": re.compile(r"<execute>(.*?)</execute>", re.IGNORECASE | re.DOTALL),
    "respond": re.compile(r"<respond>(.*?)</respond>", re.IGNORECASE | re.DOTALL),
    "submit": re.compile(r"<submit>(.*?)</submit>", re.IGNORECASE | re.DOTALL),
}
_RE_SF_OUTPUT = re.compile(r"^\s*Salesforce instance output:\s*(.*)$", re.IGNORECASE | re.DOTALL)


class CRMArenaAdapter(BaseDatasetAdapter):
    dataset_name = "crmarena"

    def discover(self, source: Path) -> list[Path]:
        if source.is_file():
            return [source]
        return sorted(source.rglob("results_*.json"))

    def load_raw(self, source: Path) -> Iterable[dict[str, Any]]:
        for path in self.discover(source):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            if _RE_LFS_POINTER.search(text):
                # Git-LFS pointer file, not actual results payload.
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if not isinstance(payload, list):
                continue
            for idx, row in enumerate(payload):
                if not isinstance(row, dict):
                    continue
                yield {
                    "source_file": str(path),
                    "row_index": idx,
                    "row": row,
                }

    def normalize_episode(self, raw_episode: dict[str, Any]) -> Trace:
        row = raw_episode["row"]
        path = Path(raw_episode["source_file"])
        row_idx = int(raw_episode["row_index"])

        task_id = str(row.get("task_id", row_idx))
        episode_id = f"{path.stem}__task_{task_id}__row_{row_idx}"
        reward = row.get("reward")
        if isinstance(reward, (int, float)):
            final_outcome = FinalOutcome.SUCCESS if float(reward) >= (1 - 1e-6) else FinalOutcome.FAILURE
        elif isinstance(reward, dict):
            # Fuzzy metrics payloads: treat exact-match as success, otherwise failure.
            final_outcome = FinalOutcome.SUCCESS if bool(reward.get("em", False)) else FinalOutcome.FAILURE
        else:
            final_outcome = FinalOutcome.UNKNOWN

        events: list[NormalizedEvent] = []
        traj = row.get("traj") or []
        step_index = 0
        i = 0
        while i < len(traj):
            msg = traj[i] if isinstance(traj[i], dict) else {}
            role = str(msg.get("role", "")).lower()
            content = msg.get("content")
            content_text = content if isinstance(content, str) else ""

            if role == "assistant":
                thought = self._extract("thought", content_text)
                execute = self._extract("execute", content_text)
                respond = self._extract("respond", content_text)
                submit = self._extract("submit", content_text)

                if execute:
                    action_name, tool_name, tool_args = self._parse_execute(execute)
                    observation = self._next_salesforce_output(traj, i + 1)
                    events.append(
                        NormalizedEvent(
                            episode_id=episode_id,
                            dataset_name=self.dataset_name,
                            task_id=task_id,
                            step_index=step_index,
                            timestamp=None,
                            goal=None,
                            agent_thought=thought,
                            action_type="tool",
                            action_name=action_name,
                            action_payload={"execute_raw": execute, "assistant_content": content_text},
                            tool_name=tool_name,
                            tool_args=tool_args,
                            observation=observation,
                            state_snapshot={},
                            reward=None,
                            step_outcome=None,
                            is_terminal=False,
                            final_outcome=final_outcome,
                            metadata={"role": role, "source": "assistant_execute"},
                        )
                    )
                    step_index += 1
                elif submit:
                    events.append(
                        NormalizedEvent(
                            episode_id=episode_id,
                            dataset_name=self.dataset_name,
                            task_id=task_id,
                            step_index=step_index,
                            timestamp=None,
                            goal=None,
                            agent_thought=thought,
                            action_type="control",
                            action_name="submit",
                            action_payload={"submit_value": submit, "assistant_content": content_text},
                            tool_name=None,
                            tool_args={},
                            observation=None,
                            state_snapshot={},
                            reward=None,
                            step_outcome=None,
                            is_terminal=False,
                            final_outcome=final_outcome,
                            metadata={"role": role, "source": "assistant_submit"},
                        )
                    )
                    step_index += 1
                else:
                    reply = respond or content_text
                    events.append(
                        NormalizedEvent(
                            episode_id=episode_id,
                            dataset_name=self.dataset_name,
                            task_id=task_id,
                            step_index=step_index,
                            timestamp=None,
                            goal=None,
                            agent_thought=thought,
                            action_type="message",
                            action_name="assistant_response",
                            action_payload={"assistant_content": content_text},
                            tool_name=None,
                            tool_args={},
                            observation=reply,
                            state_snapshot={},
                            reward=None,
                            step_outcome=None,
                            is_terminal=False,
                            final_outcome=final_outcome,
                            metadata={"role": role, "source": "assistant_response"},
                        )
                    )
                    step_index += 1

            elif role == "user":
                obs = self._salesforce_output_from_text(content_text)
                events.append(
                    NormalizedEvent(
                        episode_id=episode_id,
                        dataset_name=self.dataset_name,
                        task_id=task_id,
                        step_index=step_index,
                        timestamp=None,
                        goal=None,
                        agent_thought=None,
                        action_type="message",
                        action_name="user_message",
                        action_payload={"user_content": content_text},
                        tool_name=None,
                        tool_args={},
                        observation=obs or content_text,
                        state_snapshot={},
                        reward=None,
                        step_outcome=None,
                        is_terminal=False,
                        final_outcome=final_outcome,
                        metadata={"role": role, "source": "user_message"},
                    )
                )
                step_index += 1

            i += 1

        if events:
            events[-1].is_terminal = True
            events[-1].reward = 1.0 if final_outcome == FinalOutcome.SUCCESS else 0.0 if final_outcome == FinalOutcome.FAILURE else None

        return Trace(
            episode_id=episode_id,
            dataset_name=self.dataset_name,
            task_id=task_id,
            goal=None,
            events=events,
            final_outcome=final_outcome,
            metadata={
                "source": "crmarena",
                "source_file": str(path),
                "task_type": row.get("task_type"),
                "ground_truth_answer": row.get("gt_answer"),
                "reward": reward,
                "agent_info": row.get("agent_info"),
                "row_index": row_idx,
                "raw_keys": sorted(row.keys()),
            },
        )

    def _extract(self, tag: str, text: str) -> str | None:
        m = _RE_TAG[tag].search(text)
        if not m:
            return None
        value = m.group(1).strip()
        return value or None

    def _salesforce_output_from_text(self, text: str) -> str | None:
        m = _RE_SF_OUTPUT.search(text or "")
        if not m:
            return None
        return m.group(1).strip() or None

    def _next_salesforce_output(self, traj: list[Any], start_idx: int) -> str | None:
        if start_idx >= len(traj):
            return None
        nxt = traj[start_idx] if isinstance(traj[start_idx], dict) else {}
        if str(nxt.get("role", "")).lower() != "user":
            return None
        return self._salesforce_output_from_text(str(nxt.get("content", "")))

    def _parse_execute(self, execute_text: str) -> tuple[str, str, dict[str, Any]]:
        q = execute_text.strip()
        low = q.lower()
        if low.startswith("select "):
            return ("soql_select", "salesforce_soql", {"query": q, "query_type": "select"})
        if low.startswith("find "):
            return ("sosl_find", "salesforce_sosl", {"query": q, "query_type": "find"})
        if low.startswith("update "):
            return ("soql_update", "salesforce_soql", {"query": q, "query_type": "update"})
        return ("execute_query", "salesforce_query", {"query": q, "query_type": "unknown"})
