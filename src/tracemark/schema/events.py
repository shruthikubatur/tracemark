from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FinalOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class NormalizedEvent(BaseModel):
    """Common event schema across heterogeneous agent trajectory datasets."""

    model_config = ConfigDict(extra="allow")

    episode_id: str
    dataset_name: str
    task_id: str
    step_index: int = Field(ge=0)
    timestamp: str | None = None
    goal: str | None = None
    agent_thought: str | None = None
    action_type: str | None = None
    action_name: str | None = None
    action_payload: dict[str, Any] | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    observation: str | None = None
    state_snapshot: dict[str, Any] | None = None
    reward: float | None = None
    step_outcome: str | None = None
    is_terminal: bool = False
    final_outcome: FinalOutcome = FinalOutcome.UNKNOWN
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("episode_id", "dataset_name", "task_id")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must be non-empty")
        return value
