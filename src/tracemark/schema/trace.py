from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tracemark.schema.events import FinalOutcome, NormalizedEvent


class Trace(BaseModel):
    """Normalized episode-level container for ordered trajectory events."""

    model_config = ConfigDict(extra="allow")

    episode_id: str
    dataset_name: str
    task_id: str
    goal: str | None = None
    events: list[NormalizedEvent] = Field(default_factory=list)
    final_outcome: FinalOutcome = FinalOutcome.UNKNOWN
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_order(self) -> "Trace":
        step_indexes = [event.step_index for event in self.events]
        if step_indexes != sorted(step_indexes):
            raise ValueError("Events must be sorted by step_index")
        return self

    def prefix(self, end_step: int) -> "Trace":
        """Return events up to and including end_step."""
        return Trace(
            episode_id=self.episode_id,
            dataset_name=self.dataset_name,
            task_id=self.task_id,
            goal=self.goal,
            events=[e for e in self.events if e.step_index <= end_step],
            final_outcome=self.final_outcome,
            metadata=self.metadata,
        )
