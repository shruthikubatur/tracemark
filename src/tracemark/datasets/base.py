from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Any

from tracemark.schema import Trace


class BaseDatasetAdapter(ABC):
    dataset_name: str

    @abstractmethod
    def discover(self, source: Path) -> list[Path]:
        """Discover raw files under source path."""

    @abstractmethod
    def load_raw(self, source: Path) -> Iterable[dict[str, Any]]:
        """Load raw episodes in dataset-native format."""

    @abstractmethod
    def normalize_episode(self, raw_episode: dict[str, Any]) -> Trace:
        """Map one raw episode into the common Trace schema."""

    def iter_traces(self, source: Path) -> Iterable[Trace]:
        for episode in self.load_raw(source):
            yield self.normalize_episode(episode)
