"""Dataset adapters that convert raw agent logs into tracemark's trace format."""

from tracemark.datasets.base import BaseDatasetAdapter
from tracemark.datasets.crmarena import CRMArenaAdapter
from tracemark.datasets.taubench import TauBenchAdapter

ADAPTERS = {
    "crmarena": CRMArenaAdapter,
    "taubench": TauBenchAdapter,
}


def get_adapter(name: str) -> BaseDatasetAdapter:
    key = name.strip().lower()
    if key not in ADAPTERS:
        known = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown dataset {name!r}. Known datasets: {known}")
    return ADAPTERS[key]()


__all__ = ["ADAPTERS", "BaseDatasetAdapter", "CRMArenaAdapter", "TauBenchAdapter", "get_adapter"]
