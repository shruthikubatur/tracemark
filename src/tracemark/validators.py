from __future__ import annotations

from tracemark.schema import Trace


def validate_trace(trace: Trace) -> list[str]:
    """Return validation warnings; do not raise to preserve research iteration speed."""
    warnings: list[str] = []
    if not trace.events:
        warnings.append("Trace has no events")
    if trace.events and trace.events[-1].is_terminal is False:
        warnings.append("Last event is not marked terminal")
    return warnings
