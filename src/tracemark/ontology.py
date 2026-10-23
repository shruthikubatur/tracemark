"""Failure-label ontology for the annotation tool (step-level + trajectory-level)."""

from __future__ import annotations

from typing import Literal

SCHEMA_VERSION = "labeling_v1_june_2026"

STEP_PRIMITIVES = ["Boundary", "State", "Decision"]

PROPAGATION_SOURCE_PRIMITIVES = list(STEP_PRIMITIVES)

CONFIDENCE_LEVELS = ["high", "medium", "low"]

PRIMITIVE_HELPER_TEXT: dict[str, str] = {
    "Boundary": (
        "The agent maps the goal, entity, scope, time, quantity, or task requirement incorrectly."
    ),
    "State": (
        "The agent fails to acquire, preserve, update, or use relevant context, "
        "or is guided by misleading context."
    ),
    "Decision": (
        "The agent has enough of the right representation and context, but chooses the wrong tool, "
        "wrong sequence, wrong response, or wrong commitment strategy."
    ),
}

SUBTYPES_BY_PRIMITIVE: dict[str, list[str]] = {
    "Boundary": ["semantic", "scope", "temporal", "arithmetic"],
    "State": ["missing_context", "unused_context", "misguiding_context"],
    "Decision": ["tool_selection", "response_selection", "commitment_strategy"],
}

TAGS_BY_PRIMITIVE_SUBTYPE: dict[tuple[str, str], list[str]] = {
    ("Boundary", "semantic"): [
        "goal_misinterpretation",
        "entity_misinterpretation",
        "capability_misinterpretation",
    ],
    ("Boundary", "scope"): ["scope_underreach", "scope_overreach"],
    ("Boundary", "temporal"): [],
    ("Boundary", "arithmetic"): [],
    ("State", "missing_context"): [],
    ("State", "unused_context"): [],
    ("State", "misguiding_context"): [],
    ("Decision", "tool_selection"): ["tool_choice", "tool_sequence"],
    ("Decision", "response_selection"): ["unsupported_conclusion", "hallucinated_response"],
    ("Decision", "commitment_strategy"): ["premature_commitment", "no_clarification"],
}

TRAJECTORY_LABEL_DEFINITIONS: dict[str, str] = {
    "propagation": "An upstream failure affects later steps or the final outcome.",
    "looping": "The agent repeats actions or reasoning without meaningful progress.",
    "external_system": (
        "The trace is materially affected by something outside the agent's internal "
        "reasoning/action loop, such as defective input, missing supplied context, bad tool output, "
        "grader issue, or platform constraint."
    ),
}

# Previous ontology values — rejected on save; loaded rows using these are marked unmapped.
LEGACY_PRIMITIVES = frozenset(
    {
        "Recovery",
        "Looping",
        "Decision Failure",
        "External / System Failure",
        "Propagation",
    }
)

LEGACY_SUBTYPES = frozenset(
    {
        "representation",
        "incomplete_context",
        "corrupt_context",
        "no_retry",
        "no_escalation",
        "redundant_call",
        "retry_loop",
        "wrong_tool",
        "wrong_response",
        "failure_to_defer",
        "failure_to_escalate",
        "resource_limit",
        "tool_api_failure",
        "upstream_input_defect",
        "context_supply_failure",
        "platform_constraint",
    }
)

LEGACY_TAGS = frozenset(
    {
        "fuzzy_entity_resolution",
        "wrong_parameter_value",
        "missing_filter",
        "temporal_parse_error",
        "unsupported_semantic_mapping",
        "stale_context",
        "silent_tool_error",
        "looping",
        "hallucination (ungrounded response)",
        "information_not_acquired",
        "partial_information",
        "unused_information",
        "outdated_information",
        "no_recovery_attempt",
        "attempted_but_failed",
    }
)

LabelStatus = Literal["valid", "unmapped"]


def subtypes_for_primitive(primitive: str) -> list[str]:
    return list(SUBTYPES_BY_PRIMITIVE.get(primitive, []))


def tags_for_primitive_subtype(primitive: str, subtype: str) -> list[str]:
    return list(TAGS_BY_PRIMITIVE_SUBTYPE.get((primitive, subtype), []))


def tag_required(primitive: str, subtype: str) -> bool:
    return bool(tags_for_primitive_subtype(primitive, subtype))


def all_step_subtypes() -> list[str]:
    out: list[str] = []
    for prim in STEP_PRIMITIVES:
        for s in SUBTYPES_BY_PRIMITIVE.get(prim, []):
            if s not in out:
                out.append(s)
    return out


def all_tags() -> list[str]:
    out: list[str] = []
    for tags in TAGS_BY_PRIMITIVE_SUBTYPE.values():
        for t in tags:
            if t not in out:
                out.append(t)
    return out


def _norm(value: str | None) -> str:
    return (value or "").strip()


def is_legacy_label(*, primitive: str, subtype: str, tag: str) -> bool:
    if primitive in LEGACY_PRIMITIVES:
        return True
    if subtype in LEGACY_SUBTYPES:
        return True
    if tag in LEGACY_TAGS:
        return True
    return False


def classify_step_label(
    *,
    primitive: str,
    subtype: str,
    tag: str = "",
) -> LabelStatus:
    prim = _norm(primitive)
    sub = _norm(subtype)
    tg = _norm(tag)
    if is_legacy_label(primitive=prim, subtype=sub, tag=tg):
        return "unmapped"
    if not validate_step_label(primitive=prim, subtype=sub, tag=tg):
        return "valid"
    return "unmapped"


def validate_step_label(
    *,
    primitive: str,
    subtype: str,
    tag: str = "",
) -> list[str]:
    """Return a list of validation error messages (empty when valid)."""
    errors: list[str] = []
    prim = _norm(primitive)
    sub = _norm(subtype)
    tg = _norm(tag)

    if not prim:
        errors.append("primitive is required")
        return errors

    if prim in LEGACY_PRIMITIVES:
        errors.append(f"legacy primitive {prim!r} is no longer valid")
        return errors

    if prim not in STEP_PRIMITIVES:
        errors.append(f"unknown primitive {prim!r}; expected one of: {', '.join(STEP_PRIMITIVES)}")
        return errors

    if not sub:
        errors.append(f"subtype is required for primitive {prim!r}")
        return errors

    if sub in LEGACY_SUBTYPES:
        errors.append(f"legacy subtype {sub!r} is no longer valid")
        return errors

    allowed_subtypes = SUBTYPES_BY_PRIMITIVE.get(prim, [])
    if sub not in allowed_subtypes:
        errors.append(
            f"subtype {sub!r} not valid for {prim!r}; expected one of: {', '.join(allowed_subtypes)}"
        )
        return errors

    if tg in LEGACY_TAGS:
        errors.append(f"legacy tag {tg!r} is no longer valid")
        return errors

    allowed_tags = TAGS_BY_PRIMITIVE_SUBTYPE.get((prim, sub), [])
    if allowed_tags:
        if not tg:
            errors.append(f"tag is required for {prim!r} / {sub!r}")
        elif tg not in allowed_tags:
            errors.append(
                f"tag {tg!r} not valid for {prim!r} / {sub!r}; expected one of: {', '.join(allowed_tags)}"
            )
    elif tg:
        errors.append(f"tag is not allowed for {prim!r} / {sub!r} (leaf label)")

    return errors
