from __future__ import annotations

import pytest

from tracemark.ontology import (
    LEGACY_PRIMITIVES,
    LEGACY_SUBTYPES,
    LEGACY_TAGS,
    SCHEMA_VERSION,
    STEP_PRIMITIVES,
    classify_step_label,
    validate_step_label,
)


@pytest.mark.parametrize(
    ("primitive", "subtype", "tag"),
    [
        ("Boundary", "semantic", "goal_misinterpretation"),
        ("Boundary", "semantic", "entity_misinterpretation"),
        ("Boundary", "semantic", "capability_misinterpretation"),
        ("Boundary", "scope", "scope_underreach"),
        ("Boundary", "scope", "scope_overreach"),
        ("Boundary", "temporal", ""),
        ("Boundary", "arithmetic", ""),
        ("State", "missing_context", ""),
        ("State", "unused_context", ""),
        ("State", "misguiding_context", ""),
        ("Decision", "tool_selection", "tool_choice"),
        ("Decision", "tool_selection", "tool_sequence"),
        ("Decision", "response_selection", "unsupported_conclusion"),
        ("Decision", "response_selection", "hallucinated_response"),
        ("Decision", "commitment_strategy", "premature_commitment"),
        ("Decision", "commitment_strategy", "no_clarification"),
    ],
)
def test_new_labels_are_accepted(primitive: str, subtype: str, tag: str) -> None:
    assert validate_step_label(primitive=primitive, subtype=subtype, tag=tag) == []
    assert classify_step_label(primitive=primitive, subtype=subtype, tag=tag) == "valid"


@pytest.mark.parametrize(
    ("primitive", "subtype", "tag", "fragment"),
    [
        ("Recovery", "no_clarification", "", "legacy primitive"),
        ("Looping", "retry_loop", "", "legacy primitive"),
        ("Decision Failure", "wrong_tool", "", "legacy primitive"),
        ("External / System Failure", "platform_constraint", "", "legacy primitive"),
        ("Boundary", "representation", "", "legacy subtype"),
        ("State", "incomplete_context", "", "legacy subtype"),
        ("State", "corrupt_context", "", "legacy subtype"),
        ("Decision", "wrong_tool", "", "legacy subtype"),
        ("Boundary", "semantic", "stale_context", "legacy tag"),
        ("Boundary", "semantic", "fuzzy_entity_resolution", "legacy tag"),
        ("State", "missing_context", "unused_information", "legacy tag"),
    ],
)
def test_legacy_labels_are_rejected(primitive: str, subtype: str, tag: str, fragment: str) -> None:
    errors = validate_step_label(primitive=primitive, subtype=subtype, tag=tag)
    assert errors, f"expected rejection for {fragment}"
    assert classify_step_label(primitive=primitive, subtype=subtype, tag=tag) == "unmapped"


def test_tag_required_for_semantic_but_not_temporal() -> None:
    assert validate_step_label(primitive="Boundary", subtype="semantic", tag="") != []
    assert validate_step_label(primitive="Boundary", subtype="temporal", tag="") == []
    assert validate_step_label(primitive="Boundary", subtype="temporal", tag="goal_misinterpretation") != []


def test_state_labels_are_leaf_without_tag() -> None:
    for subtype in ("missing_context", "unused_context", "misguiding_context"):
        assert validate_step_label(primitive="State", subtype=subtype, tag="") == []


def test_schema_version_bumped() -> None:
    assert SCHEMA_VERSION == "labeling_v1_june_2026"


def test_step_primitives_are_exactly_three() -> None:
    assert STEP_PRIMITIVES == ["Boundary", "State", "Decision"]
    assert "Recovery" not in STEP_PRIMITIVES
    assert "Recovery" in LEGACY_PRIMITIVES
    assert "incomplete_context" in LEGACY_SUBTYPES
    assert "stale_context" in LEGACY_TAGS
