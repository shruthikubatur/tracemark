#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import sys

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_TRACES = REPO_ROOT / "examples" / "normalized.jsonl"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tracemark.intercode.sql_eval import (
    MySQLConfig,
    check_mysql_connection,
    execute_gold_sql,
    resolve_gold_for_trace,
)
from tracemark.intercode.task_lookup import (
    intercode_root_from_env,
    lookup_intercode_task,
)
from tracemark.ontology import (
    CONFIDENCE_LEVELS,
    PRIMITIVE_HELPER_TEXT,
    PROPAGATION_SOURCE_PRIMITIVES,
    SCHEMA_VERSION,
    STEP_PRIMITIVES,
    TRAJECTORY_LABEL_DEFINITIONS,
    classify_step_label,
    subtypes_for_primitive,
    tags_for_primitive_subtype,
    validate_step_label,
)

LOOPING_PATTERN_EXAMPLES = "repeated_tool_call, retry_loop, redundant_call, repeated_restatement_of_same_task_set, …"


def _short(text: str | None, n: int = 220) -> str:
    if not text:
        return ""
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 3] + "..."


def _short_json(value: Any, n: int = 220) -> str:
    if value is None:
        return ""
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except Exception:
        text = str(value)
    return _short(text, n=n)


def _state_snapshot_summary(snap: Any, n: int = 140) -> str:
    if not snap or not isinstance(snap, dict):
        return ""
    url = snap.get("url")
    if url is not None and str(url).strip():
        u = str(url).strip()
        return u if len(u) <= n else u[: n - 3] + "..."
    page = snap.get("page")
    if page is not None and str(page).strip():
        return str(page).strip()
    return _short_json(snap, n=n)


def _event_metadata_summary(md: Any, n: int = 120) -> str:
    if not md or not isinstance(md, dict):
        return ""
    parts: list[str] = []
    for key in ("message_role", "role", "source", "turn_idx", "cost"):
        if key in md and md[key] is not None and md[key] != "":
            parts.append(f"{key}={md[key]}")
    if parts:
        s = " | ".join(parts)
        return s if len(s) <= n else s[: n - 3] + "..."
    return _short_json(md, n=n)


# Datasets with built-in adapters. They always appear in the run dropdown, and their
# traces are expected at outputs/<name>/normalized.jsonl.
BUILTIN_DATASET_HELP = {
    "taubench": (
        "python scripts/normalize_traces.py --dataset taubench \\\n"
        "  --input path/to/results.json \\\n"
        "  --output outputs/taubench/normalized.jsonl"
    ),
    "crmarena": (
        "git clone https://github.com/SalesforceAIResearch/CRMArena.git /tmp/crmarena\n"
        "python scripts/download_crmarena_results.py --repo-root /tmp/crmarena \\\n"
        "  --output-dir data/raw/crmarena\n"
        "python scripts/normalize_traces.py --dataset crmarena \\\n"
        "  --input data/raw/crmarena \\\n"
        "  --output outputs/crmarena/normalized.jsonl"
    ),
}


def _run_options(discovered: list[str]) -> list[str]:
    """Discovered runs first, then the bundled examples, then built-in datasets not yet converted."""
    options = list(discovered)
    if EXAMPLE_TRACES.is_file():
        options.append("examples")
    options.extend(name for name in BUILTIN_DATASET_HELP if name not in discovered)
    return options


def _discover_runs(outputs_root: str = "outputs") -> list[str]:
    root = Path(outputs_root)
    if not root.exists():
        return []
    runs: list[str] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "normalized.jsonl").exists():
            runs.append(d.name)
    return runs


def _resolve_paths_for_run(run_name: str, outputs_root: str = "outputs") -> tuple[str, str]:
    base = Path(outputs_root) / run_name
    normalized_path = str(base / "normalized.jsonl")
    annotations_path = str(base / "annotations" / "labels_v0.jsonl")
    return normalized_path, annotations_path


ROOT_CAUSE_OPTIONS = ("No", "Yes")

ROOT_CAUSE_TOOLTIP = (
    "Mark Yes if this step is a root cause of why the trace failed. "
    "Failure traces require at least one step labeled Yes."
)

SECONDARY_TOOLTIP = (
    "Mark if this label is another secondary cause of the failure "
    "(in addition to the primary/root cause)."
)


def _empty_step_row() -> dict[str, Any]:
    return {
        "step": 0,
        "primitive": "Boundary",
        "subtype": "semantic",
        "tag": "goal_misinterpretation",
        "explanation": "",
        "confidence": "medium",
        "is_root_cause": False,
        "is_secondary": False,
        "label_status": "",
    }


def _empty_propagation_row() -> dict[str, Any]:
    return {
        "source_step": 0,
        "source_primitive": "Boundary",
        "source_subtype": "semantic",
        "affected_steps": "",
        "confidence": "medium",
    }


def _subtype_options(primitive: str, current: str = "") -> list[str]:
    opts = subtypes_for_primitive(primitive)
    cur = (current or "").strip()
    if cur and cur not in opts:
        return [cur, *opts]
    return opts


def _tag_options(primitive: str, subtype: str, current: str = "") -> list[str]:
    opts = tags_for_primitive_subtype(primitive, subtype)
    cur = (current or "").strip()
    if cur and cur not in opts:
        return [cur, *opts]
    return opts


def _coerce_step_label_fields(
    primitive: str,
    subtype: str,
    tag: str,
    *,
    keep_invalid: bool = False,
) -> tuple[str, str, str]:
    prim = primitive if primitive in STEP_PRIMITIVES else STEP_PRIMITIVES[0]
    subtypes = subtypes_for_primitive(prim)
    sub = subtype
    if sub not in subtypes:
        sub = subtypes[0] if subtypes else ""
        if keep_invalid and subtype.strip():
            sub = subtype.strip()
    tags = tags_for_primitive_subtype(prim, sub)
    tg = tag
    if tags:
        if tg not in tags:
            tg = tags[0]
            if keep_invalid and tag.strip():
                tg = tag.strip()
    else:
        tg = ""
    return prim, sub, tg


def _init_step_rows_from_doc(step_labels: list[Any]) -> list[dict[str, Any]]:
    if not step_labels:
        return []
    out: list[dict[str, Any]] = []
    for raw in step_labels:
        if not isinstance(raw, dict):
            continue
        row = _normalize_loaded_step_label(raw)
        prim = str(row.get("primitive") or "Boundary")
        sub = str(row.get("subtype") or "")
        tag = str(row.get("tag") or "")
        status = str(row.get("label_status") or "")
        keep_invalid = status == "unmapped"
        prim, sub, tag = _coerce_step_label_fields(prim, sub, tag, keep_invalid=keep_invalid)
        if not keep_invalid:
            status = ""
        out.append(
            {
                "step": int(row.get("step") or 0),
                "primitive": prim,
                "subtype": sub,
                "tag": tag,
                "explanation": str(row.get("explanation") or ""),
                "confidence": row.get("confidence") or "medium",
                "is_root_cause": bool(row.get("is_root_cause", False)),
                "is_secondary": bool(row.get("is_secondary", False)),
                "label_status": status,
            }
        )
    return out


def _init_prop_rows_from_doc(prop_events: list[Any]) -> list[dict[str, Any]]:
    if not prop_events:
        return [_empty_propagation_row()]
    out: list[dict[str, Any]] = []
    for raw in prop_events:
        if not isinstance(raw, dict):
            continue
        prim = str(raw.get("source_primitive") or "Boundary")
        sub = str(raw.get("source_subtype") or "")
        prim, sub, _ = _coerce_step_label_fields(prim, sub, "", keep_invalid=False)
        aff = raw.get("affected_steps") or []
        out.append(
            {
                "source_step": int(raw.get("source_step") or 0),
                "source_primitive": prim,
                "source_subtype": sub,
                "affected_steps": ", ".join(str(x) for x in aff),
                "confidence": raw.get("confidence") or "medium",
            }
        )
    return out or [_empty_propagation_row()]


def _sync_session_rows(
    session_key: str,
    sig_key: str,
    signature: str,
    init_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if st.session_state.get(sig_key) != signature:
        st.session_state[session_key] = deepcopy(init_rows)
        st.session_state[sig_key] = signature
    sig_hash = str(abs(hash(signature)))
    return st.session_state[session_key], sig_hash


def _bump_editor_rev(rev_key: str) -> None:
    st.session_state[rev_key] = int(st.session_state.get(rev_key, 0)) + 1


def _reset_dependent_selectbox(
    track_key: str,
    current_value: str,
    widget_key: str,
    options: list[str],
) -> None:
    prev = st.session_state.get(track_key)
    if prev is not None and prev != current_value:
        st.session_state[widget_key] = options[0] if options else ""
    st.session_state[track_key] = current_value


def _render_step_labels_editor(
    episode_id: str,
    doc_rows: list[Any],
    *,
    max_step: int,
) -> list[dict[str, Any]]:
    sig = json.dumps(doc_rows, sort_keys=True, default=str)
    rows_key = f"sl::{episode_id}::rows"
    sig_key = f"sl::{episode_id}::sig"
    rev_key = f"sl::{episode_id}::rev"
    if st.session_state.get(sig_key) != sig:
        st.session_state[rev_key] = 0
    rows, sig_hash = _sync_session_rows(rows_key, sig_key, sig, _init_step_rows_from_doc(doc_rows))
    rev = int(st.session_state.get(rev_key, 0))
    ns = f"sl::{episode_id}::{sig_hash}::{rev}"

    unmapped_count = sum(1 for r in rows if str(r.get("label_status") or "") == "unmapped")
    if unmapped_count:
        st.warning(
            f"{unmapped_count} step label(s) use legacy or invalid ontology values (marked unmapped). "
            "Relabel them before saving."
        )

    st.caption(
        "Subtype and tag options update based on the selected primitive. "
        "Tags appear only when the subtype has child tags."
    )

    header = st.columns([1, 1.2, 1.4, 1.6, 1.8, 0.8, 0.7, 1, 0.4])
    header[0].markdown("**step**")
    header[1].markdown("**primitive**")
    header[2].markdown("**subtype**")
    header[3].markdown("**tag**")
    header[4].markdown("**explanation**")
    header[5].markdown("**Root cause**", help=ROOT_CAUSE_TOOLTIP)
    header[6].markdown("**Secondary**", help=SECONDARY_TOOLTIP)
    header[7].markdown("**conf.**")
    header[8].markdown("**del**")

    if not rows:
        st.caption("No step labels yet. Click **Add step label** to begin.")

    updated: list[dict[str, Any]] = []
    delete_idx: int | None = None

    for i, seed in enumerate(rows):
        cols = st.columns([1, 1.2, 1.4, 1.6, 1.8, 0.8, 0.7, 1, 0.4])
        p_key = f"{ns}::{i}::p"
        s_key = f"{ns}::{i}::s"
        t_key = f"{ns}::{i}::t"
        p_track = f"{ns}::{i}::_p"
        s_track = f"{ns}::{i}::_s"
        step_key = f"{ns}::{i}::step"
        expl_key = f"{ns}::{i}::expl"
        conf_key = f"{ns}::{i}::conf"
        rc_key = f"{ns}::{i}::rc"
        sec_key = f"{ns}::{i}::sec"

        if p_key not in st.session_state:
            st.session_state[p_key] = seed.get("primitive", "Boundary")
        if s_key not in st.session_state:
            st.session_state[s_key] = seed.get("subtype", "")
        if t_key not in st.session_state:
            st.session_state[t_key] = seed.get("tag", "")
        if step_key not in st.session_state:
            st.session_state[step_key] = int(seed.get("step") or 0)
        if expl_key not in st.session_state:
            st.session_state[expl_key] = str(seed.get("explanation") or "")
        if conf_key not in st.session_state:
            st.session_state[conf_key] = seed.get("confidence") or "medium"
        if rc_key not in st.session_state:
            st.session_state[rc_key] = "Yes" if seed.get("is_root_cause") else "No"
        if sec_key not in st.session_state:
            st.session_state[sec_key] = bool(seed.get("is_secondary", False))

        prim = cols[1].selectbox(
            "primitive",
            STEP_PRIMITIVES,
            index=STEP_PRIMITIVES.index(st.session_state[p_key])
            if st.session_state[p_key] in STEP_PRIMITIVES
            else 0,
            key=p_key,
            label_visibility="collapsed",
        )
        _reset_dependent_selectbox(p_track, prim, s_key, _subtype_options(prim))

        subtype_opts = _subtype_options(prim, str(st.session_state.get(s_key) or ""))
        if st.session_state[s_key] not in subtype_opts:
            st.session_state[s_key] = subtype_opts[0] if subtype_opts else ""
        subtype = cols[2].selectbox(
            "subtype",
            subtype_opts,
            index=subtype_opts.index(st.session_state[s_key]) if st.session_state[s_key] in subtype_opts else 0,
            key=s_key,
            label_visibility="collapsed",
        )
        _reset_dependent_selectbox(s_track, subtype, t_key, _tag_options(prim, subtype))

        tag = ""
        tag_opts = _tag_options(prim, subtype, str(st.session_state.get(t_key) or ""))
        if tag_opts:
            if st.session_state[t_key] not in tag_opts:
                st.session_state[t_key] = tag_opts[0]
            tag = cols[3].selectbox(
                "tag",
                tag_opts,
                index=tag_opts.index(st.session_state[t_key]) if st.session_state[t_key] in tag_opts else 0,
                key=t_key,
                label_visibility="collapsed",
            )
        else:
            cols[3].caption("—")

        step = cols[0].number_input(
            "step",
            min_value=0,
            max_value=max(max_step, 0) + 500,
            step=1,
            key=step_key,
            label_visibility="collapsed",
        )
        explanation = cols[4].text_input(
            "explanation",
            key=expl_key,
            label_visibility="collapsed",
        )
        rc_choice = cols[5].selectbox(
            "Root cause",
            ROOT_CAUSE_OPTIONS,
            index=ROOT_CAUSE_OPTIONS.index(st.session_state[rc_key])
            if st.session_state[rc_key] in ROOT_CAUSE_OPTIONS
            else 0,
            key=rc_key,
            label_visibility="collapsed",
        )
        is_secondary = cols[6].checkbox(
            "Secondary",
            key=sec_key,
            label_visibility="collapsed",
        )
        conf = cols[7].selectbox(
            "confidence",
            CONFIDENCE_LEVELS,
            index=CONFIDENCE_LEVELS.index(st.session_state[conf_key])
            if st.session_state[conf_key] in CONFIDENCE_LEVELS
            else 1,
            key=conf_key,
            label_visibility="collapsed",
        )
        if cols[8].button("✕", key=f"{ns}::{i}::del"):
            delete_idx = i

        status = classify_step_label(primitive=prim, subtype=subtype, tag=tag)
        label_status = "unmapped" if status == "unmapped" else ""
        if label_status:
            st.caption(f"Row {i + 1}: unmapped — choose valid primitive/subtype/tag before saving.")

        updated.append(
            {
                "step": int(step),
                "primitive": prim,
                "subtype": subtype,
                "tag": tag,
                "explanation": explanation,
                "confidence": conf,
                "is_root_cause": rc_choice == "Yes",
                "is_secondary": is_secondary,
                "label_status": label_status,
            }
        )

    btn_add, _ = st.columns([1, 5])
    if btn_add.button("Add step label", key=f"sl::{episode_id}::add"):
        prim, sub, tag = _coerce_step_label_fields("Boundary", "semantic", "goal_misinterpretation")
        updated.append(
            {
                "step": 0,
                "primitive": prim,
                "subtype": sub,
                "tag": tag,
                "explanation": "",
                "confidence": "medium",
                "is_root_cause": False,
                "is_secondary": False,
                "label_status": "",
            }
        )
        _bump_editor_rev(rev_key)
        st.session_state[rows_key] = updated
        st.rerun()

    if delete_idx is not None:
        updated.pop(delete_idx)
        _bump_editor_rev(rev_key)
        st.session_state[rows_key] = updated
        st.rerun()

    st.session_state[rows_key] = updated
    return updated


def _render_propagation_editor(episode_id: str, doc_rows: list[Any], *, max_step: int) -> list[dict[str, Any]]:
    sig = json.dumps(doc_rows, sort_keys=True, default=str)
    rows_key = f"prop::{episode_id}::rows"
    sig_key = f"prop::{episode_id}::sig"
    rows, sig_hash = _sync_session_rows(rows_key, sig_key, sig, _init_prop_rows_from_doc(doc_rows))
    ns = f"prop::{episode_id}::{sig_hash}"

    st.caption(
        f"`source_primitive` is one of {', '.join(PROPAGATION_SOURCE_PRIMITIVES)}. "
        "Subtype options update based on the selected source primitive. "
        f"{TRAJECTORY_LABEL_DEFINITIONS['propagation']}"
    )

    header = st.columns([1, 1.2, 1.4, 2, 1, 0.4])
    header[0].markdown("**source step**")
    header[1].markdown("**source primitive**")
    header[2].markdown("**source subtype**")
    header[3].markdown("**affected steps**")
    header[4].markdown("**conf.**")
    header[5].markdown("**del**")

    updated: list[dict[str, Any]] = []
    delete_idx: int | None = None

    for i, seed in enumerate(rows):
        cols = st.columns([1, 1.2, 1.4, 2, 1, 0.4])
        p_key = f"{ns}::{i}::p"
        s_key = f"{ns}::{i}::s"
        p_track = f"{ns}::{i}::_p"

        if p_key not in st.session_state:
            st.session_state[p_key] = seed.get("source_primitive", "Boundary")
        if s_key not in st.session_state:
            st.session_state[s_key] = seed.get("source_subtype", "")

        prim = cols[1].selectbox(
            "source_primitive",
            PROPAGATION_SOURCE_PRIMITIVES,
            index=PROPAGATION_SOURCE_PRIMITIVES.index(st.session_state[p_key])
            if st.session_state[p_key] in PROPAGATION_SOURCE_PRIMITIVES
            else 0,
            key=p_key,
            label_visibility="collapsed",
        )
        _reset_dependent_selectbox(p_track, prim, s_key, subtypes_for_primitive(prim))

        subtype_opts = _subtype_options(prim, str(st.session_state.get(s_key) or ""))
        if st.session_state[s_key] not in subtype_opts:
            st.session_state[s_key] = subtype_opts[0] if subtype_opts else ""
        subtype = cols[2].selectbox(
            "source_subtype",
            subtype_opts,
            index=subtype_opts.index(st.session_state[s_key]) if st.session_state[s_key] in subtype_opts else 0,
            key=s_key,
            label_visibility="collapsed",
        )

        source_step = cols[0].number_input(
            "source_step",
            min_value=0,
            max_value=max(max_step, 0) + 500,
            value=int(seed.get("source_step") or 0),
            step=1,
            key=f"{ns}::{i}::step",
            label_visibility="collapsed",
        )
        affected = cols[3].text_input(
            "affected_steps",
            value=str(seed.get("affected_steps") or ""),
            key=f"{ns}::{i}::aff",
            label_visibility="collapsed",
        )
        conf = cols[4].selectbox(
            "confidence",
            CONFIDENCE_LEVELS,
            index=CONFIDENCE_LEVELS.index(seed.get("confidence", "medium"))
            if seed.get("confidence") in CONFIDENCE_LEVELS
            else 1,
            key=f"{ns}::{i}::conf",
            label_visibility="collapsed",
        )
        if cols[5].button("✕", key=f"{ns}::{i}::del"):
            delete_idx = i

        updated.append(
            {
                "source_step": int(source_step),
                "source_primitive": prim,
                "source_subtype": subtype,
                "affected_steps": affected,
                "confidence": conf,
            }
        )

    btn_add, _ = st.columns([1, 5])
    if btn_add.button("Add propagation event", key=f"prop::{episode_id}::add"):
        prim, sub, _ = _coerce_step_label_fields("Boundary", "semantic", "")
        updated.append(
            {
                "source_step": 0,
                "source_primitive": prim,
                "source_subtype": sub,
                "affected_steps": "",
                "confidence": "medium",
            }
        )
        st.session_state[rows_key] = updated
        st.rerun()

    if delete_idx is not None:
        updated.pop(delete_idx)
        if not updated:
            updated = [_empty_propagation_row()]
        st.session_state[rows_key] = updated
        st.rerun()

    st.session_state[rows_key] = updated
    return updated


def _ordered_step_indices(events: list[dict[str, Any]]) -> list[int]:
    """One label per event row, in order (uses step_index when present, else list position)."""
    out: list[int] = []
    for i, ev in enumerate(events):
        si = ev.get("step_index")
        if si is None:
            out.append(i)
            continue
        try:
            out.append(int(si))
        except (TypeError, ValueError):
            out.append(i)
    return out


def _event_for_focus_step(events: list[dict[str, Any]], step_idx: int) -> dict[str, Any]:
    """Resolve the event row whose step_index matches step_idx (fallback: list index)."""
    for ev in events:
        try:
            if int(ev.get("step_index", -10**9)) == step_idx:
                return ev
        except (TypeError, ValueError):
            continue
    if 0 <= step_idx < len(events):
        return events[step_idx]
    return {}


def _list_index_for_focus_step(events: list[dict[str, Any]], step_idx: int) -> int | None:
    """List index for neighbors / JSON blocks when step_index matches step_idx."""
    for j, ev in enumerate(events):
        try:
            if int(ev.get("step_index", -10**9)) == step_idx:
                return j
        except (TypeError, ValueError):
            continue
    if 0 <= step_idx < len(events):
        return step_idx
    return None


def default_label_document(episode_id: str) -> dict[str, Any]:
    return {
        "annotation_key": episode_id,
        "schema_version": SCHEMA_VERSION,
        "trace_id": episode_id,
        "episode_id": episode_id,
        "is_failure_trace": False,
        "step_labels": [],
        "propagation_events": [],
        "looping": {"present": False},
        "external_system": {"present": False},
        "notes": "",
    }


def _strip_optional_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return s if s else None


def _parse_int_list_field(raw: Any, field_name: str) -> tuple[list[int] | None, str | None]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return [], None
    s = str(raw).strip()
    if not s:
        return [], None
    out: list[int] = []
    for part in re.split(r"[,;\s]+", s):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            return None, f"{field_name}: invalid integer token {part!r}"
    return out, None


def step_labels_from_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    rows_out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        try:
            step = int(row.get("step", 0))
        except (TypeError, ValueError):
            errors.append(f"Row {i}: invalid step")
            continue
        prim = _strip_optional_str(row.get("primitive"))
        if not prim:
            continue
        subtype = _strip_optional_str(row.get("subtype")) or ""
        tag = _strip_optional_str(row.get("tag")) or ""
        row_errors = validate_step_label(primitive=prim, subtype=subtype, tag=tag)
        for msg in row_errors:
            errors.append(f"Row {i}: {msg}")
        if row_errors:
            continue
        conf = _strip_optional_str(row.get("confidence")) or "medium"
        if conf not in CONFIDENCE_LEVELS:
            errors.append(f"Row {i}: confidence must be one of {CONFIDENCE_LEVELS}")
            continue
        rec: dict[str, Any] = {
            "step": step,
            "primitive": prim,
            "subtype": subtype,
            "confidence": conf,
            "is_root_cause": bool(row.get("is_root_cause", False)),
            "is_secondary": bool(row.get("is_secondary", False)),
        }
        if tag:
            rec["tag"] = tag
        expl = _strip_optional_str(row.get("explanation"))
        if expl:
            rec["explanation"] = expl
        rows_out.append(rec)
    return rows_out, errors


def validate_failure_root_causes(
    step_labels: list[dict[str, Any]],
    *,
    is_failure: bool,
) -> list[str]:
    if not is_failure:
        return []
    if not any(bool(sl.get("is_root_cause")) for sl in step_labels):
        return ["Failure traces require at least one step label with Root cause = Yes."]
    return []


def step_labels_from_dataframe(df: pd.DataFrame) -> tuple[list[dict[str, Any]] | None, list[str]]:
    if df is None or df.empty:
        return [], []
    rows = df.to_dict(orient="records")
    return step_labels_from_rows(rows)


def propagation_from_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        try:
            source_step = int(row.get("source_step", 0))
        except (TypeError, ValueError):
            errors.append(f"Propagation row {i}: invalid source_step")
            continue
        sp = _strip_optional_str(row.get("source_primitive"))
        if not sp:
            continue
        if sp not in PROPAGATION_SOURCE_PRIMITIVES:
            errors.append(f"Propagation row {i}: source_primitive must be one of {PROPAGATION_SOURCE_PRIMITIVES}")
            continue
        sst = _strip_optional_str(row.get("source_subtype")) or ""
        if sst and sst not in subtypes_for_primitive(sp):
            errors.append(
                f"Propagation row {i}: source_subtype {sst!r} not valid for {sp!r}; "
                f"expected one of: {', '.join(subtypes_for_primitive(sp))}"
            )
            continue
        aff_raw = row.get("affected_steps")
        affected, err = _parse_int_list_field(aff_raw, "affected_steps")
        if err:
            errors.append(f"Propagation row {i}: {err}")
            continue
        if not affected:
            continue
        conf = _strip_optional_str(row.get("confidence")) or "medium"
        if conf not in CONFIDENCE_LEVELS:
            errors.append(f"Propagation row {i}: invalid confidence")
            continue
        out.append(
            {
                "source_step": source_step,
                "source_primitive": sp,
                "source_subtype": sst,
                "affected_steps": affected,
                "confidence": conf,
            }
        )
    return out, errors


def _file_version(path: str) -> float:
    """Modification time, or 0 if missing. Used so caches refresh when a file changes."""
    try:
        return Path(path).expanduser().stat().st_mtime
    except (OSError, ValueError):
        return 0.0


@st.cache_resource
def load_normalized(path: str, version: float = 0.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path or not path.strip():
        return rows
    p = Path(path).expanduser()
    if not p.is_file():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_annotations(path: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = row.get("annotation_key") or row.get("episode_id") or row.get("trace_id")
            if key is not None:
                out[str(key)] = row
    return out


def save_annotations(path: str, annotation_rows: dict[str, dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in annotation_rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(p)


def _normalize_loaded_step_label(sl: dict[str, Any]) -> dict[str, Any]:
    """Map legacy fields and mark labels that no longer exist in the ontology."""
    out = deepcopy(sl)
    tag = out.get("tag")
    if not tag and out.get("observation_tag"):
        out["tag"] = out.pop("observation_tag")
    out.pop("observation_tag", None)
    out.pop("observation_tag_custom", None)
    status = classify_step_label(
        primitive=str(out.get("primitive") or ""),
        subtype=str(out.get("subtype") or ""),
        tag=str(out.get("tag") or ""),
    )
    if status == "unmapped":
        out["label_status"] = "unmapped"
    else:
        out.pop("label_status", None)
    return out


def _coerce_loaded_document(row: dict[str, Any], episode_id: str) -> dict[str, Any]:
    """Merge a JSONL row into the current label document shape."""
    base = default_label_document(episode_id)
    merge_keys = (
        "step_labels",
        "propagation_events",
        "looping",
        "external_system",
        "notes",
        "is_failure_trace",
        "trace_id",
    )
    if row.get("step_labels") is not None:
        for k in merge_keys:
            if k in row:
                base[k] = deepcopy(row[k])
        base["trace_id"] = row.get("trace_id") or episode_id
        base["schema_version"] = SCHEMA_VERSION
        base["step_labels"] = [_normalize_loaded_step_label(sl) for sl in base.get("step_labels") or [] if isinstance(sl, dict)]
        return base
    return default_label_document(episode_id)


@st.cache_resource
def _load_result_lines(log_path: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    p = Path(log_path)
    if not p.exists():
        return mapping
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            low = line.lower()
            if "[result]" not in low:
                continue
            m = re.search(r"(?:config_files/|/)(\d+)\.json\b", low)
            if m:
                idx = m.group(1)
                mapping[idx] = line
                mapping[f"render_{idx}"] = line
    return mapping


def _coerce_answer_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for v in value:
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() != "none":
                out.append(s)
        return out
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip() and p.strip().lower() != "none"]
    s = str(value).strip()
    return [s] if s and s.lower() != "none" else []


def _crmarena_gt_pred_diff(md: dict[str, Any]) -> dict[str, Any] | None:
    gt_list = _coerce_answer_tokens(md.get("ground_truth_answer"))
    agent_info = md.get("agent_info") or {}
    end_reason = agent_info.get("end_reason") or {}
    pred_list: list[str] = []
    pred_source = "none"
    parsed = end_reason.get("parsed_answer")
    if isinstance(parsed, list) and parsed:
        pred_list = _coerce_answer_tokens(parsed)
        pred_source = "parsed_answer"
    elif isinstance(parsed, str) and parsed.strip():
        pred_list = _coerce_answer_tokens(parsed)
        pred_source = "parsed_answer"
    if not pred_list:
        content = end_reason.get("content")
        if isinstance(content, str) and content.strip():
            pred_list = _coerce_answer_tokens(content)
            pred_source = "end_reason.content"
    if not gt_list and not pred_list:
        return None

    gt_set = set(gt_list)
    pred_set = set(pred_list)
    missing = sorted(gt_set - pred_set)
    extra = sorted(pred_set - gt_set)
    both = sorted(gt_set & pred_set)
    parts = []
    if missing:
        parts.append(f"{len(missing)} GT token(s) missing from prediction")
    if extra:
        parts.append(f"{len(extra)} prediction token(s) not in GT")
    if not missing and not extra and gt_list and pred_list:
        if gt_list == pred_list:
            parts.append("GT and prediction lists are identical (set view)")
        else:
            parts.append("same unique tokens; list length or ordering differs between GT and prediction")
    summary = "; ".join(parts) if parts else "empty GT and prediction lists"
    return {
        "prediction_source": pred_source,
        "missing_from_prediction": missing,
        "extra_in_prediction": extra,
        "in_both": both,
        "counts": {
            "gt_list": len(gt_list),
            "pred_list": len(pred_list),
            "gt_unique": len(gt_set),
            "pred_unique": len(pred_set),
        },
        "summary": summary,
    }


def _truncate_jsonish(value: Any, max_chars: int = 2000) -> str:
    raw = json.dumps(value, ensure_ascii=False, default=str)
    if len(raw) > max_chars:
        return raw[:max_chars] + " …"
    return raw


def _tau_bench_reward_action_rows(action_checks: list[Any] | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(action_checks or []):
        if not isinstance(raw, dict):
            continue
        act = raw.get("action")
        names = ""
        args_s = ""
        if isinstance(act, dict):
            names = str(act.get("name", "") or "")
            args_s = (
                json.dumps(act.get("arguments"), ensure_ascii=False, sort_keys=True)
                if act.get("arguments") is not None
                else ""
            )
        rows.append(
            {
                "#": idx,
                "tool": names,
                "arguments": args_s,
                "match": raw.get("action_match"),
                "reward": raw.get("action_reward"),
                "tool_type": raw.get("tool_type"),
            }
        )
    return pd.DataFrame(rows)


def _tau_bench_check_rows(rows: list[Any] | None, flatten_key: str = "payload") -> pd.DataFrame | None:
    if not isinstance(rows, list) or not rows:
        return None
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(rows):
        if not isinstance(raw, dict):
            continue
        flat = {"#": idx}
        extras: dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, (dict, list)):
                extras[k] = v
            else:
                flat[k] = v
        if extras:
            flat[flatten_key] = _truncate_jsonish(extras, max_chars=4000)
        out.append(flat)
    return pd.DataFrame(out) if out else None


def _render_taubench_outcome_panel(trace: dict[str, Any], evidence: dict[str, Any]) -> None:
    md = trace.get("metadata") or {}

    infra = md.get("simulation_info_summary")
    if isinstance(infra, dict) and infra.get("error"):
        snippet = str(infra.get("error_type") or "infrastructure_error")
        body = str(infra.get("error") or "")[:2400]
        st.warning(f"**Simulation did not grade normally** ({snippet}): {body}")
        if infra.get("failed_after_attempts") is not None:
            st.caption(f"Retries: {infra.get('failed_after_attempts')}")

    crit = md.get("evaluation_criteria") or evidence.get("task_evaluation_criteria")
    cols = st.columns(5)
    cols[0].metric("Tau reward", f"{evidence.get('reward', '')!s}")
    cols[1].metric("Termination", str(evidence.get("termination_reason") or "—"))
    aa = evidence.get("action_summary")
    if isinstance(aa, dict) and aa.get("total"):
        cols[2].metric("Action checks OK", f"{aa.get('correct', 0)}/{aa['total']}")
    else:
        cols[2].metric("Action checks OK", "—")

    env_s = evidence.get("env_summary")
    if isinstance(env_s, dict) and env_s.get("total"):
        cols[3].metric("Env assertions", f"{env_s.get('passed', 0)}/{env_s['total']}")
    else:
        cols[3].metric("Env assertions", "—")

    comm_s = evidence.get("communicate_summary")
    if isinstance(comm_s, dict) and comm_s.get("total"):
        cols[4].metric("Communicate", f"{comm_s.get('passed', 0)}/{comm_s['total']}")
    else:
        cols[4].metric("Communicate", "—")

    rb = evidence.get("reward_basis")
    bd = evidence.get("reward_breakdown")
    if rb or bd:
        rb_disp = ", ".join(str(x) for x in rb) if isinstance(rb, list) else repr(rb)
        st.caption(f"Reward basis: {rb_disp} · Breakdown: {bd}")

    gtabs = st.tabs(["Task rubric (expected)", "Graded checks (this run)", "DB check"])
    with gtabs[0]:
        if not isinstance(crit, dict):
            st.caption("No `evaluation_criteria` on this trace—re-normalize with an up-to-date adapter to embed task rubrics from Tau `results.json` / `tasks.json`.")
        else:
            nls = crit.get("nl_assertions") or []
            if isinstance(nls, list) and nls:
                st.markdown("**NL assertions (expected wording / behavior)**")
                for line in nls:
                    st.markdown(f"- {line}")

            ci = crit.get("communicate_info") or []
            if isinstance(ci, list) and ci:
                st.markdown("**Communicate hints (gold)**")
                df_c = _tau_bench_check_rows(ci, flatten_key="extra")
                if df_c is not None:
                    st.dataframe(df_c, use_container_width=True)

            acts = crit.get("actions") or []
            if isinstance(acts, list) and acts:
                st.markdown("**Expected assistant actions**")
                gold_rows = []
                for i, act in enumerate(acts):
                    if not isinstance(act, dict):
                        continue
                    gold_rows.append(
                        {
                            "#": i,
                            "name": act.get("name"),
                            "arguments": json.dumps(act.get("arguments"), ensure_ascii=False, sort_keys=True)
                            if act.get("arguments") is not None
                            else "",
                            "action_id": act.get("action_id"),
                        }
                    )
                if gold_rows:
                    st.dataframe(pd.DataFrame(gold_rows), use_container_width=True)

            env_gold = crit.get("env_assertions")
            if isinstance(env_gold, list) and env_gold:
                st.markdown("**Expected env assertions (spec)**")
                df_e = _tau_bench_check_rows(env_gold, flatten_key="spec")
                if df_e is not None:
                    st.dataframe(df_e, use_container_width=True)

            rbb = crit.get("reward_basis")
            if isinstance(rbb, list) and rbb:
                st.caption(f"Task `reward_basis`: {', '.join(str(x) for x in rbb)}")

    with gtabs[1]:
        ac = evidence.get("reward_action_checks")
        df_a = _tau_bench_reward_action_rows(ac if isinstance(ac, list) else None)
        if df_a.empty:
            st.caption("No graded action_checks in reward_info.")
        else:
            st.markdown("**Action alignment** (`reward_info.action_checks`)")
            st.dataframe(df_a, use_container_width=True)

        for title, ev_key in [
            ("Env assertions (`reward_info.env_assertions`)", "reward_env_assertions"),
            ("Communicate checks", "reward_communicate_checks"),
            ("NL checks (reward_info)", "reward_nl_assertions"),
        ]:
            lst = evidence.get(ev_key)
            df_x = _tau_bench_check_rows(lst if isinstance(lst, list) else None)
            if df_x is None:
                continue
            st.markdown(f"**{title}**")
            st.dataframe(df_x, use_container_width=True)

        ri_notes = md.get("reward_component_info")
        if isinstance(ri_notes, dict) and any(ri_notes.values()):
            st.markdown("**Grader notes (`reward_info.info`)**")
            st.json(ri_notes, expanded=False)

    with gtabs[2]:
        dbc = evidence.get("reward_db_check")
        if isinstance(dbc, dict) and dbc:
            st.json(dbc, expanded=True)
        else:
            st.caption("No `db_check` in reward_info.")

    with st.expander("Raw Tau outcome blob (merged JSON)", expanded=False):
        st.json(evidence, expanded=False)


def _resolve_intercode_outcome_evidence(
    trace: dict[str, Any], intercode_root: Path | None = None,
) -> dict[str, Any]:
    md = trace.get("metadata") or {}
    root = intercode_root or intercode_root_from_env()
    if root is None and isinstance(md.get("intercode_root"), str) and md.get("intercode_root"):
        root = Path(md["intercode_root"])

    gold_sql = md.get("gold_sql") or md.get("ground_truth_sql")
    task_db = md.get("task_db")
    task_file = md.get("task_file")
    if not gold_sql and root is not None:
        task_row = lookup_intercode_task(md.get("dataset_path"), trace.get("task_id"), root)
        if task_row:
            gold_sql = task_row.get("gold")
            task_db = task_db or task_row.get("db")
            task_file = task_file or task_row.get("_task_file")

    best_turn = md.get("best_turn") if isinstance(md.get("best_turn"), dict) else {}
    eval_info = md.get("eval_info") if isinstance(md.get("eval_info"), dict) else {}
    expected_rows = md.get("expected_eval_obs") or eval_info.get("eval_obs")
    agent_rows = md.get("agent_obs_at_submit") or eval_info.get("agent_obs")
    if not agent_rows and best_turn.get("observation") is not None:
        agent_rows = best_turn.get("observation")

    max_reward = md.get("max_reward")
    if max_reward is None:
        summary = md.get("summary") if isinstance(md.get("summary"), dict) else {}
        max_reward = summary.get("max_reward")

    return {
        "outcome_source": "intercode_sql_reward",
        "grading_hint": (
            "InterCode SQL success when max per-turn reward is 1.0 after implicit `submit`. "
            "Reward is intersection-over-union between the agent's last submitted query result "
            "and rows from executing the gold SQL (`eval_obs`). Gold SQL is in the task file, not the trajectory log."
        ),
        "is_success_by_reward": isinstance(max_reward, (int, float)) and float(max_reward) >= 1.0 - 1e-9,
        "max_reward": max_reward,
        "max_reward_turn_index": md.get("max_reward_idx"),
        "gold_sql": gold_sql,
        "task_db": task_db,
        "task_file": task_file,
        "dataset_path": md.get("dataset_path"),
        "intercode_root": str(root) if root else None,
        "gold_sql_resolved": gold_sql is not None,
        "expected_eval_obs": expected_rows,
        "agent_obs_at_best_turn": agent_rows,
        "best_turn": best_turn,
        "eval_info": eval_info,
        "hardness": md.get("hardness"),
    }


def _render_intercode_outcome_panel(
    trace: dict[str, Any],
    evidence: dict[str, Any],
    intercode_root: Path | None = None,
) -> None:
    cols = st.columns(4)
    cols[0].metric("Max reward", f"{evidence.get('max_reward', '')!s}")
    cols[1].metric("Success", "yes" if evidence.get("is_success_by_reward") else "no")
    cols[2].metric("Best turn", str(evidence.get("max_reward_turn_index", "—")))
    cols[3].metric("DB", str(evidence.get("task_db") or "—"))

    if not evidence.get("gold_sql_resolved"):
        st.warning(
            "Gold SQL not loaded. Re-run normalization with `--intercode-root /path/to/intercode`, "
            "set `TRACEMARK_INTERCODE_ROOT`, or enter the InterCode repo path in the sidebar."
        )
    else:
        st.markdown("**Gold SQL (expected query)**")
        st.code(str(evidence.get("gold_sql") or ""), language="sql")

    st.markdown("**Natural-language task**")
    st.write(trace.get("goal") or "—")

    bt = evidence.get("best_turn") if isinstance(evidence.get("best_turn"), dict) else {}
    if bt.get("action"):
        st.markdown(f"**Agent SQL at best-scoring turn (index {bt.get('turn_index')})**")
        st.code(str(bt.get("action")), language="sql")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Agent result rows (at submit / best turn)**")
        st.text(str(evidence.get("agent_obs_at_best_turn") or "—"))
    with c2:
        st.markdown("**Expected result rows (`eval_obs` from gold SQL)**")
        exp = evidence.get("expected_eval_obs")
        exp_err = (trace.get("metadata") or {}).get("expected_eval_obs_error")
        if exp:
            st.text(str(exp))
        elif exp_err:
            st.warning(str(exp_err))
        else:
            st.caption(
                "Not in the trajectory log. Run gold SQL against InterCode MySQL — see sidebar button "
                "or `python scripts/enrich_intercode_eval_obs.py`."
            )

    live_key = f"intercode_live_eval::{trace.get('episode_id')}"
    if st.button("Run gold SQL in MySQL (this episode)", key=f"ic_mysql::{trace.get('episode_id')}"):
        ic_root_btn = intercode_root if intercode_root and intercode_root.is_dir() else intercode_root_from_env()
        ok, msg = check_mysql_connection(MySQLConfig())
        if not ok:
            st.error(msg)
        else:
            db, gold, resolve_err = resolve_gold_for_trace(trace, ic_root_btn)
            if resolve_err or not db or not gold:
                st.error(resolve_err or "Missing db / gold SQL")
            else:
                try:
                    result = execute_gold_sql(db, gold)
                    st.session_state[live_key] = result
                except RuntimeError as exc:
                    st.error(str(exc))

    if live_key in st.session_state:
        st.markdown("**Live MySQL result (not saved to JSONL)**")
        st.text(str(st.session_state[live_key].get("eval_obs", "")))
        if st.session_state[live_key].get("error"):
            st.caption(f"Error: {st.session_state[live_key]['error']}")

    if evidence.get("task_file"):
        st.caption(f"Task file: `{evidence.get('task_file')}`")
    if evidence.get("dataset_path"):
        st.caption(f"Trajectory dataset ref: `{evidence.get('dataset_path')}`")


def _resolve_outcome_evidence(
    trace: dict[str, Any], intercode_root: Path | None = None,
) -> dict[str, Any]:
    md = trace.get("metadata") or {}
    if str(trace.get("dataset_name", "")).lower() in {"intercode_sql", "intercode"}:
        return _resolve_intercode_outcome_evidence(trace, intercode_root=intercode_root)

    if str(trace.get("dataset_name", "")).lower() == "crmarena":
        reward = md.get("reward")
        if isinstance(reward, dict):
            em = bool(reward.get("em", False))
            score_summary = {
                "em": em,
                "f1": reward.get("f1"),
                "bleu": reward.get("bleu"),
                "rouge": reward.get("rouge"),
            }
            success_flag = em
            grading_hint = "Success when reward dict has em=true."
        else:
            score_summary = {"reward_scalar": reward}
            success_flag = isinstance(reward, (int, float)) and abs(float(reward) - 1.0) <= 1e-6
            grading_hint = "Success when scalar reward is 1.0 (failure includes reward 0)."
        answer_diff = _crmarena_gt_pred_diff(md)
        return {
            "outcome_source": "crmarena_reward",
            "grading_hint": grading_hint,
            "is_success_by_reward": success_flag,
            "task_type": md.get("task_type"),
            "ground_truth_answer": md.get("ground_truth_answer"),
            "score_summary": score_summary,
            "gt_vs_prediction_diff": answer_diff,
            "agent_end_reason": (md.get("agent_info") or {}).get("end_reason"),
            "agent_turn_info": {
                "num_turns": (md.get("agent_info") or {}).get("num_turns"),
                "total_cost": (md.get("agent_info") or {}).get("total_cost"),
            },
        }

    if str(trace.get("dataset_name", "")).lower() == "taubench":
        reward = md.get("reward")
        reward_basis = md.get("reward_basis")
        breakdown = md.get("reward_breakdown")
        db_check = md.get("db_check")
        action_checks = md.get("action_checks")
        communicate_checks = md.get("communicate_checks")
        env_assertions = md.get("env_assertions")
        nl_assertions = md.get("nl_assertions")
        termination_reason = md.get("termination_reason")

        action_summary = None
        if isinstance(action_checks, list) and action_checks:
            total = len(action_checks)
            correct = sum(1 for c in action_checks if isinstance(c, dict) and c.get("action_match") is True)
            action_summary = {"correct": correct, "total": total, "accuracy": (correct / total) if total else None}

        env_summary = None
        if isinstance(env_assertions, list) and env_assertions:
            total = len(env_assertions)
            passed = sum(1 for c in env_assertions if isinstance(c, dict) and c.get("passed") is True)
            env_summary = {"passed": passed, "total": total, "rate": (passed / total) if total else None}

        comm_summary = None
        if isinstance(communicate_checks, list) and communicate_checks:
            total = len(communicate_checks)
            passed = sum(1 for c in communicate_checks if isinstance(c, dict) and c.get("passed") is True)
            comm_summary = {"passed": passed, "total": total, "rate": (passed / total) if total else None}

        nl_summary = None
        if isinstance(nl_assertions, list) and nl_assertions:
            total = len(nl_assertions)
            passed = sum(1 for c in nl_assertions if isinstance(c, dict) and c.get("passed") is True)
            nl_summary = {"passed": passed, "total": total, "rate": (passed / total) if total else None}

        grading_hint = (
            "Binary success when `reward_info.reward` ≈ 1.0 (partial breakdown in reward_breakdown). "
            "`evaluation_criteria` on the trace is the task rubric (same content as Tau `domains/*/tasks.json` "
            "when merged into results). For infra failures (`reward_info` null), termination_reason + "
            "simulation_info_summary explain why grading is missing."
        )

        return {
            "outcome_source": "taubench_reward_info",
            "grading_hint": grading_hint,
            "reward": reward,
            "task_evaluation_criteria": md.get("evaluation_criteria"),
            "simulation_failure": md.get("simulation_info_summary"),
            "is_success_by_reward": isinstance(reward, (int, float)) and abs(float(reward) - 1.0) <= 1e-6,
            "termination_reason": termination_reason,
            "reward_basis": reward_basis,
            "reward_breakdown": breakdown,
            "reward_db_check": db_check,
            "reward_action_checks": action_checks if isinstance(action_checks, list) else [],
            "reward_env_assertions": env_assertions if isinstance(env_assertions, list) else [],
            "reward_communicate_checks": communicate_checks if isinstance(communicate_checks, list) else [],
            "reward_nl_assertions": nl_assertions if isinstance(nl_assertions, list) else [],
            "action_summary": action_summary,
            "env_summary": env_summary,
            "communicate_summary": comm_summary,
            "nl_summary": nl_summary,
        }

    source_file = md.get("source_file")
    raw_render_id = md.get("raw_render_episode_id")
    outcome_source = md.get("outcome_source", "unknown")
    if not source_file:
        return {"outcome_source": outcome_source, "evidence": None}

    parent = Path(source_file).parent
    log_candidates = [parent / "merged_log.txt", parent / "merge_log.txt"]
    log_path = next((p for p in log_candidates if p.exists()), None)
    if log_path is None:
        return {
            "outcome_source": outcome_source,
            "source_file": source_file,
            "log_path": None,
            "matched_result_line": None,
        }

    lines = _load_result_lines(str(log_path))
    idx = None
    if raw_render_id and raw_render_id.startswith("render_"):
        idx = raw_render_id.removeprefix("render_")
    matched_line = None
    if raw_render_id:
        matched_line = lines.get(raw_render_id)
    if matched_line is None and idx is not None:
        matched_line = lines.get(idx)

    return {
        "outcome_source": outcome_source,
        "source_file": source_file,
        "log_path": str(log_path),
        "raw_render_episode_id": raw_render_id,
        "matched_result_line": matched_line,
    }


def _document_for_episode(annotation_rows: dict[str, dict[str, Any]], episode_id: str) -> dict[str, Any]:
    row = annotation_rows.get(episode_id)
    if not row:
        return default_label_document(episode_id)
    return _coerce_loaded_document(row, episode_id)


def _build_looping_payload(present: bool, steps_raw: str, pattern: str, confidence: str) -> dict[str, Any]:
    out: dict[str, Any] = {"present": bool(present)}
    if not present:
        return out
    steps, _ = _parse_int_list_field(steps_raw, "steps")
    if steps:
        out["steps"] = steps
    pat = (pattern or "").strip()
    if pat:
        out["pattern"] = pat
    if confidence in CONFIDENCE_LEVELS:
        out["confidence"] = confidence
    return out


def _build_external_system_payload(
    present: bool, explanation: str, confidence: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {"present": bool(present)}
    if not present:
        return out
    expl = (explanation or "").strip()
    if expl:
        out["explanation"] = expl
    if confidence in CONFIDENCE_LEVELS:
        out["confidence"] = confidence
    return out


def main() -> None:
    st.set_page_config(page_title="tracemark", layout="wide")
    st.title("tracemark: failure primitive labeling")

    with st.sidebar:
        st.header("Data")
        discovered = _discover_runs("outputs")
        runs = _run_options(discovered)
        selected_run = st.selectbox(
            "Dataset / run",
            options=runs if runs else [""],
            index=0,
            format_func=lambda r: f"{r} (not converted yet)"
            if r in BUILTIN_DATASET_HELP and r not in discovered
            else r,
        )
        if selected_run == "examples":
            normalized_path = str(EXAMPLE_TRACES)
            annotations_path = "outputs/examples/annotations/labels_v0.jsonl"
        elif selected_run:
            normalized_path, annotations_path = _resolve_paths_for_run(selected_run)
        else:
            normalized_path, annotations_path = "", ""

        st.caption(
            "Runs are detected from `outputs/<run>/normalized.jsonl`. "
            "With none present, the bundled example traces load."
        )
        with st.expander("Advanced path override", expanded=not discovered):
            normalized_path = st.text_input("normalized.jsonl", normalized_path)
            annotations_path = st.text_input("Label store (JSONL)", annotations_path)

        st.divider()
        st.subheader("InterCode SQL")
        default_ic_root = str(intercode_root_from_env() or "")
        intercode_root_input = st.text_input(
            "InterCode repo root (for gold SQL lookup)",
            value=default_ic_root,
            help="Clone https://github.com/princeton-nlp/intercode — used when normalized traces lack gold_sql.",
        )
        intercode_root = Path(intercode_root_input).expanduser() if intercode_root_input.strip() else None
        if intercode_root_input.strip() and not intercode_root.is_dir():
            st.caption("Path not found; gold SQL lookup will be skipped.")

        st.divider()
        st.subheader("Reset labels")
        confirm_reset = st.checkbox("I understand this will delete the label JSONL for this path")
        if st.button("Reset label file", disabled=not confirm_reset):
            p = Path(annotations_path)
            if p.exists():
                p.unlink()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Label file reset.")
            st.rerun()

    traces = load_normalized(normalized_path, _file_version(normalized_path))
    annotation_rows = load_annotations(annotations_path)

    if not traces and selected_run in BUILTIN_DATASET_HELP and not Path(normalized_path).is_file():
        st.info(
            f"No {selected_run} traces yet. Run these commands from the tracemark folder, "
            "then reload this page."
        )
        st.code(BUILTIN_DATASET_HELP[selected_run], language="bash")
        return

    if not traces:
        st.error(
            f"No traces found at `{normalized_path or '(empty path)'}`. "
            "Set the normalized.jsonl path in the sidebar under Advanced path override."
        )
        return

    trace_by_id = {t["episode_id"]: t for t in traces}
    outcome_counts = Counter(t.get("final_outcome", "unknown") for t in traces)

    st.subheader("Dataset snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Episodes", len(traces))
    c2.metric("Failures", outcome_counts.get("failure", 0))
    c3.metric("Successes", outcome_counts.get("success", 0))

    st.subheader("Episode filters")
    colf1, colf2 = st.columns(2)
    outcome_filter = colf1.multiselect("Outcome", options=sorted(outcome_counts.keys()), default=sorted(outcome_counts.keys()))
    min_events = colf2.number_input("Min events", min_value=1, value=1, step=1)

    filtered_ids: list[str] = []
    for t in traces:
        eid = t.get("episode_id")
        if not eid:
            continue
        if t.get("final_outcome", "unknown") not in outcome_filter:
            continue
        if len(t.get("events", [])) < int(min_events):
            continue
        filtered_ids.append(str(eid))

    st.write(f"Filtered episodes: **{len(filtered_ids)}**")
    if not filtered_ids:
        st.warning("No episodes match current filters.")
        return

    selected_episode = st.selectbox("Episode", filtered_ids, index=0)
    trace = trace_by_id[selected_episode]
    events = trace.get("events", [])
    max_step = max((int(ev.get("step_index", 0)) for ev in events), default=0)

    doc = _document_for_episode(annotation_rows, selected_episode)

    st.subheader("Episode metadata")
    st.json(
        {
            "episode_id": trace.get("episode_id"),
            "trace_id": doc.get("trace_id"),
            "dataset_name": trace.get("dataset_name"),
            "outputs_run": selected_run,
            "task_id": trace.get("task_id"),
            "outcome": trace.get("final_outcome"),
            "goal": trace.get("goal"),
            "event_count": len(events),
            "label_path": annotations_path,
            "metadata": trace.get("metadata", {}),
        },
        expanded=False,
    )

    st.subheader("Outcome evidence")
    ds = str(trace.get("dataset_name", "")).lower()
    outcome_evidence = _resolve_outcome_evidence(
        trace,
        intercode_root=intercode_root if ds in {"intercode_sql", "intercode"} else None,
    )
    gh = outcome_evidence.get("grading_hint")
    if isinstance(gh, str) and gh.strip():
        st.caption(gh)
    diff = outcome_evidence.get("gt_vs_prediction_diff")
    if isinstance(diff, dict) and diff.get("summary"):
        src = diff.get("prediction_source", "")
        st.info(f"**GT vs prediction:** {diff['summary']} _(parsed from: {src})_")
    if ds == "taubench":
        _render_taubench_outcome_panel(trace, outcome_evidence)
    elif ds in {"intercode_sql", "intercode"}:
        _render_intercode_outcome_panel(trace, outcome_evidence, intercode_root=intercode_root)
    else:
        st.json(outcome_evidence, expanded=False)

    st.divider()
    st.subheader("Labels")
    st.caption(
        "Step-level: Boundary, State, or Decision (primitive → subtype → optional tag). "
        "Trajectory-level: propagation, looping, and external_system are stored separately."
    )
    with st.expander("Primitive definitions", expanded=False):
        for prim in STEP_PRIMITIVES:
            st.markdown(f"**{prim}** — {PRIMITIVE_HELPER_TEXT[prim]}")
        st.markdown("**Trajectory labels**")
        for name, text in TRAJECTORY_LABEL_DEFINITIONS.items():
            st.markdown(f"- **{name}**: {text}")

    is_failure = st.checkbox("Failure trace (`is_failure_trace`)", value=bool(doc.get("is_failure_trace")), key=f"fail::{selected_episode}")
    trace_id_override = st.text_input(
        "`trace_id` (defaults to episode id)",
        value=str(doc.get("trace_id") or selected_episode),
        key=f"tid::{selected_episode}",
    )

    tab_steps, tab_prop, tab_loop, tab_ext, tab_preview = st.tabs(
        ["Step-level", "Propagation", "Looping", "External system", "JSON preview"]
    )

    with tab_steps:
        edited_step_rows = _render_step_labels_editor(
            selected_episode,
            doc.get("step_labels") or [],
            max_step=max_step,
        )

    with tab_prop:
        edited_prop_rows = _render_propagation_editor(
            selected_episode,
            doc.get("propagation_events") or [],
            max_step=max_step,
        )

    with tab_loop:
        loop = doc.get("looping") or {"present": False}
        st.caption(TRAJECTORY_LABEL_DEFINITIONS["looping"])
        loop_present = st.checkbox("Looping present", value=bool(loop.get("present")), key=f"loop_p::{selected_episode}")
        loop_steps = st.text_input(
            "Looping steps (comma-separated)",
            value=", ".join(str(s) for s in (loop.get("steps") or [])),
            key=f"loop_s::{selected_episode}",
        )
        loop_pattern = st.text_input(
            "Looping pattern (free text; common values below)",
            value=str(loop.get("pattern") or ""),
            key=f"loop_pat::{selected_episode}",
            help=LOOPING_PATTERN_EXAMPLES,
        )
        loop_conf = st.selectbox(
            "Looping confidence",
            CONFIDENCE_LEVELS,
            index=CONFIDENCE_LEVELS.index(loop["confidence"]) if loop.get("confidence") in CONFIDENCE_LEVELS else 1,
            key=f"loop_conf::{selected_episode}",
        )

    with tab_ext:
        ext = doc.get("external_system") or {"present": False}
        st.caption(TRAJECTORY_LABEL_DEFINITIONS["external_system"])
        ext_present = st.checkbox(
            "External system factor present",
            value=bool(ext.get("present")),
            key=f"ext_p::{selected_episode}",
        )
        ext_explanation = st.text_area(
            "External system explanation",
            value=str(ext.get("explanation") or ""),
            height=80,
            key=f"ext_expl::{selected_episode}",
        )
        ext_conf = st.selectbox(
            "External system confidence",
            CONFIDENCE_LEVELS,
            index=CONFIDENCE_LEVELS.index(ext["confidence"]) if ext.get("confidence") in CONFIDENCE_LEVELS else 1,
            key=f"ext_conf::{selected_episode}",
        )

    notes = st.text_area("Trajectory notes (`notes`)", value=str(doc.get("notes") or ""), height=100, key=f"notes::{selected_episode}")

    sl, e_step = step_labels_from_rows(edited_step_rows)
    pe, e_prop = propagation_from_rows(edited_prop_rows)
    loop_final = _build_looping_payload(loop_present, loop_steps, loop_pattern, loop_conf)
    ext_final = _build_external_system_payload(ext_present, ext_explanation, ext_conf)
    val_errors = e_step + e_prop + validate_failure_root_causes(sl, is_failure=is_failure)

    with tab_preview:
        preview_doc = {
            "schema_version": SCHEMA_VERSION,
            "trace_id": trace_id_override.strip() or selected_episode,
            "episode_id": selected_episode,
            "is_failure_trace": is_failure,
            "step_labels": sl,
            "propagation_events": pe,
            "looping": loop_final,
            "external_system": ext_final,
            "notes": notes.strip(),
        }
        if val_errors:
            st.error("Fix validation errors before save:\n" + "\n".join(val_errors))
        st.json(preview_doc, expanded=True)

    col_save, col_hint = st.columns([1, 2])
    with col_save:
        save_clicked = st.button("Save labels to JSONL", type="primary")
    with col_hint:
        st.caption(f"Writes one JSON object per episode to `{annotations_path}`.")

    if save_clicked:
        if val_errors:
            st.error("Not saved:\n" + "\n".join(val_errors))
        else:
            for ev in pe:
                if len(ev.get("affected_steps") or []) < 2:
                    st.warning(
                        f"Propagation from step {ev.get('source_step')}: affected_steps has fewer than 2 steps. "
                        "Instructions recommend ≥2 downstream steps; saving anyway."
                    )
            out_row = {
                "annotation_key": selected_episode,
                "schema_version": SCHEMA_VERSION,
                "trace_id": trace_id_override.strip() or selected_episode,
                "episode_id": selected_episode,
                "is_failure_trace": bool(is_failure),
                "step_labels": sl,
                "propagation_events": pe,
                "looping": loop_final,
                "external_system": ext_final,
                "notes": notes.strip(),
            }
            annotation_rows[selected_episode] = out_row
            save_annotations(annotations_path, annotation_rows)
            st.success("Saved.")
            st.cache_resource.clear()

    st.divider()
    st.subheader("Trajectory")
    step_pick_options = _ordered_step_indices(events)
    focus_widget_key = f"focus_step_pick::{selected_episode}"
    traj_df_key = f"traj_df::{selected_episode}"
    has_traj = bool(step_pick_options)
    cur_focus = 0
    if not step_pick_options:
        st.info("This episode has no events.")
    else:
        st.caption(
            "Choose **Focus step** from the dropdown, or select a **single row** in the trajectory table. "
            "The **focus_here** column (●) only marks the current focus; it is not a control."
        )
        if focus_widget_key not in st.session_state:
            st.session_state[focus_widget_key] = step_pick_options[0]

        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
        show_full = tcol1.checkbox("Full observation/thought", value=False)
        show_full_tool_args = tcol2.checkbox("Full tool args", value=False)
        show_full_action_payload = tcol3.checkbox("Full action payload", value=False)
        show_full_state = tcol4.checkbox("Full state snapshot", value=False)

        cur_for_table = int(st.session_state[focus_widget_key])
        event_rows = []
        for ev in events:
            s = int(ev.get("step_index", 0))
            mark = "●" if s == cur_for_table else ""
            snap = ev.get("state_snapshot") or {}
            if show_full_state:
                state_cell = _short_json(snap, n=8000) if snap else ""
            else:
                state_cell = _state_snapshot_summary(snap)
            event_rows.append(
                {
                    "focus_here": mark,
                    "step": s,
                    "timestamp": ev.get("timestamp") or "",
                    "action_type": ev.get("action_type") or "",
                    "action": ev.get("action_name"),
                    "tool": ev.get("tool_name"),
                    "state": state_cell,
                    "tool_args": ev.get("tool_args") if show_full_tool_args else _short_json(ev.get("tool_args")),
                    "action_payload": ev.get("action_payload") if show_full_action_payload else _short_json(ev.get("action_payload")),
                    "reward": ev.get("reward"),
                    "step_outcome": ev.get("step_outcome") or "",
                    "event_final_outcome": ev.get("final_outcome", ""),
                    "event_goal": _short(ev.get("goal"), n=160) if not show_full else (ev.get("goal") or ""),
                    "metadata": _event_metadata_summary(ev.get("metadata")),
                    "terminal": ev.get("is_terminal"),
                    "observation": ev.get("observation") if show_full else _short(ev.get("observation")),
                    "thought": ev.get("agent_thought") if show_full else _short(ev.get("agent_thought")),
                }
            )
        traj_df = pd.DataFrame(event_rows)
        sel = st.dataframe(
            traj_df,
            use_container_width=True,
            height=420,
            on_select="rerun",
            selection_mode="single-row",
            key=traj_df_key,
        )
        if sel.selection.rows:
            row_i = int(sel.selection.rows[0])
            if 0 <= row_i < len(traj_df):
                picked = int(traj_df.iloc[row_i]["step"])
                # Must run before st.selectbox(..., key=focus_widget_key) — Streamlit forbids
                # assigning that key after the widget is created.
                st.session_state[focus_widget_key] = picked

        st.selectbox(
            "Focus step (full text / neighbor context)",
            options=step_pick_options,
            key=focus_widget_key,
        )
        cur_focus = int(st.session_state[focus_widget_key])

    if has_traj:
        j = _list_index_for_focus_step(events, cur_focus)
        ev_full = _event_for_focus_step(events, cur_focus)
        with st.expander("Full text at focus step", expanded=False):
            st.markdown("**Observation**")
            st.text(ev_full.get("observation") or "")
            st.markdown("**Thought**")
            st.text(ev_full.get("agent_thought") or "")

        st.subheader("Step context")
        step = cur_focus
        left_col, right_col = st.columns(2)
        prev_ev = events[j - 1] if j is not None and j > 0 else None
        cur_ev = ev_full
        next_ev = events[j + 1] if j is not None and 0 <= j + 1 < len(events) else None
        with left_col:
            st.markdown(f"**Step {step - 1} (previous)**")
            st.json(prev_ev if prev_ev else {"info": "N/A"}, expanded=False)
            st.markdown(f"**Step {step} (focus)**")
            st.json(cur_ev if cur_ev else {"info": "N/A"}, expanded=False)
        with right_col:
            st.markdown(f"**Step {step + 1} (next)**")
            st.json(next_ev if next_ev else {"info": "N/A"}, expanded=False)
            if cur_ev and next_ev:
                st.markdown("**Quick diff**")
                st.write(
                    {
                        "action_change": f"{cur_ev.get('action_name')} -> {next_ev.get('action_name')}",
                        "tool_change": f"{cur_ev.get('tool_name')} -> {next_ev.get('tool_name')}",
                        "url_change": f"{(cur_ev.get('state_snapshot') or {}).get('url')} -> {(next_ev.get('state_snapshot') or {}).get('url')}",
                    }
                )

    st.subheader("Labeling progress (this run)")
    labeled = sum(
        1
        for v in annotation_rows.values()
        if v.get("step_labels")
        or v.get("is_failure_trace")
        or v.get("propagation_events")
        or (v.get("looping") or {}).get("present")
        or (v.get("external_system") or {}).get("present")
    )
    st.write({"episodes_in_run": len(traces), "episodes_with_saved_labels": labeled, "label_file": annotations_path})


if __name__ == "__main__":
    main()
