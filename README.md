# tracemark

tracemark is a small annotation tool for labeling failures in AI agent trajectories. An annotator steps through a trace, marks the steps where things went wrong, and records:

- a **failure primitive** for each step (Boundary, State, or Decision), with a subtype and tag
- whether the step is a **root cause** or a **secondary cause**
- trajectory-level labels: **propagation**, **looping**, and **external system** effects
- a confidence level and a free-text explanation

It runs locally as a Streamlit app. Labels are saved as JSONL.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# optional, only for InterCode SQL traces:
pip install -e ".[intercode]"
```

## Quick start

```bash
streamlit run scripts/annotation_ui.py
```

In the sidebar, open **Advanced path override**, then:

1. Set **normalized.jsonl** to `examples/normalized.jsonl`.
2. Set **Label store (JSONL)** to a new path, for example `outputs/labels_example.jsonl`.

Give each annotator their own label file so two people never write to the same file.

## Input format

The tool reads one JSON object per line. Each object is one episode:

| Field | Required | Notes |
|---|---|---|
| `episode_id` | yes | Unique ID for the episode |
| `final_outcome` | yes | `success` or `failure` |
| `events` | yes | List of steps, see below |
| `goal` | no | The task given to the agent |
| `dataset_name`, `task_id`, `trace_id` | no | Shown as metadata |
| `metadata` | no | Any extra info |

Each event can have `step_index` (required), `action_type`, `action_name`, `tool_name`, `tool_args`, `observation`, `agent_thought`, `state_snapshot`, `reward`, and `is_terminal`.

See `examples/normalized.jsonl` for a minimal example.

## Taxonomy

The label taxonomy is defined in `src/tracemark/ontology.py` and described in [docs/LABELING.md](docs/LABELING.md).

## Splitting work between annotators

`scripts/split_golden_manifest.py` splits an episode manifest into two halves. `scripts/filter_normalized_by_manifest.py` filters a trace file down to the episodes in one manifest. The workflow is in [docs/LABELING.md](docs/LABELING.md).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

Contributions are accepted under the same license.

## Citation

If you use tracemark in your research, please cite it using the metadata in [CITATION.cff](CITATION.cff).
