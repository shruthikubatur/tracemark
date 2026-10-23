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

## Loading CRMArena and tau-bench

tracemark includes converters for two public agent benchmarks. Each one turns raw result files into the input format above.

| Dataset | `--dataset` | Input |
|---|---|---|
| tau-bench (tau2) | `taubench` | A `results.json` file, or a run folder containing one |
| CRMArena | `crmarena` | A folder of `results_*.json` files |

Write the output to `outputs/<run_name>/normalized.jsonl`. The UI then lists the run in its sidebar dropdown.

**tau-bench**

```bash
python scripts/normalize_traces.py --dataset taubench \
  --input path/to/results.json \
  --output outputs/taubench_run/normalized.jsonl
```

**CRMArena.** The result files in the upstream repo are stored with Git LFS. Use the helper to download them first:

```bash
git clone https://github.com/SalesforceAIResearch/CRMArena.git /tmp/crmarena

python scripts/download_crmarena_results.py \
  --repo-root /tmp/crmarena \
  --output-dir data/raw/crmarena/b2b_multi_o1 \
  --org b2b --turns multi-turn --model-contains o1-2024-12-17

python scripts/normalize_traces.py --dataset crmarena \
  --input data/raw/crmarena/b2b_multi_o1 \
  --output outputs/crmarena_b2b_multi_o1/normalized.jsonl
```

To add another dataset, write an adapter in `src/tracemark/datasets/` that subclasses `BaseDatasetAdapter`, and register it in `src/tracemark/datasets/__init__.py`.

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
