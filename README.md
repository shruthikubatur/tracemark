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
| tau-bench (tau2) | `taubench` | A tau2 results JSON file, or a run folder containing `results.json` |
| CRMArena | `crmarena` | A folder of `results_*.json` files |

The UI always lists `taubench` and `crmarena` in its sidebar dropdown and reads them from `outputs/taubench/` and `outputs/crmarena/`. Use those output paths, as below. Any other `outputs/<run_name>/normalized.jsonl` also appears as its own run.

**tau-bench**

The [tau2-bench repo](https://github.com/sierra-research/tau2-bench) (MIT license) publishes the trajectories from its paper in `data/tau2/results/final/`. Each file holds one model on one domain, with 4 trials per task. File names follow `<agent model>_<domain>_<mode>_<user model>_4trials.json`. Domains are `airline`, `retail` and `telecom`, plus `telecom-workflow`. Files range from about 8 MB to 50 MB.

Download one file. This example is the smallest: o4-mini on airline, 200 trajectories. The URL is pinned to a fixed commit, so the data does not change under you.

```bash
mkdir -p data/raw/taubench
curl -L -o data/raw/taubench/o4-mini_airline.json \
  https://raw.githubusercontent.com/sierra-research/tau2-bench/033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f/data/tau2/results/final/o4-mini-2025-04-16_airline_default_gpt-4.1-2025-04-14_4trials.json

python scripts/normalize_traces.py --dataset taubench \
  --input data/raw/taubench/o4-mini_airline.json \
  --output outputs/taubench/normalized.jsonl
```

To get a different model or domain, replace the file name in the URL. The full list is in [that folder](https://github.com/sierra-research/tau2-bench/tree/033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f/data/tau2/results/final).

Note: these are the June 2025 paper results. The tau2 maintainers later fixed some airline and retail tasks, so those trajectories use the older task versions. Telecom is unchanged. For trajectories on the current tasks, run tau2 yourself (`tau2 run ...`, see its README). It saves results under `data/simulations/`, and this adapter reads those too.

**CRMArena.** The result files in the upstream repo are stored with Git LFS. Use the helper to download them first:

```bash
git clone https://github.com/SalesforceAIResearch/CRMArena.git /tmp/crmarena

python scripts/download_crmarena_results.py \
  --repo-root /tmp/crmarena \
  --output-dir data/raw/crmarena/b2b_multi_o1 \
  --org b2b --turns multi-turn --model-contains o1-2024-12-17

python scripts/normalize_traces.py --dataset crmarena \
  --input data/raw/crmarena/b2b_multi_o1 \
  --output outputs/crmarena/normalized.jsonl
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
