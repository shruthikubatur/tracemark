# tracemark

A local Streamlit tool for labeling failures in AI agent trajectories. Annotators step through a trace and record step-level failure primitives (Boundary, State, Decision), root-cause and secondary-cause flags, and trajectory-level labels (propagation, looping, external system). Labels are saved as JSONL.

## Layout

- `scripts/annotation_ui.py`: the Streamlit app. Long; read the part you need.
- `src/tracemark/ontology.py`: the label taxonomy and validation. Single source of truth for label names.
- `src/tracemark/intercode/`: optional InterCode SQL helpers used by the UI.
- `scripts/split_golden_manifest.py`, `scripts/filter_normalized_by_manifest.py`: split work between annotators.
- `examples/normalized.jsonl`: two synthetic traces. The UI loads these when `outputs/` has no runs.
- `docs/LABELING.md`: labeling workflow and taxonomy descriptions.
- Input format: see "Input format" in `README.md`.

## Commands

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m streamlit run scripts/annotation_ui.py --server.address localhost
```

Always launch with `python -m streamlit` so the venv's Python is used.

## Checking UI changes

After changing `annotation_ui.py`, run a headless smoke test as well as `pytest`:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("scripts/annotation_ui.py", default_timeout=60).run()
assert not at.exception
print([(m.label, m.value) for m in at.metric])
```

Test the empty case too (no `outputs/` folder). A past bug only showed up there.

## Conventions

- Python 3.10+. Keep runtime dependencies minimal (currently pandas and streamlit).
- Licensed Apache 2.0. Contributions are accepted under the same license.
- Never commit datasets, converted traces or labels. `data/` and `outputs/` are gitignored.
- Do not rename taxonomy labels without updating `ontology.py`, its tests and `docs/LABELING.md` together.
- Docs style: short, simple sentences. No em dashes or double dashes.
- Work on branches; open pull requests into `main`.

## Current work: Who&When

Extending tracemark to the Who&When multi-agent failure attribution dataset (Hugging Face `Kevin355/Who_and_When`; paper arXiv 2505.00212). Two subsets: Algorithm-Generated and Hand-Crafted. 184 failure logs annotated with the responsible agent, the decisive error step and an explanation.

- Planned converter: `scripts/convert_who_and_when.py`, writing tracemark's `normalized.jsonl`.
- Verify field names and step indexing from the actual data. Do not assume them.
- Blind labeling matters. The UI displays episode `metadata`, so the dataset's answers (who failed, when, why) must not go there or anywhere else the UI shows. Store them in a separate sidecar file keyed by `episode_id`, for scoring later.
- Every Who&When log is a failure, so `final_outcome` is `failure` for all rows.
- Open design questions: showing which agent acted at each step, whether the taxonomy covers multi-agent failures such as bad handoffs, handling long traces, and how to score agreement with the dataset's decisive error step.
