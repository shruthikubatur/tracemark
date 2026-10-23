# Golden-set labeling workflow

This document is the checklist for taking a fixed episode list, splitting it into two chunks, producing thin `normalized.jsonl` slices for the Streamlit UI, and handing work to another person.

Prerequisites:

- Python venv with this repo installed (`pip install -e .`).
- A **full** `normalized.jsonl` for each dataset you label, in the format described in the main [README.md](../README.md#input-format).

Paths below are **examples**. Replace them with your real directories and filenames.

---

## 1. Golden manifest

A manifest is a small JSON file that lists the episodes to label:

```json
{
  "episodes": [
    {"episode_id": "example-001"},
    {"episode_id": "example-002"}
  ]
}
```

Build it however suits your study. Each `episode_id` must match an `episode_id` in your `normalized.jsonl`.

---

## 2. Split the golden manifest into two halves

This writes two JSON files: **first half** of the `episodes` array → `part_a`, **second half** → `part_b`. Order is preserved; splits are deterministic for a given parent file.

**CRMArena**

```bash
python scripts/split_golden_manifest.py \
  --input outputs/golden/crmarena_balanced_n50_seed42.json
```

Default output directory is the same folder as the input. You should get:

- `outputs/golden/crmarena_balanced_n50_seed42__part_a.json`
- `outputs/golden/crmarena_balanced_n50_seed42__part_b.json`

**Tau-bench**

```bash
python scripts/split_golden_manifest.py \
  --input outputs/golden/taubench_balanced_n50_seed42.json
```

**InterCode SQL**

```bash
python scripts/split_golden_manifest.py \
  --input outputs/golden/intercode_sql_multiturn_n50_seed42.json
```

Optional: use custom suffixes instead of `part_a` / `part_b`:

```bash
python scripts/split_golden_manifest.py \
  --input outputs/golden/crmarena_balanced_n50_seed42.json \
  --label-a chunk_01 \
  --label-b chunk_02
```

---

## 3. Filter `normalized.jsonl` to only golden episodes

The annotation UI loads one `normalized.jsonl`. Filtering keeps the episode dropdown small and avoids labeling the wrong row.

**CRMArena — chunk A**

```bash
python scripts/filter_normalized_by_manifest.py \
  --manifest outputs/golden/crmarena_balanced_n50_seed42__part_a.json \
  --normalized outputs/crmarena_b2b_multi_o1/normalized.jsonl \
  --output outputs/crmarena_b2b_multi_o1/normalized_golden_part_a.jsonl
```

**CRMArena — chunk B**

```bash
python scripts/filter_normalized_by_manifest.py \
  --manifest outputs/golden/crmarena_balanced_n50_seed42__part_b.json \
  --normalized outputs/crmarena_b2b_multi_o1/normalized.jsonl \
  --output outputs/crmarena_b2b_multi_o1/normalized_golden_part_b.jsonl
```

**Tau-bench** (same pattern; point `--normalized` at the run that produced those `episode_id`s):

```bash
python scripts/filter_normalized_by_manifest.py \
  --manifest outputs/golden/taubench_balanced_n50_seed42__part_a.json \
  --normalized outputs/taubench_run/normalized.jsonl \
  --output outputs/taubench_run/normalized_golden_part_a.jsonl

python scripts/filter_normalized_by_manifest.py \
  --manifest outputs/golden/taubench_balanced_n50_seed42__part_b.json \
  --normalized outputs/taubench_run/normalized.jsonl \
  --output outputs/taubench_run/normalized_golden_part_b.jsonl
```

**InterCode SQL** (same `--normalized` for both chunks)

```bash
python scripts/filter_normalized_by_manifest.py \
  --manifest outputs/golden/intercode_sql_multiturn_n50_seed42__part_a.json \
  --normalized outputs/intercode_sql_gpt35_multiturn/normalized.jsonl \
  --output outputs/intercode_sql_gpt35_multiturn/normalized_golden_part_a.jsonl

python scripts/filter_normalized_by_manifest.py \
  --manifest outputs/golden/intercode_sql_multiturn_n50_seed42__part_b.json \
  --normalized outputs/intercode_sql_gpt35_multiturn/normalized.jsonl \
  --output outputs/intercode_sql_gpt35_multiturn/normalized_golden_part_b.jsonl
```

If the script prints warnings about missing `episode_id`s, the manifest and `normalized.jsonl` came from different normalization runs or paths — fix alignment before labeling.

---

## 4. Run the annotation UI

```bash
streamlit run scripts/annotation_ui.py
```

In the sidebar:

1. Open **Advanced path override**.
2. Set **normalized.jsonl** to your filtered file, e.g. `outputs/crmarena_b2b_multi_o1/normalized_golden_part_b.jsonl`.
3. Set **Label store (JSONL)** to a **new** path for that chunk so two people never write the same file, e.g. `outputs/crmarena_b2b_multi_o1/annotations/labels_v0_part_b.jsonl`.

The default run dropdown may not match your paths; overrides are fine.

---

## Failure taxonomy (current)

Use the following primitive, subtype, and tag names in labeling so annotations stay consistent with the UI schema.

### Step-level primitives

**Boundary** — The agent maps the goal, entity, scope, time, quantity, or task requirement incorrectly.

- `semantic`
  - `goal_misinterpretation`
  - `entity_misinterpretation`
  - `capability_misinterpretation`
- `scope`
  - `scope_underreach`
  - `scope_overreach`
- `temporal` (leaf — no child tag)
- `arithmetic` (leaf — no child tag)

**State** — The agent fails to acquire, preserve, update, or use relevant context, or is guided by misleading context.

- `missing_context` (leaf)
- `unused_context` (leaf)
- `misguiding_context` (leaf)

**Decision** — The agent has enough of the right representation and context, but chooses the wrong tool, wrong sequence, wrong response, or wrong commitment strategy.

- `tool_selection`
  - `tool_choice`
  - `tool_sequence`
- `response_selection`
  - `unsupported_conclusion`
  - `hallucinated_response`
- `commitment_strategy`
  - `premature_commitment`
  - `no_clarification`

### Trajectory-level labels (separate from step-level)

These are stored in separate fields, not as step-level primitives:

- **propagation** — An upstream failure affects later steps or the final outcome.
- **looping** — The agent repeats actions or reasoning without meaningful progress.
- **external_system** — The trace is materially affected by something outside the agent's internal reasoning/action loop (defective input, missing supplied context, bad tool output, grader issue, platform constraint, etc.).

Propagation source primitives are **Boundary**, **State**, or **Decision** (any step-level primitive).

---

## What to commit vs what to share

### Default in this repo

`outputs/` is listed in `.gitignore`, so **generated files under `outputs/` are not committed** unless someone force-adds them.

### Recommended for a small team

| Artifact | Commit to git? | Why |
|----------|----------------|-----|
| **This file (`LABELING.md`)** | Yes | Everyone knows the same steps. |
| **Golden parent manifest** (the single `*_balanced_n50_*.json` before split) | Optional but good | Reproducible definition of the 50 episodes; small JSON. |
| **Split manifests (`__part_a.json`, `__part_b.json`)** | Optional | Small; avoids ambiguity about which half is which. |
| **Filtered `normalized_golden_*.jsonl`** | Usually no | Large; regenerates in seconds from manifest + full `normalized.jsonl`. |
| **Label JSONL** | Optional / later | Merge policy is up to you; often kept until review then committed or exported. |

**Practical handoff without committing binaries:**

1. Commit **`LABELING.md`** (and optionally manifests under a tracked folder, e.g. `data/manifests/golden/`, if you want them in git).
2. **Share with your teammate** (Drive, Slack, ticket attachment): their **split manifest** (`__part_b.json` or whichever chunk they own) **or** the parent manifest plus an explicit line: “You label **part_b** (rows in the second half of `episodes`).”
3. Teammate places the JSON where they want, builds **their** filtered JSONL with `filter_normalized_by_manifest.py` (they need the same **full** `normalized.jsonl` locally, from the same raw run you used).

**If you do want everything in git:** copy the small manifest JSONs into `data/manifests/` (not ignored), commit those only; keep large JSONL out or use Git LFS.

---

## What your teammate runs (minimal)

They need the repo, venv, **your** full `normalized.jsonl` for that dataset (same normalization as yours), and **their** split manifest file.

Example for **CRMArena, part_b only**:

```bash
# one-time
cd tracemark && python -m venv .venv && source .venv/bin/activate
pip install -e .

# put crmarena_balanced_n50_seed42__part_b.json where they keep manifests, then:
python scripts/filter_normalized_by_manifest.py \
  --manifest path/to/crmarena_balanced_n50_seed42__part_b.json \
  --normalized path/to/crmarena_b2b_multi_o1/normalized.jsonl \
  --output path/to/crmarena_b2b_multi_o1/normalized_golden_part_b.jsonl

streamlit run scripts/annotation_ui.py
```

In the UI: Advanced override → `normalized_golden_part_b.jsonl` and a **dedicated** labels JSONL path.

They do **not** need to re-run `split_golden_manifest.py` unless they are regenerating the study from scratch.

---

## Quick reference: who labels what

| Chunk file suffix | Episodes | Typical use |
|-------------------|----------|-------------|
| `__part_a.json` | First half of the parent manifest’s `episodes` list | Chunk A |
| `__part_b.json` | Second half | Chunk B |

Agree out of band (chat, doc) which human uses **part_a** vs **part_b** for each dataset (CRMArena, Tau-bench, InterCode SQL, etc.). The JSON metadata field `split_chunk` records which file is which.

---

## Troubleshooting

- **Filter wrote fewer lines than expected:** some `episode_id`s in the manifest do not appear in your `normalized.jsonl`. Check that both came from the same run.
- **UI shows wrong or empty traces:** Confirm the filtered JSONL path and that `streamlit` was restarted after regenerating files.

For general UI behavior, see the main [README.md](../README.md).
