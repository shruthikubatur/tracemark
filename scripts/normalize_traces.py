"""Convert raw agent logs into the normalized.jsonl format the annotation UI reads.

Example:
    python scripts/normalize_traces.py --dataset taubench \
        --input path/to/results.json --output outputs/taubench_run/normalized.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tracemark.datasets import ADAPTERS, get_adapter  # noqa: E402
from tracemark.validators import validate_trace  # noqa: E402


def run_normalization(dataset: str, input_path: Path, output_path: Path) -> int:
    adapter = get_adapter(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for trace in adapter.iter_traces(input_path):
            payload = trace.model_dump()
            payload["validation_warnings"] = validate_trace(trace)
            f.write(json.dumps(payload) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--input", required=True, help="Raw results file or folder")
    parser.add_argument("--output", required=True, help="Path for normalized.jsonl")
    args = parser.parse_args()
    count = run_normalization(args.dataset, Path(args.input), Path(args.output))
    print(f"Wrote {count} traces to {args.output}")


if __name__ == "__main__":
    main()
