"""Regenerate the shipped data files from the seeded generators.

Writes:
- data/demo_stream.json  -- the demo scenario stream (reference rows,
  batches, planted truth), floats rounded to 4 decimals.
- data/scenarios.json    -- the canonical evaluation scenario specs.

Run from the repository root:  python scripts/build_cases.py
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_blame.generate import default_scenarios, demo_scenario, generate_stream


def _round_row(row):
    out = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = round(value, 4)
        else:
            out[key] = value
    return out


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    stream = generate_stream(demo_scenario())
    payload = {
        "reference_rows": [_round_row(r) for r in stream.reference_rows],
        "batches": [
            {
                "batch_id": b.batch_id,
                "source": b.source,
                "day": b.day,
                "rows": [_round_row(r) for r in b.rows],
            }
            for b in stream.batches
        ],
        "truth": stream.truth,
    }
    with io.open(os.path.join(data_dir, "demo_stream.json"), "w",
                 encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    specs = []
    for sc in default_scenarios():
        specs.append({
            "seed": sc.seed,
            "n_batches": sc.n_batches,
            "batch_size": sc.batch_size,
            "ref_size": sc.ref_size,
            "culprits": [
                {
                    "batch_index": c.batch_index,
                    "shift_type": c.shift_type,
                    "feature": c.feature,
                    "magnitude": c.magnitude,
                }
                for c in sc.culprits
            ],
        })
    with io.open(os.path.join(data_dir, "scenarios.json"), "w",
                 encoding="utf-8") as fh:
        json.dump(specs, fh, indent=2)
    print("wrote data/demo_stream.json and data/scenarios.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
