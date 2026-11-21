"""Example: audit the shipped demo stream from data/demo_stream.json.

Loads the stored stream (reference rows, batches, planted truth),
recomputes the drift audit from the stored rows, and prints the
blame report.  Run build_cases first if the data file is missing:

    python scripts/build_cases.py
    python examples/run_audit.py
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_blame.blame import greedy_minimal_set, lobo_blame, pooled_counts, shapley_blame
from drift_blame.drift import Bins, count_rows, drift_report
from drift_blame.models import Batch, Stream
from drift_blame.report import stream_report


def load_stream(path):
    with io.open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    batches = [
        Batch(batch_id=b["batch_id"], source=b["source"], day=b["day"],
              rows=b["rows"])
        for b in payload["batches"]
    ]
    return Stream(reference_rows=payload["reference_rows"], batches=batches,
                  truth=payload["truth"])


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stream = load_stream(os.path.join(root, "data", "demo_stream.json"))
    bins = Bins(stream.reference_rows)
    ref_counts = count_rows(bins, stream.reference_rows)
    batch_counts = [count_rows(bins, b.rows) for b in stream.batches]
    report = drift_report(ref_counts, pooled_counts(batch_counts))
    lobo = lobo_blame(ref_counts, batch_counts)
    shapley = shapley_blame(ref_counts, batch_counts, n_permutations=64, seed=42)
    removed, trace = greedy_minimal_set(ref_counts, batch_counts)
    print(stream_report(stream, ref_counts, batch_counts, report, lobo,
                        shapley, removed, trace))
    print("Planted culprits (ground truth): %s" % ", ".join(stream.culprit_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
