"""Notebook 01 -- demo run (runnable as a plain Python script).

Generates the seeded demo stream (two planted culprit batches),
audits it, and prints the full blame report: per-feature drift,
LOBO and Shapley batch blame, and the greedy minimal blame set.
The planted ground truth is printed last so the recovery can be
checked by eye.

Run from the repository root:  python notebooks/01_demo_run.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_blame.blame import greedy_minimal_set, lobo_blame, pooled_counts, shapley_blame
from drift_blame.drift import Bins, count_rows, drift_report
from drift_blame.generate import demo_scenario, generate_stream
from drift_blame.report import stream_report


def main():
    stream = generate_stream(demo_scenario())
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
    recovered = sorted(stream.batches[i].batch_id for i in removed)
    print("Minimal blame set recovered:    %s" % ", ".join(recovered))
    return 0


if __name__ == "__main__":
    sys.exit(main())
