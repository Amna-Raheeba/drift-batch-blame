"""Headline-drift guard.

Re-runs the full seeded evaluation and compares the aggregate
headline numbers against the shipped results/evaluation.json.
Exits non-zero if any headline value moved by more than the
tolerance -- i.e. if the code no longer reproduces the published
results.

Run from the repository root:  python scripts/reproduce_results.py
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_blame.evaluate import run_evaluation

TOLERANCE = 0.02


def headline_values(results):
    agg = results["aggregate"]
    values = {
        "minimal_set_exact_rate": agg["minimal_set_exact_rate"],
        "minimal_set_mean_f1": agg["minimal_set_mean_f1"],
        "source_shapley_hit_rate": agg["source_shapley_hit_rate"],
        "source_lobo_hit_rate": agg["source_lobo_hit_rate"],
    }
    for name, metrics in agg["methods"].items():
        values["%s_precision_at_k" % name] = metrics["precision_at_k"]
        values["%s_recall_at_k" % name] = metrics["recall_at_k"]
    if "clean_control" in results:
        clean = results["clean_control"]
        values["clean_flagged_rate"] = clean["flagged_rate"]
        values["clean_empty_set_rate"] = clean["empty_set_rate"]
    return values


def compare_results(saved, fresh, tolerance=TOLERANCE):
    saved_values = headline_values(saved)
    fresh_values = headline_values(fresh)
    diffs = []
    for key in sorted(saved_values):
        if key not in fresh_values:
            diffs.append("%s missing from fresh run" % key)
            continue
        delta = abs(saved_values[key] - fresh_values[key])
        if delta > tolerance:
            diffs.append(
                "%s: shipped %.4f vs reproduced %.4f (delta %.4f)"
                % (key, saved_values[key], fresh_values[key], delta)
            )
    return (len(diffs) == 0), diffs


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shipped_path = os.path.join(root, "results", "evaluation.json")
    with io.open(shipped_path, "r", encoding="utf-8") as fh:
        saved = json.load(fh)
    fresh = run_evaluation()
    ok, diffs = compare_results(saved, fresh)
    if ok:
        print("PASS: reproduced headline matches shipped results "
              "within %.3f" % TOLERANCE)
        return 0
    print("FAIL: headline drift detected")
    for diff in diffs:
        print(" - %s" % diff)
    return 1


if __name__ == "__main__":
    sys.exit(main())
