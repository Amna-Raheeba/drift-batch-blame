"""Command line interface: demo and evaluate."""

import argparse
import csv
import io
import json
import os
import sys

from .blame import greedy_minimal_set, lobo_blame, pooled_counts, shapley_blame
from .drift import Bins, count_rows, drift_report
from .evaluate import run_evaluation
from .generate import demo_scenario, generate_stream
from .report import evaluation_summary, headline, stream_report


def run_demo():
    scenario = demo_scenario()
    stream = generate_stream(scenario)
    bins = Bins(stream.reference_rows)
    ref_counts = count_rows(bins, stream.reference_rows)
    batch_counts = [count_rows(bins, b.rows) for b in stream.batches]
    report = drift_report(ref_counts, pooled_counts(batch_counts))
    lobo = lobo_blame(ref_counts, batch_counts)
    shapley = shapley_blame(ref_counts, batch_counts, n_permutations=64, seed=42)
    removed, trace = greedy_minimal_set(ref_counts, batch_counts)
    text = stream_report(stream, ref_counts, batch_counts, report, lobo,
                         shapley, removed, trace)
    print(text)
    print("Planted culprits (ground truth, demo only): %s"
          % ", ".join(stream.culprit_ids))
    return 0


def run_evaluate(out_dir):
    results = run_evaluation()
    os.makedirs(out_dir, exist_ok=True)
    with io.open(os.path.join(out_dir, "evaluation.json"), "w",
                 encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    with io.open(os.path.join(out_dir, "evaluation.csv"), "w",
                 encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "seed", "n_culprits", "culprit_batches", "pooled_score",
            "pooled_max_ratio", "lobo_precision_at_k", "lobo_recall_at_k",
            "shapley_precision_at_k", "shapley_recall_at_k",
            "psi_precision_at_k", "psi_recall_at_k",
            "own_drift_precision_at_k", "own_drift_recall_at_k",
            "minimal_set_exact", "minimal_set_f1",
            "source_shapley_hit", "source_lobo_hit",
        ])
        for s in results["scenarios"]:
            writer.writerow([
                s["seed"], s["n_culprits"], ";".join(s["culprit_batches"]),
                "%.4f" % s["pooled_score"], "%.3f" % s["pooled_max_ratio"],
                "%.3f" % s["methods"]["lobo"]["precision_at_k"],
                "%.3f" % s["methods"]["lobo"]["recall_at_k"],
                "%.3f" % s["methods"]["shapley"]["precision_at_k"],
                "%.3f" % s["methods"]["shapley"]["recall_at_k"],
                "%.3f" % s["methods"]["psi_mapped"]["precision_at_k"],
                "%.3f" % s["methods"]["psi_mapped"]["recall_at_k"],
                "%.3f" % s["methods"]["own_drift"]["precision_at_k"],
                "%.3f" % s["methods"]["own_drift"]["recall_at_k"],
                s["minimal_set"]["exact_match"],
                "%.3f" % s["minimal_set"]["f1"],
                s["source"]["shapley_hit"], s["source"]["lobo_hit"],
            ])
    with io.open(os.path.join(out_dir, "evaluation_summary.md"), "w",
                 encoding="utf-8") as fh:
        fh.write(evaluation_summary(results))
    with io.open(os.path.join(out_dir, "HEADLINE.md"), "w",
                 encoding="utf-8") as fh:
        fh.write(headline(results))
    print(evaluation_summary(results))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="drift_blame",
        description="Blame the ingestion batches that caused dataset drift.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="Audit the seeded demo stream and print a report.")
    ev = sub.add_parser("evaluate", help="Run the full seeded evaluation.")
    ev.add_argument("--out", default="results",
                    help="Directory for evaluation artifacts.")
    args = parser.parse_args(argv)
    if args.command == "demo":
        return run_demo()
    if args.command == "evaluate":
        return run_evaluate(args.out)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
