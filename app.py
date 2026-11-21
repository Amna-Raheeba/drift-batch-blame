"""Streamlit dashboard for drift-batch-blame.

Three tabs:
1. Batch blame -- the seeded demo stream audited live: per-feature
   drift ratios and LOBO / Shapley blame per ingestion batch.
2. Minimal blame set -- the greedy removal trace: how the pooled
   drift score falls as blamed batches are removed, and whether the
   remainder is restored.
3. Evaluation -- the shipped aggregate results from
   results/evaluation.json.

Run:  streamlit run app.py
"""

import io
import json
import os

import pandas as pd
import streamlit as st

from drift_blame.blame import (
    greedy_minimal_set,
    lobo_blame,
    pooled_counts,
    shapley_blame,
)
from drift_blame.drift import Bins, count_rows, drift_report
from drift_blame.generate import demo_scenario, generate_stream
from drift_blame.models import ALL_FEATURES


@st.cache_data(show_spinner=False)
def load_demo_audit():
    stream = generate_stream(demo_scenario())
    bins = Bins(stream.reference_rows)
    ref_counts = count_rows(bins, stream.reference_rows)
    batch_counts = [count_rows(bins, b.rows) for b in stream.batches]
    report = drift_report(ref_counts, pooled_counts(batch_counts))
    lobo = lobo_blame(ref_counts, batch_counts)
    shapley = shapley_blame(ref_counts, batch_counts, n_permutations=64, seed=42)
    removed, trace = greedy_minimal_set(ref_counts, batch_counts)
    return stream, report, lobo, shapley, removed, trace


def tab_batch_blame(stream, report, lobo, shapley):
    st.subheader("Demo stream: which batches caused the drift?")
    st.write(
        "Pooled drift score %.4f -- max feature ratio %.2f x null "
        "(restored: %s). Two batches in this seeded stream carry "
        "planted shifts; the blame columns find them."
        % (report["score"], report["max_ratio"], report["restored"])
    )
    feature_rows = []
    for feat in ALL_FEATURES:
        stat, null, ratio = report["features"][feat]
        feature_rows.append(
            {"feature": feat, "statistic": round(stat, 4),
             "null": round(null, 4), "ratio": round(ratio, 2)}
        )
    st.dataframe(pd.DataFrame(feature_rows), hide_index=True)
    blame_rows = []
    for i, batch in enumerate(stream.batches):
        blame_rows.append(
            {"batch": batch.batch_id, "source": batch.source,
             "rows": batch.size, "LOBO blame": round(lobo[i], 4),
             "Shapley blame": round(shapley[i], 4),
             "planted culprit": batch.batch_id in stream.truth}
        )
    frame = pd.DataFrame(blame_rows)
    st.dataframe(frame, hide_index=True)
    st.bar_chart(frame.set_index("batch")[["LOBO blame", "Shapley blame"]])


def tab_minimal_set(stream, removed, trace):
    st.subheader("Greedy minimal blame set")
    names = [stream.batches[i].batch_id for i in removed]
    if names:
        st.write("Removed in order: %s" % ", ".join(names))
    else:
        st.write("No removal was needed or no removal reduced the drift.")
    trace_rows = []
    for step, entry in enumerate(trace):
        removed_idx, score, max_ratio, restored = entry
        label = "start" if removed_idx is None else stream.batches[removed_idx].batch_id
        trace_rows.append(
            {"step": step, "removed": label, "score": round(score, 4),
             "max_ratio": round(max_ratio, 2), "restored": restored}
        )
    frame = pd.DataFrame(trace_rows)
    st.dataframe(frame, hide_index=True)
    st.line_chart(frame.set_index("step")[["score"]])
    final = trace[-1]
    st.write(
        "Final state: score %.4f, max ratio %.2f x null, restored: %s. "
        "Planted culprits in this demo stream: %s."
        % (final[1], final[2], final[3], ", ".join(stream.culprit_ids))
    )


def tab_evaluation():
    st.subheader("Evaluation: 16 seeded scenarios")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "evaluation.json")
    if not os.path.exists(path):
        st.write("results/evaluation.json not found -- run "
                 "`python -m drift_blame evaluate --out results` first.")
        return
    with io.open(path, "r", encoding="utf-8") as fh:
        results = json.load(fh)
    agg = results["aggregate"]
    rows = []
    for name in ["lobo", "shapley", "psi_mapped", "own_drift"]:
        metrics = agg["methods"][name]
        rows.append(
            {"method": name,
             "precision@k": round(metrics["precision_at_k"], 3),
             "recall@k": round(metrics["recall_at_k"], 3)}
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.write(
        "Greedy minimal blame set: exact match %.3f, mean F1 %.3f. "
        "Top source contains a culprit: Shapley %.3f, LOBO %.3f."
        % (agg["minimal_set_exact_rate"], agg["minimal_set_mean_f1"],
           agg["source_shapley_hit_rate"], agg["source_lobo_hit_rate"])
    )
    if "clean_control" in results:
        clean = results["clean_control"]
        st.write(
            "Clean control (%d streams, no planted culprits): flagged "
            "%.3f, empty blame set %.3f."
            % (clean["n_streams"], clean["flagged_rate"],
               clean["empty_set_rate"])
        )


def main():
    st.title("Drift Batch Blame")
    st.write(
        "Provenance-level attribution for dataset drift: blame the "
        "ingestion batches that caused it, not just the features "
        "that moved. All data is synthetic and seeded."
    )
    stream, report, lobo, shapley, removed, trace = load_demo_audit()
    tab1, tab2, tab3 = st.tabs(
        ["Batch blame", "Minimal blame set", "Evaluation"]
    )
    with tab1:
        tab_batch_blame(stream, report, lobo, shapley)
    with tab2:
        tab_minimal_set(stream, removed, trace)
    with tab3:
        tab_evaluation()


if __name__ == "__main__":
    main()
