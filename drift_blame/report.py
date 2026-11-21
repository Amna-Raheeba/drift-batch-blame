"""Markdown reports for a single stream audit and for the evaluation."""

from .models import ALL_FEATURES


def stream_report(stream, ref_counts, batch_counts, report, lobo, shapley,
                  removed, trace):
    """Human-readable audit of one stream (used by demo and examples)."""
    lines = []
    lines.append("# Drift batch blame -- stream audit")
    lines.append("")
    lines.append("Pooled drift score: %.4f (max feature ratio %.2f x null, "
                 "restored: %s)" % (report["score"], report["max_ratio"],
                                     report["restored"]))
    lines.append("")
    lines.append("## Per-feature drift (pooled)")
    lines.append("")
    lines.append("| feature | statistic | null | ratio |")
    lines.append("|---|---|---|---|")
    for feat in ALL_FEATURES:
        stat, null, ratio = report["features"][feat]
        lines.append("| %s | %.4f | %.4f | %.2f |" % (feat, stat, null, ratio))
    lines.append("")
    lines.append("## Batch blame")
    lines.append("")
    lines.append("| batch | source | rows | LOBO blame | Shapley blame |")
    lines.append("|---|---|---|---|---|")
    order = sorted(range(len(stream.batches)), key=lambda i: -lobo[i])
    for i in order:
        b = stream.batches[i]
        lines.append("| %s | %s | %d | %.4f | %.4f |"
                     % (b.batch_id, b.source, b.size, lobo[i], shapley[i]))
    lines.append("")
    lines.append("## Minimal blame set (greedy removal)")
    lines.append("")
    if removed:
        names = [stream.batches[i].batch_id for i in removed]
        lines.append("Removed in order: %s" % ", ".join(names))
    else:
        lines.append("No removal needed or no removal reduces drift.")
    final = trace[-1]
    lines.append("")
    lines.append("After removal: score %.4f, max ratio %.2f, restored: %s"
                 % (final[1], final[2], final[3]))
    lines.append("")
    return "\n".join(lines)


def evaluation_summary(results):
    agg = results["aggregate"]
    lines = []
    lines.append("# Evaluation summary")
    lines.append("")
    lines.append("Scenarios: %d (mean planted culprits per scenario: %.2f)"
                 % (agg["n_scenarios"], agg["mean_culprits"]))
    lines.append("")
    lines.append("## Culprit ranking (k = number of planted culprits)")
    lines.append("")
    lines.append("| method | precision@k | recall@k |")
    lines.append("|---|---|---|")
    for name in ["lobo", "shapley", "psi_mapped", "own_drift"]:
        m = agg["methods"][name]
        lines.append("| %s | %.3f | %.3f |"
                     % (name, m["precision_at_k"], m["recall_at_k"]))
    lines.append("")
    lines.append("## Minimal blame set")
    lines.append("")
    lines.append("- Exact-match rate vs planted culprit set: %.3f"
                 % agg["minimal_set_exact_rate"])
    lines.append("- Mean set F1: %.3f" % agg["minimal_set_mean_f1"])
    lines.append("")
    lines.append("## Source level")
    lines.append("")
    lines.append("- Top source by Shapley contains a culprit: %.3f"
                 % agg["source_shapley_hit_rate"])
    lines.append("- Top source by summed LOBO contains a culprit: %.3f"
                 % agg["source_lobo_hit_rate"])
    if "clean_control" in results:
        clean = results["clean_control"]
        lines.append("")
        lines.append("## Clean control (no planted culprits, %d streams)"
                     % clean["n_streams"])
        lines.append("")
        lines.append("- Flagged as drifted at the 2.5 x null band: %.3f"
                     % clean["flagged_rate"])
        lines.append("- Minimal blame set empty: %.3f" % clean["empty_set_rate"])
        lines.append("- Mean minimal blame set size: %.2f"
                     % clean["mean_minimal_set_size"])
    lines.append("")
    return "\n".join(lines)


def headline(results):
    agg = results["aggregate"]
    lobo = agg["methods"]["lobo"]
    shap = agg["methods"]["shapley"]
    psi = agg["methods"]["psi_mapped"]
    own = agg["methods"]["own_drift"]
    lines = []
    lines.append("# Headline results")
    lines.append("")
    lines.append(
        "Across %d seeded scenarios (mean %.1f planted culprit batches "
        "per scenario, shift types: mean / label-rate / schema-value):"
        % (agg["n_scenarios"], agg["mean_culprits"]))
    lines.append("")
    lines.append("- LOBO blame: precision@k %.3f, recall@k %.3f"
                 % (lobo["precision_at_k"], lobo["recall_at_k"]))
    lines.append("- Shapley blame: precision@k %.3f, recall@k %.3f"
                 % (shap["precision_at_k"], shap["recall_at_k"]))
    lines.append("- PSI-mapped baseline: precision@k %.3f, recall@k %.3f"
                 % (psi["precision_at_k"], psi["recall_at_k"]))
    lines.append("- Batch own-drift baseline: precision@k %.3f, recall@k %.3f"
                 % (own["precision_at_k"], own["recall_at_k"]))
    lines.append("- Greedy minimal blame set: exact match %.3f, mean F1 %.3f"
                 % (agg["minimal_set_exact_rate"], agg["minimal_set_mean_f1"]))
    lines.append("- Top source (Shapley / LOBO) contains a culprit: %.3f / %.3f"
                 % (agg["source_shapley_hit_rate"], agg["source_lobo_hit_rate"]))
    lines.append("")
    lines.append("All numbers are measured on synthetic seeded streams; "
                 "see README section 7 for limits.")
    lines.append("")
    return "\n".join(lines)
