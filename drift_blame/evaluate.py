"""Evaluation: blame methods vs planted culprit ground truth.

For every scenario the stream is generated, every method ranks the
batches, and the rankings are scored against the planted culprits:

- precision@k and recall@k with k = number of planted culprits,
  for LOBO blame, Shapley blame, the PSI-mapped baseline and the
  batch own-drift baseline;
- minimal blame set: exact-match rate and mean set F1 against the
  planted culprit set;
- source level: whether the top source by exact source Shapley (and
  by summed LOBO) contains at least one culprit batch.
"""

from .baselines import own_drift_ranking, psi_mapped_ranking, ranking_from_scores
from .blame import (
    greedy_minimal_set,
    lobo_blame,
    pooled_counts,
    shapley_blame,
    source_shapley,
)
from .drift import Bins, count_rows, drift_report
from .generate import default_scenarios, generate_stream
from .models import Scenario


def _precision_recall_at_k(ranking, culprit_set, k):
    if k == 0:
        return 1.0, 1.0
    top = set(ranking[:k])
    hits = len(top & culprit_set)
    return hits / float(k), hits / float(len(culprit_set))


def _set_f1(predicted, truth):
    if not truth and not predicted:
        return 1.0
    if not truth or not predicted:
        return 0.0
    hits = len(set(predicted) & set(truth))
    if hits == 0:
        return 0.0
    precision = hits / float(len(predicted))
    recall = hits / float(len(truth))
    return 2.0 * precision * recall / (precision + recall)


def evaluate_scenario(scenario, shapley_permutations=64):
    stream = generate_stream(scenario)
    bins = Bins(stream.reference_rows)
    ref_counts = count_rows(bins, stream.reference_rows)
    batch_counts = [count_rows(bins, b.rows) for b in stream.batches]
    pool = pooled_counts(batch_counts)
    report = drift_report(ref_counts, pool)

    culprit_idx = set(
        i for i, b in enumerate(stream.batches) if b.batch_id in stream.truth
    )
    k = len(culprit_idx)

    lobo = lobo_blame(ref_counts, batch_counts)
    shapley = shapley_blame(
        ref_counts, batch_counts, n_permutations=shapley_permutations,
        seed=scenario.seed,
    )
    psi_ranking, psi_top_feature, _psi_scores = psi_mapped_ranking(
        ref_counts, batch_counts, pool
    )
    own_ranking, _own_scores = own_drift_ranking(ref_counts, batch_counts)

    removed, _trace = greedy_minimal_set(ref_counts, batch_counts)
    removed_set = set(removed)

    sources = [b.source for b in stream.batches]
    src_shapley = source_shapley(ref_counts, batch_counts, sources)
    top_source_shapley = max(src_shapley, key=lambda s: src_shapley[s])
    lobo_by_source = {}
    for i, s in enumerate(sources):
        lobo_by_source[s] = lobo_by_source.get(s, 0.0) + lobo[i]
    top_source_lobo = max(lobo_by_source, key=lambda s: lobo_by_source[s])
    culprit_sources = set(sources[i] for i in culprit_idx)

    methods = {}
    for name, ranking in [
        ("lobo", ranking_from_scores(lobo)),
        ("shapley", ranking_from_scores(shapley)),
        ("psi_mapped", psi_ranking),
        ("own_drift", own_ranking),
    ]:
        p, r = _precision_recall_at_k(ranking, culprit_idx, k)
        methods[name] = {"precision_at_k": p, "recall_at_k": r}

    return {
        "seed": scenario.seed,
        "n_culprits": k,
        "culprit_batches": sorted(
            stream.batches[i].batch_id for i in culprit_idx
        ),
        "culprit_shift_types": sorted(set(stream.truth.values())),
        "pooled_score": report["score"],
        "pooled_max_ratio": report["max_ratio"],
        "pooled_restored_initially": report["restored"],
        "psi_top_feature": psi_top_feature,
        "methods": methods,
        "minimal_set": {
            "batches": sorted(stream.batches[i].batch_id for i in removed_set),
            "exact_match": removed_set == culprit_idx,
            "f1": _set_f1(removed_set, culprit_idx),
        },
        "source": {
            "top_shapley": top_source_shapley,
            "top_lobo": top_source_lobo,
            "shapley_hit": top_source_shapley in culprit_sources,
            "lobo_hit": top_source_lobo in culprit_sources,
        },
    }


def evaluate_clean_control(seed):
    """A stream with no planted culprits: the audit should stay quiet."""
    stream = generate_stream(Scenario(seed=seed, culprits=[]))
    bins = Bins(stream.reference_rows)
    ref_counts = count_rows(bins, stream.reference_rows)
    batch_counts = [count_rows(bins, b.rows) for b in stream.batches]
    report = drift_report(ref_counts, pooled_counts(batch_counts))
    lobo = lobo_blame(ref_counts, batch_counts)
    removed, _trace = greedy_minimal_set(ref_counts, batch_counts)
    return {
        "seed": seed,
        "initially_restored": report["restored"],
        "pooled_max_ratio": report["max_ratio"],
        "minimal_set_size": len(removed),
        "top_lobo_blame": max(lobo) if lobo else 0.0,
    }


def run_evaluation(scenarios=None, shapley_permutations=64, clean_seeds=None):
    if scenarios is None:
        scenarios = default_scenarios()
    if clean_seeds is None:
        clean_seeds = list(range(201, 209))
    per_scenario = [evaluate_scenario(s, shapley_permutations) for s in scenarios]
    n = float(len(per_scenario))

    def mean(values):
        return sum(values) / n

    aggregate = {
        "n_scenarios": len(per_scenario),
        "mean_culprits": mean([s["n_culprits"] for s in per_scenario]),
        "methods": {},
        "minimal_set_exact_rate": mean(
            [1.0 if s["minimal_set"]["exact_match"] else 0.0 for s in per_scenario]
        ),
        "minimal_set_mean_f1": mean(
            [s["minimal_set"]["f1"] for s in per_scenario]
        ),
        "source_shapley_hit_rate": mean(
            [1.0 if s["source"]["shapley_hit"] else 0.0 for s in per_scenario]
        ),
        "source_lobo_hit_rate": mean(
            [1.0 if s["source"]["lobo_hit"] else 0.0 for s in per_scenario]
        ),
    }
    for name in ["lobo", "shapley", "psi_mapped", "own_drift"]:
        aggregate["methods"][name] = {
            "precision_at_k": mean(
                [s["methods"][name]["precision_at_k"] for s in per_scenario]
            ),
            "recall_at_k": mean(
                [s["methods"][name]["recall_at_k"] for s in per_scenario]
            ),
        }

    clean = [evaluate_clean_control(seed) for seed in clean_seeds]
    m = float(len(clean))
    clean_block = {
        "n_streams": len(clean),
        "flagged_rate": sum(
            1.0 for c in clean if not c["initially_restored"]
        ) / m,
        "empty_set_rate": sum(
            1.0 for c in clean if c["minimal_set_size"] == 0
        ) / m,
        "mean_minimal_set_size": sum(c["minimal_set_size"] for c in clean) / m,
        "streams": clean,
    }
    return {
        "aggregate": aggregate,
        "scenarios": per_scenario,
        "clean_control": clean_block,
    }
