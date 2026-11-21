"""Batch-level blame assignment.

The unit of analysis is the ingestion batch (and, aggregated, the
source) -- never the feature.  Three complementary attributions:

- **Leave-one-batch-out (LOBO) blame**: the drop in pooled drift
  score when the batch is removed.  Clean batches typically score
  at or below zero (removing them concentrates the drift); culprit
  batches carry the excess.
- **Shapley-style blame**: each batch's average marginal drift
  contribution over seeded random orderings of the stream, so blame
  is shared fairly when two culprit batches interact.  Source-level
  Shapley is exact (few sources, all subsets enumerated).
- **Greedy minimal blame set**: repeatedly remove the highest-blame
  batch (recomputing LOBO each round) until the pooled remainder is
  restored -- every feature back within its noise band -- or no
  removal reduces the drift further.  The removed set is the
  smallest repair the greedy pass can certify.
"""

import random

from .drift import Counts, drift_report, drift_score


def pooled_counts(batch_counts, indices=None):
    """Sum the counts of the selected batches (default: all)."""
    if indices is None:
        indices = range(len(batch_counts))
    total = None
    for i in indices:
        if total is None:
            total = Counts(batch_counts[i].bins)
        total = total.merged(batch_counts[i])
    if total is None:
        total = Counts(batch_counts[0].bins)
    return total


def lobo_blame(ref_counts, batch_counts):
    """Drift-score drop when each batch is removed, one at a time."""
    total = pooled_counts(batch_counts)
    full = drift_score(ref_counts, total)
    blame = []
    for i in range(len(batch_counts)):
        rest = total.minus(batch_counts[i])
        blame.append(full - drift_score(ref_counts, rest))
    return blame


def shapley_blame(ref_counts, batch_counts, n_permutations=64, seed=7):
    """Sampled Shapley blame over batches (seeded permutations)."""
    n = len(batch_counts)
    rng = random.Random(seed)
    totals = [0.0] * n
    for _ in range(n_permutations):
        order = list(range(n))
        rng.shuffle(order)
        running = Counts(batch_counts[0].bins)
        prev = 0.0
        for idx in order:
            running = running.merged(batch_counts[idx])
            current = drift_score(ref_counts, running)
            totals[idx] += current - prev
            prev = current
    return [t / float(n_permutations) for t in totals]


def source_shapley(ref_counts, batch_counts, batch_sources):
    """Exact Shapley blame over sources (all subsets enumerated)."""
    sources = sorted(set(batch_sources))
    n_sources = len(sources)
    totals = dict((s, 0.0) for s in sources)

    def subset_score(source_subset):
        indices = [
            i for i, s in enumerate(batch_sources) if s in source_subset
        ]
        if not indices:
            return 0.0
        return drift_score(ref_counts, pooled_counts(batch_counts, indices))

    n_fact = 1
    for k in range(2, n_sources + 1):
        n_fact *= k
    fact = [1] * (n_sources + 1)
    for k in range(1, n_sources + 1):
        fact[k] = fact[k - 1] * k

    for mask in range(1 << n_sources):
        subset = set(sources[i] for i in range(n_sources) if mask & (1 << i))
        base = subset_score(subset)
        for s in sources:
            if s in subset:
                continue
            with_s = subset_score(subset | {s})
            weight = (
                fact[len(subset)] * fact[n_sources - len(subset) - 1]
                / float(fact[n_sources])
            )
            totals[s] += weight * (with_s - base)
    return totals


def greedy_minimal_set(ref_counts, batch_counts, min_gain_fraction=0.02):
    """Greedy removal pass.

    Returns (removed_indices_in_order, trace).  Trace entries are
    (removed_index, score, max_ratio, restored): the first entry is
    the initial pooled state with removed_index None, and each later
    entry is the pooled state after that removal.
    """
    remaining = list(range(len(batch_counts)))
    removed = []
    trace = []
    report = drift_report(ref_counts, pooled_counts(batch_counts, remaining))
    trace.append((None, report["score"], report["max_ratio"], report["restored"]))
    initial_score = report["score"]
    while not report["restored"] and len(remaining) > 1:
        sub_counts = [batch_counts[i] for i in remaining]
        blame = lobo_blame(ref_counts, sub_counts)
        best_pos = max(range(len(remaining)), key=lambda p: blame[p])
        best_gain = blame[best_pos]
        if initial_score > 0 and best_gain <= min_gain_fraction * initial_score:
            break
        if best_gain <= 0.0:
            break
        removed.append(remaining[best_pos])
        remaining.pop(best_pos)
        report = drift_report(ref_counts, pooled_counts(batch_counts, remaining))
        trace.append(
            (removed[-1], report["score"], report["max_ratio"], report["restored"])
        )
    return removed, trace
