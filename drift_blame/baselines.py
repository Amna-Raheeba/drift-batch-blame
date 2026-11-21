"""Baselines the blame methods are compared against.

1. **Per-feature PSI mapped back to batches** (the standard drift
   monitor's answer): compute pooled per-feature PSI over the input
   features, take the single most-drifted feature, then rank batches
   by that feature's batch-level PSI against the reference.  This is
   indirect by construction: it can only see one feature, it never
   looks at the label, and a feature shared by many batches smears
   the blame across all of them.
2. **Batch own-drift**: rank each batch by its own excess drift
   against the reference, ignoring the pool.  A small culprit batch
   is statistically noisy on its own; the pooled replay methods are
   not.
"""

import math

from .drift import drift_score
from .models import CATEGORICAL_FEATURE, NUMERIC_FEATURES

_EPS = 1e-4


def _proportions(counts_vec):
    total = float(sum(counts_vec))
    if total == 0:
        return [0.0] * len(counts_vec)
    return [c / total for c in counts_vec]


def _psi(ref_vec, pool_vec):
    ref_p = _proportions(ref_vec)
    pool_p = _proportions(pool_vec)
    total = 0.0
    for p, q in zip(ref_p, pool_p):
        p = max(p, _EPS)
        q = max(q, _EPS)
        total += (p - q) * math.log(p / q)
    return total


def feature_psi(ref_counts, pool_counts):
    """Pooled PSI per input feature (label excluded, as in practice)."""
    out = {}
    for feat in NUMERIC_FEATURES:
        out[feat] = _psi(ref_counts.numeric[feat], pool_counts.numeric[feat])
    out[CATEGORICAL_FEATURE] = _psi(ref_counts.categorical, pool_counts.categorical)
    return out


def _feature_counts(counts, feat):
    if feat == CATEGORICAL_FEATURE:
        return counts.categorical
    return counts.numeric[feat]


def psi_mapped_ranking(ref_counts, batch_counts, pool_counts):
    """Rank batch indices by PSI on the pooled most-drifted feature."""
    psi = feature_psi(ref_counts, pool_counts)
    top_feature = max(psi, key=lambda f: psi[f])
    ref_vec = _feature_counts(ref_counts, top_feature)
    scores = []
    for counts in batch_counts:
        scores.append(_psi(ref_vec, _feature_counts(counts, top_feature)))
    ranking = sorted(range(len(batch_counts)), key=lambda i: -scores[i])
    return ranking, top_feature, scores


def own_drift_ranking(ref_counts, batch_counts):
    """Rank batch indices by each batch's own excess drift score."""
    scores = [drift_score(ref_counts, counts) for counts in batch_counts]
    ranking = sorted(range(len(batch_counts)), key=lambda i: -scores[i])
    return ranking, scores


def ranking_from_scores(scores):
    return sorted(range(len(scores)), key=lambda i: -scores[i])
