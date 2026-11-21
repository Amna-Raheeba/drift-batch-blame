"""Drift measurement over binned counts.

Rows are reduced to per-feature bin counts once; every pooled,
leave-one-out or subset drift evaluation is then arithmetic on
count vectors, which makes replay cheap and exact.

Per feature the statistic is standardised by its expected value
under the no-drift null at the given sample sizes:

- numeric features: binned Kolmogorov-Smirnov distance (maximum gap
  between reference and pooled empirical CDFs over reference
  quantile bins); null mean uses the Kolmogorov approximation
  0.868 * sqrt(1/n_ref + 1/n_pool).
- categorical and label: total variation distance; null mean from
  the normal approximation to per-bin proportion noise.

The headline *drift score* of a pooled set is its total excess
drift: sum over features of max(0, statistic - null).  A pooled set
is *restored* when every feature's statistic is within
RESTORE_RATIO x its null expectation.
"""

import bisect
import math

from .models import ALL_FEATURES, CATEGORIES, LABEL, NUMERIC_FEATURES

N_BINS = 25
RESTORE_RATIO = 2.5
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_KS_NULL_COEF = 0.868


class Bins(object):
    """Reference quantile bin edges per numeric feature."""

    def __init__(self, reference_rows):
        self.edges = {}
        for feat in NUMERIC_FEATURES:
            values = sorted(r[feat] for r in reference_rows)
            edges = []
            for k in range(1, N_BINS):
                idx = int(round(k * len(values) / float(N_BINS)))
                idx = min(max(idx, 1), len(values) - 1)
                edges.append(values[idx])
            # Deduplicate while keeping order.
            deduped = []
            for e in edges:
                if not deduped or e > deduped[-1]:
                    deduped.append(e)
            self.edges[feat] = deduped
        self.cat_values = list(CATEGORIES) + ["__OTHER__"]

    def n_bins(self, feat):
        return len(self.edges[feat]) + 1


class Counts(object):
    """Per-feature bin counts for a set of rows."""

    def __init__(self, bins):
        self.bins = bins
        self.n = 0
        self.numeric = dict(
            (f, [0] * bins.n_bins(f)) for f in NUMERIC_FEATURES
        )
        self.categorical = [0] * len(bins.cat_values)
        self.label = [0, 0]

    def add_row(self, row):
        self.n += 1
        for feat in NUMERIC_FEATURES:
            idx = bisect.bisect_right(self.bins.edges[feat], row[feat])
            self.numeric[feat][idx] += 1
        cat = row["device_model"]
        if cat in self.bins.cat_values:
            self.categorical[self.bins.cat_values.index(cat)] += 1
        else:
            self.categorical[-1] += 1
        self.label[int(row[LABEL])] += 1

    def merged(self, other):
        out = Counts(self.bins)
        out.n = self.n + other.n
        for feat in NUMERIC_FEATURES:
            out.numeric[feat] = [
                a + b for a, b in zip(self.numeric[feat], other.numeric[feat])
            ]
        out.categorical = [a + b for a, b in zip(self.categorical, other.categorical)]
        out.label = [a + b for a, b in zip(self.label, other.label)]
        return out

    def minus(self, other):
        out = Counts(self.bins)
        out.n = self.n - other.n
        for feat in NUMERIC_FEATURES:
            out.numeric[feat] = [
                a - b for a, b in zip(self.numeric[feat], other.numeric[feat])
            ]
        out.categorical = [a - b for a, b in zip(self.categorical, other.categorical)]
        out.label = [a - b for a, b in zip(self.label, other.label)]
        return out


def count_rows(bins, rows):
    counts = Counts(bins)
    for row in rows:
        counts.add_row(row)
    return counts


def _tv_and_null(ref_counts_vec, pool_counts_vec, n_ref, n_pool):
    total_ref = float(sum(ref_counts_vec))
    total_pool = float(sum(pool_counts_vec))
    if total_ref == 0 or total_pool == 0:
        return 0.0, 0.0
    k = len(ref_counts_vec)
    tv = 0.0
    null = 0.0
    for i in range(k):
        p_raw = ref_counts_vec[i] / total_ref
        q = pool_counts_vec[i] / total_pool
        tv += abs(p_raw - q)
        # Smoothed reference proportion for the null scale only, so
        # unseen categories still get a finite noise band.
        p = (ref_counts_vec[i] + 0.5) / (total_ref + 0.5 * k)
        var = p * (1.0 - p) * (1.0 / total_ref + 1.0 / total_pool)
        null += _SQRT_2_OVER_PI * math.sqrt(max(var, 0.0))
    return 0.5 * tv, 0.5 * null


def _ks_and_null(ref_counts_vec, pool_counts_vec, n_ref, n_pool):
    total_ref = float(sum(ref_counts_vec))
    total_pool = float(sum(pool_counts_vec))
    if total_ref == 0 or total_pool == 0:
        return 0.0, 0.0
    ks = 0.0
    cum_ref = 0.0
    cum_pool = 0.0
    for i in range(len(ref_counts_vec)):
        cum_ref += ref_counts_vec[i] / total_ref
        cum_pool += pool_counts_vec[i] / total_pool
        gap = abs(cum_ref - cum_pool)
        if gap > ks:
            ks = gap
    null = _KS_NULL_COEF * math.sqrt(1.0 / total_ref + 1.0 / total_pool)
    return ks, null


def feature_stats(ref_counts, pool_counts):
    """Per-feature (statistic, null, ratio) for a pooled count set."""
    stats = {}
    for feat in NUMERIC_FEATURES:
        stat, null = _ks_and_null(
            ref_counts.numeric[feat], pool_counts.numeric[feat],
            ref_counts.n, pool_counts.n,
        )
        stats[feat] = (stat, null, stat / null if null > 0 else 0.0)
    stat, null = _tv_and_null(
        ref_counts.categorical, pool_counts.categorical,
        ref_counts.n, pool_counts.n,
    )
    stats["device_model"] = (stat, null, stat / null if null > 0 else 0.0)
    stat, null = _tv_and_null(
        ref_counts.label, pool_counts.label, ref_counts.n, pool_counts.n,
    )
    stats[LABEL] = (stat, null, stat / null if null > 0 else 0.0)
    return stats


def drift_report(ref_counts, pool_counts):
    """Full drift report for one pooled set against the reference."""
    if pool_counts.n == 0:
        return {
            "score": 0.0,
            "max_ratio": 0.0,
            "restored": True,
            "features": {},
        }
    stats = feature_stats(ref_counts, pool_counts)
    score = 0.0
    max_ratio = 0.0
    restored = True
    for feat in ALL_FEATURES:
        stat, null, ratio = stats[feat]
        score += max(0.0, stat - null)
        if ratio > max_ratio:
            max_ratio = ratio
        if ratio > RESTORE_RATIO:
            restored = False
    return {
        "score": score,
        "max_ratio": max_ratio,
        "restored": restored,
        "features": stats,
    }


def drift_score(ref_counts, pool_counts):
    return drift_report(ref_counts, pool_counts)["score"]
