from drift_blame.drift import (
    N_BINS,
    RESTORE_RATIO,
    Bins,
    Counts,
    count_rows,
    drift_report,
    drift_score,
    feature_stats,
)
from drift_blame.generate import generate_stream
from drift_blame.models import Culprit, Scenario, SHIFT_MEAN


def _small_stream(seed=5):
    return generate_stream(Scenario(
        seed=seed, n_batches=4, batch_size=100, ref_size=800, culprits=[],
    ))


def test_bins_edges_increasing_and_bounded():
    stream = _small_stream()
    bins = Bins(stream.reference_rows)
    for feat, edges in bins.edges.items():
        assert len(edges) <= N_BINS - 1
        assert all(b > a for a, b in zip(edges, edges[1:]))


def test_counts_total_matches_rows():
    stream = _small_stream()
    bins = Bins(stream.reference_rows)
    counts = count_rows(bins, stream.reference_rows)
    assert counts.n == 800
    for feat in counts.numeric:
        assert sum(counts.numeric[feat]) == 800
    assert sum(counts.categorical) == 800
    assert sum(counts.label) == 800


def test_counts_merge_and_minus_roundtrip():
    stream = _small_stream()
    bins = Bins(stream.reference_rows)
    first = count_rows(bins, stream.batches[0].rows)
    second = count_rows(bins, stream.batches[1].rows)
    merged = first.merged(second)
    assert merged.n == first.n + second.n
    back = merged.minus(second)
    assert back.n == first.n
    for feat in first.numeric:
        assert back.numeric[feat] == first.numeric[feat]
    assert back.categorical == first.categorical
    assert back.label == first.label


def test_identical_counts_have_zero_drift():
    stream = _small_stream()
    bins = Bins(stream.reference_rows)
    ref = count_rows(bins, stream.reference_rows)
    report = drift_report(ref, ref)
    assert report["score"] == 0.0
    assert report["max_ratio"] == 0.0
    assert report["restored"] is True


def test_empty_pool_is_restored_with_zero_score():
    stream = _small_stream()
    bins = Bins(stream.reference_rows)
    ref = count_rows(bins, stream.reference_rows)
    empty = Counts(bins)
    assert drift_score(ref, empty) == 0.0
    assert drift_report(ref, empty)["restored"] is True


def test_shifted_pool_exceeds_band_on_shifted_feature():
    stream = generate_stream(Scenario(
        seed=9, culprits=[Culprit(batch_index=2, shift_type=SHIFT_MEAN,
                                  feature="glucose_mg_dl", magnitude=1.8)],
    ))
    bins = Bins(stream.reference_rows)
    ref = count_rows(bins, stream.reference_rows)
    pooled = Counts(bins)
    for batch in stream.batches:
        pooled = pooled.merged(count_rows(bins, batch.rows))
    stats = feature_stats(ref, pooled)
    stat, null, ratio = stats["glucose_mg_dl"]
    assert ratio > RESTORE_RATIO
    report = drift_report(ref, pooled)
    assert report["restored"] is False
    assert report["score"] > 0.0


def test_unseen_category_lands_in_other_bucket():
    stream = _small_stream()
    bins = Bins(stream.reference_rows)
    rows = [dict(r) for r in stream.reference_rows[:10]]
    for row in rows:
        row["device_model"] = "ZZZ"
    counts = count_rows(bins, rows)
    assert counts.categorical[-1] == 10
    assert sum(counts.categorical) == 10


def test_feature_stats_cover_all_features():
    stream = _small_stream()
    bins = Bins(stream.reference_rows)
    ref = count_rows(bins, stream.reference_rows)
    pooled = count_rows(bins, stream.batches[0].rows)
    stats = feature_stats(ref, pooled)
    assert sorted(stats.keys()) == sorted(
        ["age", "systolic_bp", "glucose_mg_dl", "creatinine", "bmi",
         "device_model", "readmitted"]
    )
    for stat, null, ratio in stats.values():
        assert stat >= 0.0
        assert null > 0.0
        assert ratio >= 0.0
