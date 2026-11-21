from drift_blame.blame import (
    greedy_minimal_set,
    lobo_blame,
    pooled_counts,
    shapley_blame,
    source_shapley,
)
from drift_blame.drift import Bins, count_rows, drift_report, drift_score
from drift_blame.generate import demo_scenario, generate_stream
from drift_blame.models import Scenario


def _demo_counts():
    stream = generate_stream(demo_scenario())
    bins = Bins(stream.reference_rows)
    ref = count_rows(bins, stream.reference_rows)
    batch_counts = [count_rows(bins, b.rows) for b in stream.batches]
    return stream, ref, batch_counts


def test_pooled_counts_matches_sum():
    stream, ref, batch_counts = _demo_counts()
    pooled = pooled_counts(batch_counts)
    assert pooled.n == sum(b.size for b in stream.batches)
    subset = pooled_counts(batch_counts, [0, 1])
    assert subset.n == stream.batches[0].size + stream.batches[1].size


def test_lobo_ranks_culprits_top_two():
    stream, ref, batch_counts = _demo_counts()
    blame = lobo_blame(ref, batch_counts)
    order = sorted(range(len(blame)), key=lambda i: -blame[i])
    top_two = set(stream.batches[i].batch_id for i in order[:2])
    assert top_two == set(stream.culprit_ids)


def test_lobo_clean_batches_near_or_below_zero():
    stream, ref, batch_counts = _demo_counts()
    blame = lobo_blame(ref, batch_counts)
    for i, batch in enumerate(stream.batches):
        if batch.batch_id not in stream.truth:
            assert blame[i] <= 0.02


def test_shapley_sums_to_pooled_score():
    stream, ref, batch_counts = _demo_counts()
    shapley = shapley_blame(ref, batch_counts, n_permutations=32, seed=1)
    total = drift_score(ref, pooled_counts(batch_counts))
    assert abs(sum(shapley) - total) < 0.01


def test_shapley_ranks_culprits_top_two():
    stream, ref, batch_counts = _demo_counts()
    shapley = shapley_blame(ref, batch_counts, n_permutations=32, seed=1)
    order = sorted(range(len(shapley)), key=lambda i: -shapley[i])
    top_two = set(stream.batches[i].batch_id for i in order[:2])
    assert top_two == set(stream.culprit_ids)


def test_source_shapley_sums_to_pooled_score():
    stream, ref, batch_counts = _demo_counts()
    sources = [b.source for b in stream.batches]
    result = source_shapley(ref, batch_counts, sources)
    total = drift_score(ref, pooled_counts(batch_counts))
    assert abs(sum(result.values()) - total) < 0.01
    assert sorted(result.keys()) == sorted(set(sources))


def test_source_shapley_top_source_holds_a_culprit():
    stream, ref, batch_counts = _demo_counts()
    sources = [b.source for b in stream.batches]
    result = source_shapley(ref, batch_counts, sources)
    top = max(result, key=lambda s: result[s])
    culprit_sources = set(
        b.source for b in stream.batches if b.batch_id in stream.truth
    )
    assert top in culprit_sources


def test_greedy_recovers_demo_culprits_and_restores():
    stream, ref, batch_counts = _demo_counts()
    removed, trace = greedy_minimal_set(ref, batch_counts)
    recovered = set(stream.batches[i].batch_id for i in removed)
    assert recovered == set(stream.culprit_ids)
    assert trace[0][0] is None
    assert trace[-1][3] is True
    assert len(trace) == len(removed) + 1


def test_greedy_on_clean_stream_removes_nothing():
    stream = generate_stream(Scenario(seed=201, culprits=[]))
    bins = Bins(stream.reference_rows)
    ref = count_rows(bins, stream.reference_rows)
    batch_counts = [count_rows(bins, b.rows) for b in stream.batches]
    removed, trace = greedy_minimal_set(ref, batch_counts)
    assert removed == []
    report = drift_report(ref, pooled_counts(batch_counts))
    assert report["restored"] is True
