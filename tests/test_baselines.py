from drift_blame.baselines import (
    feature_psi,
    own_drift_ranking,
    psi_mapped_ranking,
    ranking_from_scores,
)
from drift_blame.blame import pooled_counts
from drift_blame.drift import Bins, count_rows
from drift_blame.generate import demo_scenario, generate_stream
from drift_blame.models import LABEL


def _demo_counts():
    stream = generate_stream(demo_scenario())
    bins = Bins(stream.reference_rows)
    ref = count_rows(bins, stream.reference_rows)
    batch_counts = [count_rows(bins, b.rows) for b in stream.batches]
    return stream, ref, batch_counts


def test_feature_psi_excludes_label():
    stream, ref, batch_counts = _demo_counts()
    psi = feature_psi(ref, pooled_counts(batch_counts))
    assert LABEL not in psi
    assert "glucose_mg_dl" in psi
    assert all(value >= 0.0 for value in psi.values())


def test_psi_top_feature_is_the_shifted_one():
    stream, ref, batch_counts = _demo_counts()
    ranking, top_feature, scores = psi_mapped_ranking(
        ref, batch_counts, pooled_counts(batch_counts)
    )
    assert top_feature == "glucose_mg_dl"
    assert sorted(ranking) == list(range(len(batch_counts)))
    assert len(scores) == len(batch_counts)


def test_psi_mapped_ranking_is_a_permutation():
    stream, ref, batch_counts = _demo_counts()
    ranking, _top, _scores = psi_mapped_ranking(
        ref, batch_counts, pooled_counts(batch_counts)
    )
    assert sorted(ranking) == list(range(10))


def test_own_drift_finds_the_mean_shift_culprit():
    stream, ref, batch_counts = _demo_counts()
    ranking, scores = own_drift_ranking(ref, batch_counts)
    top_batch = stream.batches[ranking[0]].batch_id
    # The glucose-shifted batch drifts most against the reference on
    # its own; the label-shifted batch is second or close behind.
    assert top_batch == "b04"
    top_three = set(stream.batches[i].batch_id for i in ranking[:3])
    assert "b08" in top_three


def test_ranking_from_scores_orders_descending():
    ranking = ranking_from_scores([0.1, 0.5, -0.2, 0.5])
    assert ranking[0] in (1, 3)
    assert ranking[-1] == 2
    assert sorted(ranking) == [0, 1, 2, 3]
