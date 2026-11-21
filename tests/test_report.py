from drift_blame.blame import greedy_minimal_set, lobo_blame, pooled_counts, shapley_blame
from drift_blame.drift import Bins, count_rows, drift_report
from drift_blame.evaluate import run_evaluation
from drift_blame.generate import demo_scenario, generate_stream
from drift_blame.models import Culprit, Scenario, SHIFT_MEAN
from drift_blame.report import evaluation_summary, headline, stream_report


def _demo_report():
    stream = generate_stream(demo_scenario())
    bins = Bins(stream.reference_rows)
    ref = count_rows(bins, stream.reference_rows)
    batch_counts = [count_rows(bins, b.rows) for b in stream.batches]
    report = drift_report(ref, pooled_counts(batch_counts))
    lobo = lobo_blame(ref, batch_counts)
    shapley = shapley_blame(ref, batch_counts, n_permutations=16, seed=42)
    removed, trace = greedy_minimal_set(ref, batch_counts)
    return stream_report(stream, ref, batch_counts, report, lobo, shapley,
                         removed, trace)


def test_stream_report_mentions_batches_and_sections():
    text = _demo_report()
    assert "## Batch blame" in text
    assert "## Minimal blame set" in text
    assert "b04" in text
    assert "b08" in text
    assert "Removed in order: b04, b08" in text


def test_stream_report_has_feature_table():
    text = _demo_report()
    assert "glucose_mg_dl" in text
    assert "readmitted" in text


def _small_results():
    scenario = Scenario(
        seed=301, n_batches=5, batch_size=200, ref_size=1500,
        culprits=[Culprit(batch_index=2, shift_type=SHIFT_MEAN,
                          feature="glucose_mg_dl", magnitude=2.1)],
    )
    return run_evaluation(scenarios=[scenario], shapley_permutations=16,
                          clean_seeds=[202])


def test_evaluation_summary_sections():
    text = evaluation_summary(_small_results())
    assert "## Culprit ranking" in text
    assert "## Minimal blame set" in text
    assert "## Clean control" in text
    assert "psi_mapped" in text


def test_headline_mentions_all_methods():
    text = headline(_small_results())
    for token in ["LOBO", "Shapley", "PSI-mapped", "own-drift", "minimal"]:
        assert token in text
