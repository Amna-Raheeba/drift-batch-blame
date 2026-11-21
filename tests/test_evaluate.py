from drift_blame.evaluate import (
    evaluate_clean_control,
    evaluate_scenario,
    run_evaluation,
)
from drift_blame.models import Culprit, Scenario, SHIFT_MEAN


def _small_scenario(seed=301):
    return Scenario(
        seed=seed, n_batches=5, batch_size=200, ref_size=1500,
        culprits=[Culprit(batch_index=2, shift_type=SHIFT_MEAN,
                          feature="glucose_mg_dl", magnitude=2.1)],
    )


def test_evaluate_scenario_structure():
    result = evaluate_scenario(_small_scenario(), shapley_permutations=16)
    assert result["n_culprits"] == 1
    assert result["culprit_batches"] == ["b03"]
    for name in ["lobo", "shapley", "psi_mapped", "own_drift"]:
        metrics = result["methods"][name]
        assert 0.0 <= metrics["precision_at_k"] <= 1.0
        assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert "exact_match" in result["minimal_set"]
    assert 0.0 <= result["minimal_set"]["f1"] <= 1.0
    assert result["pooled_max_ratio"] > 0.0


def test_evaluate_scenario_finds_strong_culprit():
    result = evaluate_scenario(_small_scenario(), shapley_permutations=16)
    assert result["methods"]["lobo"]["recall_at_k"] == 1.0
    assert result["minimal_set"]["exact_match"] is True


def test_clean_control_structure():
    result = evaluate_clean_control(202)
    assert result["initially_restored"] is True
    assert result["minimal_set_size"] == 0


def test_run_evaluation_aggregates():
    results = run_evaluation(
        scenarios=[_small_scenario(301), _small_scenario(302)],
        shapley_permutations=16,
        clean_seeds=[202],
    )
    agg = results["aggregate"]
    assert agg["n_scenarios"] == 2
    assert len(results["scenarios"]) == 2
    assert results["clean_control"]["n_streams"] == 1
    for name in ["lobo", "shapley", "psi_mapped", "own_drift"]:
        assert 0.0 <= agg["methods"][name]["precision_at_k"] <= 1.0
    assert 0.0 <= agg["minimal_set_exact_rate"] <= 1.0
    assert 0.0 <= agg["minimal_set_mean_f1"] <= 1.0
