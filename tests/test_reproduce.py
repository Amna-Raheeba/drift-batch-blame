import importlib.util
import io
import json
import os


def _load_reproduce_module():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "scripts", "reproduce_results.py")
    spec = importlib.util.spec_from_file_location("reproduce_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_results(lobo_precision):
    return {
        "aggregate": {
            "minimal_set_exact_rate": 0.5,
            "minimal_set_mean_f1": 0.7,
            "source_shapley_hit_rate": 1.0,
            "source_lobo_hit_rate": 0.9,
            "methods": {
                "lobo": {"precision_at_k": lobo_precision, "recall_at_k": 0.9},
                "shapley": {"precision_at_k": 1.0, "recall_at_k": 1.0},
                "psi_mapped": {"precision_at_k": 0.7, "recall_at_k": 0.7},
                "own_drift": {"precision_at_k": 1.0, "recall_at_k": 1.0},
            },
        },
        "clean_control": {"flagged_rate": 0.0, "empty_set_rate": 1.0},
    }


def test_compare_identical_results_passes():
    module = _load_reproduce_module()
    ok, diffs = module.compare_results(_fake_results(0.95), _fake_results(0.95))
    assert ok is True
    assert diffs == []


def test_compare_detects_drift_beyond_tolerance():
    module = _load_reproduce_module()
    ok, diffs = module.compare_results(_fake_results(0.95), _fake_results(0.80))
    assert ok is False
    assert any("lobo_precision_at_k" in d for d in diffs)


def test_compare_within_tolerance_passes():
    module = _load_reproduce_module()
    ok, diffs = module.compare_results(_fake_results(0.95), _fake_results(0.94))
    assert ok is True


def test_shipped_results_exist_and_parse():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "results", "evaluation.json")
    with io.open(path, "r", encoding="utf-8") as fh:
        results = json.load(fh)
    module = _load_reproduce_module()
    values = module.headline_values(results)
    assert values["shapley_precision_at_k"] >= 0.0
    assert "clean_flagged_rate" in values
