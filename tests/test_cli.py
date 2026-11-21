import io
import json
import os

from drift_blame.cli import main


def test_demo_command_prints_report(capsys):
    code = main(["demo"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Batch blame" in out
    assert "b04" in out
    assert "Planted culprits" in out


def test_evaluate_command_writes_artifacts(tmp_path):
    out_dir = str(tmp_path / "results")
    code = main(["evaluate", "--out", out_dir])
    assert code == 0
    for name in ["evaluation.json", "evaluation.csv",
                 "evaluation_summary.md", "HEADLINE.md"]:
        assert os.path.exists(os.path.join(out_dir, name))
    with io.open(os.path.join(out_dir, "evaluation.json"),
                 encoding="utf-8") as fh:
        results = json.load(fh)
    assert results["aggregate"]["n_scenarios"] == 16
    assert "clean_control" in results


def test_no_command_returns_usage_error(capsys):
    code = main([])
    assert code == 2
