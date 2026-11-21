"""Notebook 02 -- evaluation experiment, reduced smoke run.

Runs the first six scenarios of the canonical suite plus two clean
control streams and prints the summary.  The full 16-scenario
evaluation that produced results/ is:

    python -m drift_blame evaluate --out results

Run from the repository root:  python notebooks/02_evaluation_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_blame.evaluate import run_evaluation
from drift_blame.generate import default_scenarios
from drift_blame.report import evaluation_summary


def main():
    scenarios = default_scenarios()[:6]
    results = run_evaluation(scenarios=scenarios, clean_seeds=[201, 202])
    print(evaluation_summary(results))
    print("(Reduced run: 6 of 16 scenarios. Full numbers in results/.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
