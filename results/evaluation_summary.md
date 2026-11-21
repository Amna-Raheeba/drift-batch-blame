# Evaluation summary

Scenarios: 16 (mean planted culprits per scenario: 1.94)

## Culprit ranking (k = number of planted culprits)

| method | precision@k | recall@k |
|---|---|---|
| lobo | 0.969 | 0.969 |
| shapley | 1.000 | 1.000 |
| psi_mapped | 0.708 | 0.708 |
| own_drift | 1.000 | 1.000 |

## Minimal blame set

- Exact-match rate vs planted culprit set: 0.625
- Mean set F1: 0.767

## Source level

- Top source by Shapley contains a culprit: 1.000
- Top source by summed LOBO contains a culprit: 0.938

## Clean control (no planted culprits, 8 streams)

- Flagged as drifted at the 2.5 x null band: 0.000
- Minimal blame set empty: 1.000
- Mean minimal blame set size: 0.00
