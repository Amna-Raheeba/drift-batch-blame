# Headline results

Across 16 seeded scenarios (mean 1.9 planted culprit batches per scenario, shift types: mean / label-rate / schema-value):

- LOBO blame: precision@k 0.969, recall@k 0.969
- Shapley blame: precision@k 1.000, recall@k 1.000
- PSI-mapped baseline: precision@k 0.708, recall@k 0.708
- Batch own-drift baseline: precision@k 1.000, recall@k 1.000
- Greedy minimal blame set: exact match 0.625, mean F1 0.767
- Top source (Shapley / LOBO) contains a culprit: 1.000 / 0.938

All numbers are measured on synthetic seeded streams; see README section 7 for limits.
