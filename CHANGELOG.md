# Changelog

## 0.1.0 — 2026-11-21

Initial release.

- Seeded synthetic stream generator: reference sample plus
  timestamped batches from four named sources, with planted
  culprit batches carrying mean, label-rate, or schema-value
  shifts, recorded as ground truth.
- Count-based drift measurement: per-feature standardised
  statistics (binned KS for numeric features, total variation for
  categorical and label) with no-drift null expectations, pooled
  excess-drift score, and a 2.5 x null restoration band.
- Batch blame: leave-one-batch-out replay, sampled Shapley blame
  over batches, exact Shapley blame over sources, and a greedy
  minimal blame set with restoration certificate.
- Baselines: per-feature PSI mapped back to batches, and per-batch
  own-drift ranking.
- Evaluation over 16 seeded scenarios plus 8 clean control
  streams: Shapley precision@k / recall@k 1.000, LOBO 0.969,
  PSI-mapped baseline 0.708; minimal blame set exact match 0.625
  (mean F1 0.767, all misses under-removals); clean control never
  flagged.
- Streamlit dashboard (batch blame, minimal-set trace,
  evaluation), CLI (`demo`, `evaluate`), runnable notebooks,
  reproduce guard, and a 57-test suite with CI on Python
  3.10/3.11/3.12.
