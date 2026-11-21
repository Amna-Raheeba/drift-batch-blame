# Drift Batch Blame

**Blame the batch, not the feature.** When a dataset drifts, this
tool attributes the drift to the *ingestion batches* that caused
it — and returns the minimal set of batches whose removal restores
the reference distribution.

## 1. What it is

Drift monitors answer "the data changed, mostly in feature X".
Operations teams need a different answer: *which delivery, from
which source, broke the data — and what do we drop to fix it?*

Datasets arrive as timestamped batches from named sources. This
auditor pools the batches against a reference sample, measures
drift, and then replays the pool under removals to assign blame at
the level of provenance: per batch, per source, and as a minimal
removal set. Everything is offline, deterministic and seeded. Pure
Python standard library; no API keys, no model downloads, no
network. All data is synthetic, generated in-repo from a declared
process — there is no real patient or production data anywhere in
this project.

## 2. The novel mechanism

**Leave-one-batch-out replay + Shapley-style attribution over
ingestion provenance.** Three complementary outputs:

1. **LOBO blame** — for each batch, the drop in pooled drift when
   that batch is removed. Clean batches score at or below zero
   (removing them concentrates the drift); culprit batches carry
   the excess.
2. **Shapley blame** — each batch's average marginal drift
   contribution over seeded random orderings of the stream, so
   blame is shared fairly when two culprit batches interact (e.g.
   two shifts on the same feature, where leave-one-out masks each
   behind the other). Source-level Shapley is exact: all subsets
   of sources are enumerated.
3. **Minimal blame set** — a greedy pass that repeatedly removes
   the highest-blame batch until the pooled remainder is
   *restored*: every feature back within a declared noise band
   (statistic ≤ 2.5 × its no-drift null expectation). The removed
   set is a certified repair, not just a ranking.

Drift itself is measured per feature with sample-size-standardised
statistics — binned Kolmogorov–Smirnov for numeric features, total
variation for the categorical feature and the label — over bin
counts, so every replay is exact arithmetic on count vectors. The
pooled drift score is the total excess over the null band.

## 3. Prior art and the differentiator

Per-feature drift monitors (PSI/KS dashboards in the Evidently
style) attribute drift to *features*; fairness-drift Shapley work
(DriftFair, 2026) also attributes to features; CJSD and CausalShift
decompose drift *type*. None attributes drift to data-supply
batches or sources, and none returns a minimal removal set with a
restoration certificate. Batch-level Shapley attribution over
ingestion provenance is the unit of analysis here, and it is the
one a data engineer can act on: drop these deliveries, call this
source.

## 4. Measured results

Seeded evaluation: 16 scenarios, each a 3,000-row reference plus 10
batches × 300 rows from 4 sources, with 1–3 planted culprit batches
(mean 1.94) carrying a known shift — a mean shift (1.5–2.1 σ on one
feature), a label-rate shift (prevalence 0.55–0.65), or a
schema-value shift (an unseen category in 40–60% of the batch).
Full per-scenario detail in `results/`.

| method | precision@k | recall@k |
|---|---|---|
| Shapley blame | 1.000 | 1.000 |
| LOBO blame | 0.969 | 0.969 |
| Batch own-drift (baseline) | 1.000 | 1.000 |
| Per-feature PSI mapped to batches (baseline) | 0.708 | 0.708 |

- **Minimal blame set:** exact match with the planted culprit set
  in 0.625 of scenarios, mean set F1 0.767. Every miss is an
  *under*-removal — a culprit whose pooled footprint sits under
  the 2.5 × null restoration band, so the certificate stays
  conservative — never an over-removal.
- **Source level:** the top source by Shapley contains a culprit
  in 1.000 of scenarios (summed LOBO: 0.938).
- **Clean control:** on 8 streams with no planted culprits, the
  audit flags 0.000 as drifted and returns an empty blame set for
  every stream.
- The one LOBO ranking miss is a masking case: two culprits
  shifting the same feature, where removing either alone barely
  moves the pooled statistic. Shapley blame, which averages over
  orderings, ranks both correctly — the reason both are reported.
- The PSI baseline fails exactly as predicted: it sees one feature
  at a time, never looks at the label (label-rate culprits are
  invisible to it), and smears blame across every batch that
  touches the drifted feature. The own-drift baseline ties on
  ranking at these shift magnitudes — a single large shifted batch
  is easy to spot alone — but it produces no minimal set, no
  restoration certificate, and no fair sharing when culprits
  interact.

## 5. How to run

```bash
pip install -r requirements.txt

python -m drift_blame demo                 # audit the seeded demo stream
python -m drift_blame evaluate --out results   # full 16-scenario evaluation
python scripts/build_cases.py              # regenerate data/ files
python scripts/reproduce_results.py        # headline-drift guard vs results/
python notebooks/01_demo_run.py            # demo as a script
python notebooks/02_evaluation_experiment.py   # reduced evaluation
streamlit run app.py                       # dashboard (3 tabs)
pytest -q                                  # test suite
```

The demo stream plants two culprits — batch `b04` (glucose mean
shift, 1.8 σ) and `b08` (label-rate shift to 0.60). The audit ranks
both top by LOBO and Shapley blame, and the greedy minimal set
recovers exactly `{b04, b08}`, restoring the pooled remainder
(max feature ratio 3.64 → 1.60 × null).

## 6. Repository structure

- `drift_blame/` — the package: `models` (data model + DGP
  constants), `generate` (seeded stream generator with planted
  shifts), `drift` (count-based standardised drift), `blame`
  (LOBO, Shapley, source Shapley, greedy minimal set), `baselines`
  (PSI-mapped and own-drift), `evaluate`, `report`, `cli`.
- `data/` — `demo_stream.json` (the stored demo stream) and
  `scenarios.json` (the canonical evaluation specs), both
  regenerated by `scripts/build_cases.py`.
- `results/` — shipped evaluation artifacts (`evaluation.json`,
  `evaluation.csv`, `evaluation_summary.md`, `HEADLINE.md`).
- `notebooks/` — runnable demo and evaluation scripts.
- `examples/` — an audit of the stored demo stream plus its real
  output (`sample_report.md`).
- `tests/` — 57 tests across all modules, including a dashboard
  smoke test and a reproduce-guard test.
- `app.py` — Streamlit dashboard: batch blame, minimal-set trace,
  evaluation.

## 7. Honest limits

All streams are synthetic, drawn from this project's own declared
process with planted shifts, so the experiment validates the
attribution machinery against known ground truth — it does not
measure blame accuracy on any real feed. Three specific caveats:

- **Shift magnitudes are moderate-to-large** (1.5–2.1 σ mean
  shifts, prevalence moved to 0.55–0.65, 40–60% schema
  contamination). Weaker or diluted shifts (a culprit affecting
  only a few percent of a batch's rows) will degrade every method,
  the pooled ones included; that regime is not measured here.
- **The restoration band is a declared convention** (2.5 × the
  null expectation of the maximum feature statistic), not a
  calibrated guarantee. On clean streams the pooled maximum ratio
  has a heavy right tail (median ≈ 1.8, up to ≈ 3.3 over 40 seeds
  measured during development), so the band trades sensitivity for
  a low false-flag rate; the minimal set inherits that
  conservatism, which is why its exact-match rate (0.625) trails
  the blame rankings.
- **Batch Shapley is sampled** (64 seeded permutations), not exact;
  source Shapley is exact. LOBO blame can mask interacting
  culprits on the same feature, as documented in section 4.

## 8. License and author

MIT License (see `LICENSE`). Author: Amna Raheeba.
