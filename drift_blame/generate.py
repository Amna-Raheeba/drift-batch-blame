"""Seeded synthetic stream generator.

Rows are drawn from a declared base process (five numeric features,
one categorical device feature, one binary label whose probability
depends on the features).  Each source adds a small fixed benign
offset (calibration jitter) so clean streams are not sterile.  A
culprit batch then has one planted shift applied:

- ``mean_shift``: one numeric feature's mean moves by N sigmas.
- ``label_rate_shift``: labels are resampled at a target prevalence,
  breaking the feature->label relationship for that batch.
- ``schema_value_shift``: a fraction of rows carry a categorical
  value that never appears in the reference (a new device model
  silently entering the feed).

Everything is deterministic given the scenario seed.
"""

import math
import random

from .models import (
    BASE_MEANS,
    BASE_SDS,
    CATEGORIES,
    CATEGORY_PROBS,
    LABEL,
    NUMERIC_FEATURES,
    SCHEMA_NEW_VALUE,
    SHIFT_LABEL,
    SHIFT_MEAN,
    SHIFT_SCHEMA,
    SOURCES,
    Batch,
    Culprit,
    Scenario,
    Stream,
)


def _sigmoid(x):
    if x < -30.0:
        return 0.0
    if x > 30.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _label_probability(row):
    z = (
        0.030 * (row["age"] - 55.0)
        + 0.016 * (row["systolic_bp"] - 122.0)
        + 0.010 * (row["glucose_mg_dl"] - 104.0)
        + 1.15 * (row["creatinine"] - 1.0)
        + 0.045 * (row["bmi"] - 27.0)
        - 1.05
    )
    return _sigmoid(z)


def _draw_category(rng):
    r = rng.random()
    acc = 0.0
    for cat in CATEGORIES:
        acc += CATEGORY_PROBS[cat]
        if r <= acc:
            return cat
    return CATEGORIES[-1]


def _draw_row(rng, offsets):
    row = {}
    for feat in NUMERIC_FEATURES:
        row[feat] = rng.gauss(BASE_MEANS[feat] + offsets[feat], BASE_SDS[feat])
    row["device_model"] = _draw_category(rng)
    row[LABEL] = 1 if rng.random() < _label_probability(row) else 0
    return row


def _source_offsets(rng):
    """Small benign per-source calibration offsets (in raw units)."""
    offsets = {}
    for feat in NUMERIC_FEATURES:
        offsets[feat] = rng.gauss(0.0, 0.06 * BASE_SDS[feat])
    return offsets


def apply_shift(rows, culprit, rng):
    """Return a shifted copy of rows according to the culprit plan."""
    out = [dict(r) for r in rows]
    if culprit.shift_type == SHIFT_MEAN:
        delta = culprit.magnitude * BASE_SDS[culprit.feature]
        for row in out:
            row[culprit.feature] = row[culprit.feature] + delta
    elif culprit.shift_type == SHIFT_LABEL:
        for row in out:
            row[LABEL] = 1 if rng.random() < culprit.magnitude else 0
    elif culprit.shift_type == SHIFT_SCHEMA:
        for row in out:
            if rng.random() < culprit.magnitude:
                row["device_model"] = SCHEMA_NEW_VALUE
    else:
        raise ValueError("unknown shift type: %s" % culprit.shift_type)
    return out


def generate_stream(scenario):
    """Generate (reference rows, batches, planted truth) for a scenario."""
    rng = random.Random(scenario.seed)
    offsets_by_source = {}
    for source in SOURCES:
        offsets_by_source[source] = _source_offsets(rng)

    reference_rows = [
        _draw_row(rng, dict((f, 0.0) for f in NUMERIC_FEATURES))
        for _ in range(scenario.ref_size)
    ]

    culprit_by_index = dict((c.batch_index, c) for c in scenario.culprits)
    batches = []
    truth = {}
    for i in range(scenario.n_batches):
        source = SOURCES[i % len(SOURCES)] if i < len(SOURCES) else SOURCES[rng.randrange(len(SOURCES))]
        rows = [
            _draw_row(rng, offsets_by_source[source])
            for _ in range(scenario.batch_size)
        ]
        batch_id = "b%02d" % (i + 1)
        if i in culprit_by_index:
            rows = apply_shift(rows, culprit_by_index[i], rng)
            truth[batch_id] = culprit_by_index[i].shift_type
        batches.append(Batch(batch_id=batch_id, source=source, day=i + 1, rows=rows))
    return Stream(reference_rows=reference_rows, batches=batches, truth=truth)


def default_scenarios():
    """The canonical evaluation suite: 16 seeded scenarios.

    Culprit counts cycle 1, 2, 3, 1, ...; shift types rotate across
    mean / label-rate / schema-value; magnitudes vary per scenario.
    Culprit batch positions are drawn from the scenario seed so the
    suite is not aligned to fixed positions.
    """
    mean_features = ["glucose_mg_dl", "systolic_bp", "creatinine", "bmi", "age"]
    scenarios = []
    for k in range(16):
        seed = 101 + k
        rng = random.Random(seed * 7919)
        n_culprits = 1 + (k % 3)
        positions = rng.sample(range(10), n_culprits)
        culprits = []
        for j, pos in enumerate(positions):
            shift_type = [SHIFT_MEAN, SHIFT_LABEL, SHIFT_SCHEMA][(k + j) % 3]
            if shift_type == SHIFT_MEAN:
                culprits.append(Culprit(
                    batch_index=pos, shift_type=shift_type,
                    feature=mean_features[(k + j) % len(mean_features)],
                    magnitude=[1.5, 1.8, 2.1][(k + j) % 3],
                ))
            elif shift_type == SHIFT_LABEL:
                culprits.append(Culprit(
                    batch_index=pos, shift_type=shift_type,
                    magnitude=[0.55, 0.60, 0.65][(k + j) % 3],
                ))
            else:
                culprits.append(Culprit(
                    batch_index=pos, shift_type=shift_type,
                    magnitude=[0.40, 0.50, 0.60][(k + j) % 3],
                ))
        scenarios.append(Scenario(seed=seed, culprits=culprits))
    return scenarios


def demo_scenario():
    """Small fixed scenario used by the demo, dashboard and examples."""
    return Scenario(
        seed=42,
        culprits=[
            Culprit(batch_index=3, shift_type=SHIFT_MEAN,
                    feature="glucose_mg_dl", magnitude=1.8),
            Culprit(batch_index=7, shift_type=SHIFT_LABEL, magnitude=0.60),
        ],
    )
