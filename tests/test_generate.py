import pytest

from drift_blame.generate import (
    apply_shift,
    default_scenarios,
    demo_scenario,
    generate_stream,
)
from drift_blame.models import (
    BASE_MEANS,
    BASE_SDS,
    LABEL,
    SCHEMA_NEW_VALUE,
    SHIFT_LABEL,
    SHIFT_MEAN,
    SHIFT_SCHEMA,
    Culprit,
    Scenario,
)


def _mean(rows, feat):
    return sum(r[feat] for r in rows) / float(len(rows))


def test_generation_is_deterministic():
    scenario = Scenario(seed=5, culprits=[])
    first = generate_stream(scenario)
    second = generate_stream(scenario)
    assert first.reference_rows == second.reference_rows
    assert [b.rows for b in first.batches] == [b.rows for b in second.batches]


def test_stream_shape_and_sources():
    stream = generate_stream(Scenario(seed=5, culprits=[]))
    assert len(stream.reference_rows) == 3000
    assert len(stream.batches) == 10
    for batch in stream.batches:
        assert batch.size == 300
        assert batch.batch_id.startswith("b")
    assert stream.truth == {}


def test_rows_have_all_fields():
    stream = generate_stream(Scenario(seed=5, culprits=[]))
    row = stream.reference_rows[0]
    for feat in ["age", "systolic_bp", "glucose_mg_dl", "creatinine", "bmi",
                 "device_model", LABEL]:
        assert feat in row
    assert row[LABEL] in (0, 1)


def test_mean_shift_moves_only_the_target_feature():
    clean = generate_stream(Scenario(seed=9, culprits=[]))
    shifted = generate_stream(Scenario(
        seed=9,
        culprits=[Culprit(batch_index=2, shift_type=SHIFT_MEAN,
                          feature="glucose_mg_dl", magnitude=1.8)],
    ))
    clean_rows = clean.batches[2].rows
    shifted_rows = shifted.batches[2].rows
    expected = 1.8 * BASE_SDS["glucose_mg_dl"]
    diff = _mean(shifted_rows, "glucose_mg_dl") - _mean(clean_rows, "glucose_mg_dl")
    assert abs(diff - expected) < 1e-6
    for feat in ["age", "systolic_bp", "creatinine", "bmi"]:
        assert _mean(shifted_rows, feat) == _mean(clean_rows, feat)
    assert shifted.truth == {"b03": SHIFT_MEAN}


def test_label_shift_sets_prevalence():
    stream = generate_stream(Scenario(
        seed=11,
        culprits=[Culprit(batch_index=0, shift_type=SHIFT_LABEL, magnitude=0.6)],
    ))
    rows = stream.batches[0].rows
    rate = sum(r[LABEL] for r in rows) / float(len(rows))
    assert abs(rate - 0.6) < 0.09
    base = generate_stream(Scenario(seed=11, culprits=[]))
    base_rate = sum(r[LABEL] for r in base.reference_rows) / 3000.0
    assert base_rate < 0.45


def test_schema_shift_plants_unseen_value():
    stream = generate_stream(Scenario(
        seed=13,
        culprits=[Culprit(batch_index=1, shift_type=SHIFT_SCHEMA, magnitude=0.5)],
    ))
    rows = stream.batches[1].rows
    frac = sum(1 for r in rows if r["device_model"] == SCHEMA_NEW_VALUE) / float(len(rows))
    assert abs(frac - 0.5) < 0.09
    clean = generate_stream(Scenario(seed=13, culprits=[]))
    assert all(r["device_model"] != SCHEMA_NEW_VALUE
               for b in clean.batches for r in b.rows)
    assert all(r["device_model"] != SCHEMA_NEW_VALUE
               for r in clean.reference_rows)


def test_apply_shift_rejects_unknown_type():
    rows = generate_stream(Scenario(seed=3, culprits=[])).batches[0].rows
    with pytest.raises(ValueError):
        apply_shift(rows, Culprit(batch_index=0, shift_type="nope"), __import__("random").Random(1))


def test_default_scenarios_structure():
    scenarios = default_scenarios()
    assert len(scenarios) == 16
    counts = sorted(len(s.culprits) for s in scenarios)
    assert counts[0] == 1 and counts[-1] == 3
    seen_types = set()
    for scenario in scenarios:
        positions = [c.batch_index for c in scenario.culprits]
        assert len(set(positions)) == len(positions)
        for culprit in scenario.culprits:
            seen_types.add(culprit.shift_type)
            assert 0 <= culprit.batch_index < scenario.n_batches
    assert seen_types == {SHIFT_MEAN, SHIFT_LABEL, SHIFT_SCHEMA}


def test_demo_scenario_has_two_culprits():
    scenario = demo_scenario()
    stream = generate_stream(scenario)
    assert sorted(stream.truth.keys()) == ["b04", "b08"]
    assert stream.truth["b04"] == SHIFT_MEAN
    assert stream.truth["b08"] == SHIFT_LABEL


def test_clean_batch_means_near_base():
    stream = generate_stream(Scenario(seed=21, culprits=[]))
    for feat in ["age", "glucose_mg_dl"]:
        pooled = [r for b in stream.batches for r in b.rows]
        assert abs(_mean(pooled, feat) - BASE_MEANS[feat]) < 0.2 * BASE_SDS[feat]
