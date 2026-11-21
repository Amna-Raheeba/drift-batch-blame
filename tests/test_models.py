from drift_blame import models


def test_feature_sets_are_consistent():
    assert len(models.NUMERIC_FEATURES) == 5
    assert models.CATEGORICAL_FEATURE == "device_model"
    assert models.LABEL == "readmitted"
    assert models.ALL_FEATURES == (
        models.NUMERIC_FEATURES + [models.CATEGORICAL_FEATURE, models.LABEL]
    )


def test_categories_and_sources():
    assert models.CATEGORIES == ["A", "B", "C"]
    assert models.SCHEMA_NEW_VALUE not in models.CATEGORIES
    assert len(models.SOURCES) == 4
    assert len(set(models.SOURCES)) == 4


def test_shift_types():
    assert models.SHIFT_TYPES == [
        models.SHIFT_MEAN, models.SHIFT_LABEL, models.SHIFT_SCHEMA,
    ]


def test_base_params_cover_numeric_features():
    for feat in models.NUMERIC_FEATURES:
        assert feat in models.BASE_MEANS
        assert feat in models.BASE_SDS
        assert models.BASE_SDS[feat] > 0


def test_category_probs_sum_to_one():
    total = sum(models.CATEGORY_PROBS.values())
    assert abs(total - 1.0) < 1e-9
    assert sorted(models.CATEGORY_PROBS.keys()) == models.CATEGORIES


def test_batch_size_property():
    batch = models.Batch(batch_id="b01", source="north-feed", day=1,
                         rows=[{"x": 1}, {"x": 2}])
    assert batch.size == 2


def test_scenario_defaults():
    scenario = models.Scenario(seed=1)
    assert scenario.n_batches == 10
    assert scenario.batch_size == 300
    assert scenario.ref_size == 3000
    assert scenario.culprits == []


def test_stream_culprit_ids_sorted():
    stream = models.Stream(
        reference_rows=[], batches=[],
        truth={"b08": "mean_shift", "b03": "label_rate_shift"},
    )
    assert stream.culprit_ids == ["b03", "b08"]
