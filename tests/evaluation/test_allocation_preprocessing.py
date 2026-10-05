"""Independent causal row admission and equal-symbol arithmetic oracles."""

import importlib
from dataclasses import replace

import numpy as np
import pytest

from tests.strategies.test_ppo_interleaved_training import (
    _ppo_identity_market,
    pooled_market,
)
from trade_rl.data.contracts import FeatureKind
from trade_rl.strategies.rl.ppo_normalization import fit_ppo_feature_normalizer

MODULE = "trade_rl.evaluation.rl_allocation.preprocessing"


def fitter():
    try:
        return importlib.import_module(MODULE).fit_allocation_feature_preprocessing
    except ModuleNotFoundError as error:
        if error.name != MODULE:
            raise
        pytest.fail("allocation causal prefix fitter is missing")


def fit(dataset=None, **changes):
    dataset = pooled_market() if dataset is None else dataset
    args = dict(
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        fit_start=0,
        fit_stop=3,
        fit_as_of=dataset.timestamps[3],
        first_decision_index=3,
    )
    return fitter()(dataset, **(args | changes))


def test_delayed_source_excluded_at_own_clock_not_retroactive_prefix_cutoff():
    dataset = pooled_market()
    values, available = dataset.features.copy(), dataset.available_at.copy()
    values[:3, 0, 0] = [0, 1000, 2]
    available[1, 0] += np.timedelta64(1, "ns")
    dataset = replace(
        dataset, features=values, available_at=available, information_available=None
    )
    fitted = fit(dataset)
    assert fitted.normalizer.mean == (1.0,) and fitted.normalizer.scale == (1.0,)
    assert fitted.admitted_row_indices == ((0, 2),)
    assert fitted.normalizer.usable_counts == ((2,),)
    old = fit_ppo_feature_normalizer(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        start_index=0,
        stop_index=3,
    )
    assert old.mean == (334.0,)
    assert fit(pooled_market()).admitted_row_indices == ((0, 1, 2),)


def test_joint_feature_admission_and_equal_symbol_weighting_literal_oracle():
    dataset = pooled_market()
    values = np.zeros((4, 2, 2), dtype=np.float32)
    values[:, 1] = 10
    available = np.ones_like(values, dtype=bool)
    available[1:3, 1, 1] = False
    dataset = replace(
        dataset,
        features=values,
        feature_available=available,
        feature_names=("signal", "second"),
        feature_staleness=None,
        feature_staleness_hours=None,
        feature_missing_reason=None,
    )
    fitted = fit(dataset, feature_indices=(0, 1), fit_symbol_indices=(0, 1))
    assert fitted.normalizer.mean == (5.0, 5.0)  # pooled row mean would be2.5
    assert fitted.normalizer.scale == (5.0, 5.0)
    assert fitted.normalizer.usable_counts == ((3, 3), (1, 1))
    assert fitted.admitted_row_indices == ((0, 1, 2), (0,))


def test_future_outsider_excluded_values_and_dataset_lineage_have_distinct_identity():
    dataset = pooled_market()
    before = fit(dataset)
    values = dataset.features.copy()
    values[3:] += 1000
    values[:, 1] += 2000
    changed = fit(replace(dataset, features=values, dataset_id="7" * 64))
    assert changed.normalizer.mean == before.normalizer.mean == (1.0,)
    assert changed.normalizer.scale == before.normalizer.scale
    assert changed.fit_consumption_digest == before.fit_consumption_digest
    assert changed.digest != before.digest


def test_excluded_value_magnitude_not_read_into_fit_digest_but_masks_and_clocks_are():
    dataset = pooled_market()
    available = dataset.feature_available.copy()
    available[1, 0, 0] = False
    dataset = replace(dataset, feature_available=available, feature_staleness=None)
    before = fit(dataset)
    values = dataset.features.copy()
    values[1, 0, 0] = 1000
    assert (
        fit(replace(dataset, features=values)).fit_consumption_digest
        == before.fit_consumption_digest
    )
    source = dataset.available_at.copy()
    source[1, 0] += np.timedelta64(1, "ns")
    after = fit(replace(dataset, available_at=source, information_available=None))
    assert after.normalizer.mean == before.normalizer.mean
    assert after.fit_consumption_digest != before.fit_consumption_digest


def test_constant_prefix_fallback_finite_extremes_and_source_nonmutation():
    dataset = pooled_market()
    values = dataset.features.copy()
    values[:3, 0, 0] = 4
    dataset = replace(dataset, features=values)
    before = (
        dataset.features.copy(),
        dataset.feature_available.copy(),
        dataset.available_at.copy(),
    )
    fitted = fit(dataset)
    assert fitted.normalizer.mean == (4.0,) and fitted.normalizer.scale == (1.0,)
    for actual, expected in zip(
        (dataset.features, dataset.feature_available, dataset.available_at), before
    ):
        np.testing.assert_array_equal(actual, expected)
    values[:3, 0, 0] = [np.finfo(np.float32).max, -np.finfo(np.float32).max, 0]
    extreme = fit(replace(dataset, features=values))
    assert np.isfinite(extreme.normalizer.scale).all()


@pytest.mark.parametrize(
    "changes",
    [
        dict(fit_start=True),
        dict(fit_start=-1),
        dict(fit_stop=4),
        dict(fit_stop=0),
        dict(first_decision_index=True),
        dict(first_decision_index=2),
        dict(first_decision_index=4),
        dict(fit_as_of=np.datetime64("NaT")),
        dict(fit_as_of=np.datetime64("2500", "Y")),
        dict(fit_as_of=np.datetime64("2026-01-01T02", "ns")),
        dict(fit_as_of=np.datetime64("2026-01-01T04", "ns")),
        dict(feature_indices=(True,)),
        dict(fit_symbol_indices=None),
        dict(fit_symbol_indices=(2,)),
    ],
)
def test_invalid_prefix_scope_and_clocks_reject(changes):
    with pytest.raises(ValueError):
        fit(**changes)


def test_dataset_rejects_declared_available_nonfinite_prefix():
    dataset = pooled_market()
    values = dataset.features.copy()
    values[:3, 0, 0] = np.nan
    with pytest.raises(ValueError):
        fit(replace(dataset, features=values))


def test_fitter_rejects_valid_all_unavailable_prefix_without_imputation():
    dataset = pooled_market()
    mask = dataset.feature_available.copy()
    mask[:3, 0, 0] = False
    dataset = replace(dataset, feature_available=mask, feature_staleness=None)
    with pytest.raises(ValueError, match="every fit symbol needs jointly observable"):
        fit(dataset)


def test_prefix_fitter_preserves_selected_feature_dependency_guard():
    with pytest.raises(ValueError, match="outside fit symbol scope"):
        fit(_ppo_identity_market(FeatureKind.CROSS_ASSET_DISPERSION))
