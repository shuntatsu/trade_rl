from dataclasses import replace

import numpy as np
import pytest

from tests.strategies.test_ppo_interleaved_training import (
    FakeDummyVecEnv,
    FakePPO,
    _ppo_identity_market,
    install_fake_sb3,
    pooled_market,
)
from trade_rl.data.contracts import FeatureKind
from trade_rl.strategies.rl.ppo import (
    PPOIntentStrategy,
    PPOTradingEnv,
    fit_ppo_strategy,
)
from trade_rl.strategies.rl.ppo_normalization import (
    PPOFeatureNormalizer,
    fit_ppo_feature_normalizer,
)


def _fit(dataset=None, **kwargs):
    return fit_ppo_feature_normalizer(
        dataset or pooled_market(),
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        start_index=0,
        stop_index=3,
        **kwargs,
    )


def test_only_permitted_decision_rows_and_symbols_influence_statistics() -> None:
    dataset = pooled_market()
    original = _fit(dataset)
    shifted = dataset.features.copy()
    shifted[3:] += 10000
    shifted[:, 1] += 20000
    changed = _fit(replace(dataset, features=shifted))
    assert original.to_payload() == changed.to_payload()
    assert original.mean == (1.0,)
    assert original.scale == pytest.approx((np.sqrt(2 / 3),))
    assert original.usable_counts == ((3,),)


def test_symbol_balancing_is_not_distorted_by_missing_rows() -> None:
    dataset = pooled_market()
    values = dataset.features.copy()
    values[:3, 0, 0] = 0
    values[:3, 1, 0] = 10
    available = dataset.feature_available.copy()
    available[1:3, 1, 0] = False
    normalizer = fit_ppo_feature_normalizer(
        replace(
            dataset,
            features=values,
            feature_available=available,
            feature_staleness=None,
            feature_missing_reason=None,
        ),
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
    )
    assert normalizer.mean == (5.0,)
    assert normalizer.scale == (5.0,)
    assert normalizer.usable_counts == ((3,), (1,))


def test_constant_feature_and_missing_input_keep_finite_zero_mask_semantics() -> None:
    dataset = pooled_market()
    values = dataset.features.copy()
    values[:3, 0, 0] = 4
    normalizer = _fit(replace(dataset, features=values))
    assert normalizer.scale == (1.0,)
    np.testing.assert_array_equal(
        normalizer.transform(np.array([np.nan]), np.array([False])), [0]
    )
    np.testing.assert_array_equal(
        normalizer.transform(np.array([5.0]), np.array([True])), [1]
    )
    with pytest.raises(ValueError, match="usable"):
        _fit(
            replace(
                dataset,
                feature_available=np.zeros_like(dataset.feature_available),
                feature_staleness=None,
                feature_missing_reason=None,
            )
        )


def test_training_and_inference_share_transform_without_touching_other_fields() -> None:
    dataset = pooled_market()
    normalizer = _fit(dataset)
    arguments = dict(
        dataset=dataset,
        feature_indices=(0,),
        symbol_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
    )
    raw = PPOTradingEnv(**arguments)
    normalized = PPOTradingEnv(**arguments, feature_normalizer=normalizer)
    raw_observation, _ = raw.reset(seed=0)
    normalized_observation, _ = normalized.reset(seed=0)
    np.testing.assert_array_equal(raw_observation[1:], normalized_observation[1:])
    assert normalized_observation[0] == pytest.approx(-np.sqrt(1.5))
    captured = []

    class Capture:
        def predict(self, observation, *, deterministic=True):
            captured.append(observation.copy())
            return np.array(1), None

    strategy = PPOIntentStrategy(
        Capture(), feature_indices=(0,), feature_normalizer=normalizer
    )
    strategy.decide(normalized._strategy_observation())
    np.testing.assert_array_equal(captured[0], normalized_observation)
    np.testing.assert_array_equal(normalized.reset()[0], normalized_observation)


@pytest.mark.parametrize("layout", ["sequential", "interleaved"])
def test_fitter_fits_one_transform_and_returns_it_for_inference(
    layout, monkeypatch
) -> None:
    install_fake_sb3(monkeypatch)
    strategy = fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=64,
        training_layout=layout,
        rollout_steps_per_env=32 if layout == "interleaved" else None,
        normalize_features=True,
    )
    assert strategy.feature_normalizer is not None
    assert strategy.feature_normalizer.mean == (51.0,)
    environments = (
        FakeDummyVecEnv.last.envs if layout == "interleaved" else [FakePPO.last.env]
    )
    for env in environments:
        env.reset()
        assert env.feature_normalizer is strategy.feature_normalizer


def test_metadata_roundtrip_and_wrong_feature_order_or_training_scope_rejected() -> (
    None
):
    normalizer = _fit()
    assert PPOFeatureNormalizer.from_payload(normalizer.to_payload()) == normalizer
    bad = normalizer.to_payload()
    bad["scale"] = [0.0]
    with pytest.raises(ValueError, match="scale"):
        PPOFeatureNormalizer.from_payload(bad)
    with pytest.raises(ValueError, match="feature"):
        PPOIntentStrategy(object(), feature_indices=(1,), feature_normalizer=normalizer)
    with pytest.raises(ValueError, match="scope"):
        PPOTradingEnv(
            pooled_market(),
            feature_indices=(0,),
            symbol_indices=(0,),
            start_index=0,
            stop_index=2,
            gross_budget=0.1,
            feature_normalizer=normalizer,
        )


def test_omitted_normalization_preserves_legacy_encoded_values(monkeypatch) -> None:
    install_fake_sb3(monkeypatch)
    strategy = fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=64,
    )
    assert strategy.feature_normalizer is None
    assert FakePPO.last.env.reset()[0][0] == 0


def test_normalizer_rejects_nonfinite_scale_overflow_and_boolean_statistics() -> None:
    normalizer = _fit()
    for bad_value in (float("inf"), True):
        payload = normalizer.to_payload()
        payload["scale"] = [bad_value]
        with pytest.raises(ValueError):
            PPOFeatureNormalizer.from_payload(payload)
    with pytest.raises(ValueError, match="float32"):
        normalizer.transform(np.array([1e100]), np.array([True]))


@pytest.mark.parametrize(
    "field",
    [
        "feature_names",
        "feature_indices",
        "fit_symbol_indices",
        "mean",
        "scale",
        "usable_counts",
    ],
)
def test_metadata_sequences_must_be_json_arrays(field) -> None:
    payload = _fit().to_payload()
    payload[field] = {"signal": "wrong container"}
    with pytest.raises(ValueError, match="array"):
        PPOFeatureNormalizer.from_payload(payload)


def test_metadata_count_rows_must_be_json_arrays() -> None:
    payload = _fit().to_payload()
    payload["usable_counts"] = [{3: "wrong container"}]
    with pytest.raises(ValueError, match="array"):
        PPOFeatureNormalizer.from_payload(payload)


def test_normalizer_rejects_universe_dependency_outside_fit_scope() -> None:
    with pytest.raises(ValueError, match="outside fit symbol scope"):
        fit_ppo_feature_normalizer(
            _ppo_identity_market(FeatureKind.CROSS_ASSET_DISPERSION),
            feature_indices=(0,),
            fit_symbol_indices=(0,),
            start_index=0,
            stop_index=3,
        )
