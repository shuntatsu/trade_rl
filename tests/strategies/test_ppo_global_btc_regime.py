from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl import ppo


def _implementation_module():
    return importlib.import_module("trade_rl.strategies.rl.ppo_global_btc_regime")


def _market(
    *,
    reference_available_at_row_one: bool = True,
    future_reference_shift: float = 0.0,
    symbols: tuple[str, str] = ("BTCUSDT", "ETHUSDT"),
    feature_names: tuple[str, str] = ("local_signal", "1h__log_return_24bar"),
) -> MarketDataset:
    close = np.asarray(
        [
            [100.0, 200.0],
            [101.0, 201.0],
            [102.0, 202.0],
            [103.0, 203.0],
        ],
        dtype=np.float64,
    )
    features = np.asarray(
        [
            [[10.0, 0.10], [20.0, 9.10]],
            [[11.0, 0.20], [21.0, 9.20]],
            [[12.0, 0.30 + future_reference_shift], [22.0, 9.30]],
            [[13.0, 0.40 + future_reference_shift], [23.0, 9.40]],
        ],
        dtype=np.float32,
    )
    feature_available = np.ones_like(features, dtype=np.bool_)
    feature_staleness = np.asarray(
        [
            [[0.10, 0.25], [0.20, 0.90]],
            [[0.00, 0.00], [0.10, 0.80]],
            [[0.00, 0.00], [0.00, 0.70]],
            [[0.00, 0.00], [0.00, 0.60]],
        ],
        dtype=np.float32,
    )
    if not reference_available_at_row_one:
        feature_available[1, 0, 1] = False
        feature_staleness[1, 0, 1] = 1.0

    return MarketDataset(
        dataset_id="7" * 64,
        symbols=symbols,
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.asarray(
            [[1000.0], [2000.0], [3000.0], [4000.0]], dtype=np.float32
        ),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((4, 2), 1_000_000.0),
        funding_rate=np.zeros((4, 2)),
        tradable=np.ones((4, 2), dtype=np.bool_),
        feature_available=feature_available,
        feature_names=feature_names,
        global_feature_names=("unrelated_global",),
        periods_per_year=8_760,
        feature_staleness=feature_staleness,
        global_feature_available=np.ones((4, 1), dtype=np.bool_),
    )


def _strategy_observation(dataset: MarketDataset, *, symbol_index: int, index: int):
    return StrategyObservation(
        index=index,
        timestamp=dataset.timestamps[index],
        symbol=dataset.symbols[symbol_index],
        features=dataset.features[index, symbol_index],
        feature_available=dataset.feature_available[index, symbol_index],
        feature_staleness=dataset.resolved_array("feature_staleness")[
            index, symbol_index
        ],
        global_features=dataset.global_features[index],
        global_feature_available=dataset.resolved_array(
            "global_feature_available"
        )[index],
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )


def test_sealed_global_btc_regime_observation_contract_is_exact() -> None:
    module = _implementation_module()

    assert (
        module.PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA
        == "ppo_observation_v3_global_btc_regime"
    )
    assert module.PPO_GLOBAL_BTC_REGIME_REFERENCE_SYMBOL == "BTCUSDT"
    assert module.PPO_GLOBAL_BTC_REGIME_REFERENCE_FEATURE == "1h__log_return_24bar"
    assert module.ppo_global_btc_regime_observation_contract_payload() == {
        "schema_version": "ppo_observation_v3_global_btc_regime",
        "baseline_schema_version": "ppo_observation_v2",
        "reference_symbol": "BTCUSDT",
        "reference_feature": "1h__log_return_24bar",
        "reference_observation_components": [
            "value",
            "available_and_finite",
            "normalized_staleness",
        ],
        "reference_value_source": "canonical_dataset.features",
        "reference_available_source": "canonical_dataset.feature_available",
        "reference_staleness_source": "canonical_dataset.feature_staleness",
        "reference_usable_semantics": "feature_available_and_isfinite",
        "reference_unavailable_value": 0.0,
        "global_reference_timestamp_alignment": "same_dataset_row",
        "global_reference_broadcast_is_symbol_independent": True,
        "layout": [
            "local_values",
            "local_available",
            "local_staleness",
            "global_reference_value",
            "global_reference_available_and_finite",
            "global_reference_normalized_staleness",
            "current_intent",
            "current_weight",
        ],
    }


def test_candidate_env_uses_exact_layout_and_broadcasts_same_btc_reference() -> None:
    module = _implementation_module()
    env = module.PPOGlobalBTCRegimeTradingEnv(
        _market(),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )

    btc_observation, btc_info = env.reset(seed=7)
    eth_observation, eth_info = env.reset()

    np.testing.assert_array_equal(
        btc_observation,
        np.asarray([10.0, 1.0, 0.10, 0.10, 1.0, 0.25, 0.0, 0.0], np.float32),
    )
    np.testing.assert_array_equal(
        eth_observation,
        np.asarray([20.0, 1.0, 0.20, 0.10, 1.0, 0.25, 0.0, 0.0], np.float32),
    )
    assert btc_info["symbol"] == "BTCUSDT"
    assert eth_info["symbol"] == "ETHUSDT"
    assert env.observation_space.shape == (8,)


def test_unavailable_reference_zeroes_only_value_and_preserves_staleness() -> None:
    module = _implementation_module()
    env = module.PPOGlobalBTCRegimeTradingEnv(
        _market(reference_available_at_row_one=False),
        feature_indices=(0,),
        start_index=1,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )

    observation, _ = env.reset(seed=11)

    np.testing.assert_array_equal(
        observation[3:6],
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )


def test_candidate_reference_reads_same_row_only_and_ignores_future_mutation() -> None:
    module = _implementation_module()
    baseline = module.PPOGlobalBTCRegimeTradingEnv(
        _market(future_reference_shift=0.0),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
    )
    mutated = module.PPOGlobalBTCRegimeTradingEnv(
        _market(future_reference_shift=999.0),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
    )

    baseline_observation, _ = baseline.reset(seed=3)
    mutated_observation, _ = mutated.reset(seed=3)

    np.testing.assert_array_equal(baseline_observation, mutated_observation)


def test_candidate_reference_fails_closed_when_frozen_source_is_missing() -> None:
    module = _implementation_module()

    with pytest.raises(ValueError, match="BTCUSDT"):
        module.PPOGlobalBTCRegimeTradingEnv(
            _market(symbols=("XBTUSDT", "ETHUSDT")),
            feature_indices=(0,),
            start_index=0,
            stop_index=3,
            gross_budget=0.5,
        )

    with pytest.raises(ValueError, match="1h__log_return_24bar"):
        module.PPOGlobalBTCRegimeTradingEnv(
            _market(feature_names=("local_signal", "other_feature")),
            feature_indices=(0,),
            start_index=0,
            stop_index=3,
            gross_budget=0.5,
        )


def test_baseline_observation_v2_contract_and_shape_are_unchanged() -> None:
    dataset = _market()
    env = ppo.PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
    )

    observation, _ = env.reset(seed=5)

    assert ppo.PPO_OBSERVATION_SCHEMA == "ppo_observation_v2"
    assert ppo.PPO_GLOBAL_FEATURE_NAMES == ()
    assert observation.shape == (5,)
    assert ppo.ppo_observation_contract_payload()["layout"] == [
        "local_values",
        "local_available",
        "local_staleness",
        "current_intent",
        "current_weight",
    ]


class _FakePPO:
    last: _FakePPO | None = None

    def __init__(self, policy: str, env: object, **kwargs: object) -> None:
        self.policy = policy
        self.env = env
        self.kwargs = kwargs
        self.learn_timesteps: int | None = None
        self.predicted_observation: np.ndarray | None = None
        _FakePPO.last = self

    def learn(self, total_timesteps: int):
        self.learn_timesteps = total_timesteps
        return self

    def predict(self, observation: np.ndarray, *, deterministic: bool = True):
        assert deterministic is True
        self.predicted_observation = observation.copy()
        return np.asarray(1), None


def test_candidate_fit_changes_only_observation_path_not_ppo_hyperparameters(
    monkeypatch,
) -> None:
    module = _implementation_module()
    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=_FakePPO))
    dataset = _market()

    strategy = module.fit_ppo_global_btc_regime_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        total_timesteps=256,
        seed=13,
        initial_capital=1_000.0,
    )

    fitted = _FakePPO.last
    assert fitted is not None
    assert fitted.policy == "MlpPolicy"
    assert fitted.kwargs == {
        "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        "seed": 13,
        "ent_coef": 0.0,
        "verbose": 0,
    }
    assert fitted.learn_timesteps == 256
    assert fitted.env.observation_space.shape == (8,)
    assert isinstance(strategy, module.PPOGlobalBTCRegimeIntentStrategy)

    intent = strategy.decide(_strategy_observation(dataset, symbol_index=1, index=0))
    assert intent is PositionIntent.FLAT
    assert fitted.predicted_observation is not None
    np.testing.assert_array_equal(
        fitted.predicted_observation,
        np.asarray([20.0, 1.0, 0.20, 0.10, 1.0, 0.25, 0.0, 0.0], np.float32),
    )

    mismatched = _strategy_observation(dataset, symbol_index=1, index=0)
    object.__setattr__(
        mismatched,
        "timestamp",
        np.datetime64("2026-01-02", "ns"),
    )
    with pytest.raises(ValueError, match="timestamp"):
        strategy.decide(mismatched)
