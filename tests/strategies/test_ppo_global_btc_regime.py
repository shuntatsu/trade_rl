from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl import ppo


class _CapturePolicy:
    def __init__(self) -> None:
        self.last_observation: np.ndarray | None = None

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[object, object]:
        assert deterministic is True
        self.last_observation = np.asarray(observation).copy()
        return np.asarray(1), None


def _market(
    *,
    reference_available: bool = True,
    reference_value: float = 0.25,
    reference_staleness: float = 0.4,
    future_reference_shift: float = 0.0,
) -> MarketDataset:
    close = np.asarray(
        [
            [100.0, 50.0],
            [101.0, 51.0],
            [102.0, 52.0],
            [103.0, 53.0],
        ],
        dtype=np.float64,
    )
    features = np.asarray(
        [
            [[reference_value, 10.0], [-9.0, 20.0]],
            [[0.30, 11.0], [-8.0, 21.0]],
            [[0.35 + future_reference_shift, 12.0], [-7.0, 22.0]],
            [[0.40 + future_reference_shift, 13.0], [-6.0, 23.0]],
        ],
        dtype=np.float32,
    )
    feature_available = np.ones_like(features, dtype=np.bool_)
    feature_available[0, 0, 0] = reference_available
    feature_staleness = np.zeros_like(features, dtype=np.float32)
    feature_staleness[0, 0, 0] = reference_staleness
    if not reference_available:
        feature_staleness[0, 0, 0] = 1.0

    return MarketDataset(
        dataset_id="d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f",
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.asarray(
            [[999.0], [998.0], [997.0], [996.0]], dtype=np.float32
        ),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((4, 2), 1_000_000.0),
        funding_rate=np.zeros((4, 2)),
        tradable=np.ones((4, 2), dtype=np.bool_),
        feature_available=feature_available,
        feature_names=("1h__log_return_24bar", "local"),
        global_feature_names=("must_not_be_used",),
        periods_per_year=8_760,
        feature_staleness=feature_staleness,
        global_feature_available=np.ones((4, 1), dtype=np.bool_),
    )


def _candidate_env(dataset: MarketDataset, *, symbol_index: int) -> object:
    return ppo.PPOTradingEnv(
        dataset,
        feature_indices=(1,),
        symbol_indices=(symbol_index,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        global_context=ppo.PPO_GLOBAL_BTC_REGIME_CONTEXT,
    )


def test_candidate_contract_matches_sealed_issue605_layout() -> None:
    assert ppo.PPO_GLOBAL_BTC_REGIME_CONTEXT == "ppo_global_btc_regime_context"
    assert ppo.ppo_observation_contract_payload(
        global_context=ppo.PPO_GLOBAL_BTC_REGIME_CONTEXT
    ) == {
        "schema_version": "ppo_observation_v3_global_btc_regime",
        "global_feature_names": [],
        "includes_local_feature_staleness": True,
        "global_context": "ppo_global_btc_regime_context",
        "reference_symbol": "BTCUSDT",
        "reference_feature": "1h__log_return_24bar",
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


def test_candidate_uses_same_row_btc_reference_and_broadcasts_identically() -> None:
    dataset = _market()
    btc_observation, _ = _candidate_env(dataset, symbol_index=0).reset(seed=1)
    eth_observation, _ = _candidate_env(dataset, symbol_index=1).reset(seed=1)

    np.testing.assert_array_equal(
        btc_observation[:3], np.asarray([10.0, 1.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        eth_observation[:3], np.asarray([20.0, 1.0, 0.0], dtype=np.float32)
    )
    expected_reference = np.asarray([0.25, 1.0, 0.4], dtype=np.float32)
    np.testing.assert_array_equal(btc_observation[3:6], expected_reference)
    np.testing.assert_array_equal(eth_observation[3:6], expected_reference)
    assert btc_observation.shape == (8,)
    assert eth_observation.shape == (8,)


def test_candidate_reference_is_point_in_time_and_ignores_dataset_globals() -> None:
    baseline, _ = _candidate_env(
        _market(future_reference_shift=0.0), symbol_index=1
    ).reset(seed=2)
    future_mutated, _ = _candidate_env(
        _market(future_reference_shift=777.0), symbol_index=1
    ).reset(seed=2)
    np.testing.assert_array_equal(baseline, future_mutated)
    assert baseline[3] == pytest.approx(0.25)


def test_candidate_later_row_does_not_read_previous_reference_row() -> None:
    baseline_dataset = _market(reference_value=0.25)
    previous_row_mutated_dataset = _market(reference_value=777.0)

    def later_row_observation(dataset: MarketDataset) -> np.ndarray:
        env = ppo.PPOTradingEnv(
            dataset,
            feature_indices=(1,),
            symbol_indices=(1,),
            start_index=1,
            stop_index=3,
            gross_budget=0.5,
            initial_capital=1_000.0,
            global_context=ppo.PPO_GLOBAL_BTC_REGIME_CONTEXT,
        )
        observation, _ = env.reset(seed=21)
        return observation

    baseline = later_row_observation(baseline_dataset)
    previous_row_mutated = later_row_observation(previous_row_mutated_dataset)

    np.testing.assert_array_equal(baseline, previous_row_mutated)
    assert baseline[3] == pytest.approx(0.30)


def test_candidate_fails_closed_for_unavailable_or_nonfinite_reference() -> None:
    unavailable, _ = _candidate_env(
        _market(reference_available=False, reference_value=123.0), symbol_index=1
    ).reset(seed=3)
    np.testing.assert_array_equal(
        unavailable[3:6], np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    )

    nonfinite_dataset = _market(
        reference_available=True,
        reference_value=0.25,
        reference_staleness=0.7,
    )
    corrupted_features = nonfinite_dataset.features.copy()
    corrupted_features[0, 0, 0] = np.nan
    corrupted_features.setflags(write=False)
    object.__setattr__(nonfinite_dataset, "features", corrupted_features)
    nonfinite, _ = _candidate_env(nonfinite_dataset, symbol_index=1).reset(seed=3)
    np.testing.assert_array_equal(
        nonfinite[3:6], np.asarray([0.0, 0.0, 0.7], dtype=np.float32)
    )


def test_training_and_inference_share_candidate_encoder_contract() -> None:
    dataset = _market()
    env_observation, _ = _candidate_env(dataset, symbol_index=1).reset(seed=4)
    capture = _CapturePolicy()
    strategy = ppo.PPOIntentStrategy(
        capture,
        feature_indices=(1,),
        dataset=dataset,
        global_context=ppo.PPO_GLOBAL_BTC_REGIME_CONTEXT,
    )
    observation = StrategyObservation(
        index=0,
        timestamp=dataset.timestamps[0],
        symbol="ETHUSDT",
        features=dataset.features[0, 1],
        feature_available=dataset.feature_available[0, 1],
        feature_staleness=dataset.resolved_array("feature_staleness")[0, 1],
        global_features=dataset.global_features[0],
        global_feature_available=dataset.resolved_array("global_feature_available")[0],
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )
    assert strategy.decide(observation) is PositionIntent.FLAT
    assert capture.last_observation is not None
    np.testing.assert_array_equal(capture.last_observation, env_observation)


def test_baseline_v2_contract_and_encoding_remain_exactly_unchanged() -> None:
    assert ppo.ppo_observation_contract_payload() == {
        "schema_version": "ppo_observation_v2",
        "global_feature_names": [],
        "includes_local_feature_staleness": True,
        "layout": [
            "local_values",
            "local_available",
            "local_staleness",
            "current_intent",
            "current_weight",
        ],
    }
    baseline = ppo.PPOTradingEnv(
        _market(),
        feature_indices=(1,),
        symbol_indices=(1,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )
    observation, _ = baseline.reset(seed=5)
    np.testing.assert_array_equal(
        observation,
        np.asarray([20.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
