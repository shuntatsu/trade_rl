from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl import ppo
from trade_rl.strategies.rl.ppo import PPOIntentStrategy, PPOTradingEnv


class _FlatPolicy:
    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[object, object]:
        del observation
        assert deterministic is True
        return np.asarray(1), None


def _market(
    *,
    symbol: str = "BTCUSDT",
    future_shift: float = 0.0,
    global_shift: float = 0.0,
) -> MarketDataset:
    close = np.asarray([[100.0], [101.0], [102.0], [103.0]])
    features = np.asarray(
        [
            [[10.0, 20.0]],
            [[11.0, 21.0]],
            [[12.0 + future_shift, 22.0 + future_shift]],
            [[13.0 + future_shift, 23.0 + future_shift]],
        ],
        dtype=np.float32,
    )
    feature_available = np.ones_like(features, dtype=np.bool_)
    feature_available[0, 0, 1] = False
    feature_staleness = np.asarray(
        [
            [[0.25, 1.0]],
            [[0.0, 0.5]],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    global_features = np.asarray(
        [
            [0.4 + global_shift],
            [0.5 + global_shift],
            [0.6 + global_shift],
            [0.7 + global_shift],
        ],
        dtype=np.float32,
    )

    return MarketDataset(
        dataset_id="9" * 64,
        symbols=(symbol,),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=features,
        global_features=global_features,
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((4, 1), 1_000_000.0),
        funding_rate=np.zeros((4, 1)),
        tradable=np.ones((4, 1), dtype=np.bool_),
        feature_available=feature_available,
        feature_names=("fast", "slow"),
        global_feature_names=("arbitrary_cross_symbol_context",),
        periods_per_year=8_760,
        feature_staleness=feature_staleness,
        global_feature_available=np.ones((4, 1), dtype=np.bool_),
    )


def _env(dataset: MarketDataset) -> PPOTradingEnv:
    return PPOTradingEnv(
        dataset,
        feature_indices=(1, 0),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )


def _legacy_observation_without_staleness() -> StrategyObservation:
    return StrategyObservation(
        index=0,
        timestamp=np.datetime64("2026-01-01", "ns"),
        symbol="BTCUSDT",
        features=np.asarray([0.5]),
        feature_available=np.asarray([True]),
        global_features=np.asarray([999.0]),
        global_feature_available=np.asarray([True]),
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )


def test_observation_v2_contract_is_fixed_before_m2() -> None:
    assert ppo.PPO_OBSERVATION_SCHEMA == "ppo_observation_v2"
    assert ppo.PPO_GLOBAL_FEATURE_NAMES == ()
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


def test_observation_v2_encodes_exact_order_masks_and_staleness() -> None:
    observation, _ = _env(_market()).reset(seed=7)

    expected = np.asarray(
        [
            0.0,
            10.0,
            0.0,
            1.0,
            1.0,
            0.25,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(observation, expected)
    assert observation.shape == (8,)
    assert observation.flags.writeable is True


def test_observation_v2_does_not_depend_on_cross_symbol_global_aggregates() -> None:
    baseline, _ = _env(_market(global_shift=0.0)).reset(seed=11)
    mutated_globals, _ = _env(_market(global_shift=999.0)).reset(seed=11)

    np.testing.assert_array_equal(baseline, mutated_globals)


def test_observation_v2_is_symbol_name_invariant() -> None:
    first, _ = _env(_market(symbol="BTCUSDT")).reset(seed=3)
    renamed, _ = _env(_market(symbol="RENAMED")).reset(seed=3)

    np.testing.assert_array_equal(first, renamed)


def test_observation_v2_does_not_read_future_rows() -> None:
    baseline, _ = _env(_market(future_shift=0.0)).reset(seed=5)
    mutated_future, _ = _env(_market(future_shift=999.0)).reset(seed=5)

    np.testing.assert_array_equal(baseline, mutated_future)


def test_public_strategy_observation_constructor_remains_backward_compatible() -> None:
    observation = _legacy_observation_without_staleness()

    assert observation.feature_staleness is None


def test_ppo_v2_rejects_legacy_observation_without_staleness() -> None:
    strategy = PPOIntentStrategy(_FlatPolicy(), feature_indices=(0,))

    with pytest.raises(ValueError, match="requires feature staleness"):
        strategy.decide(_legacy_observation_without_staleness())
