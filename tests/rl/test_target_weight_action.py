from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import (
    ActionMode,
    ActionSpec,
    BaselineResidualComposer,
    TargetWeightAction,
)
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.rewards import AbsoluteGrowthRewardConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy, TrendTargets


def target_spec(count: int = 3) -> ActionSpec:
    return ActionSpec(
        mode=ActionMode.TARGET_WEIGHT,
        risk_tilt_enabled=False,
        target_weight_count=count,
    )


def test_direct_target_action_preserves_exact_policy_weights() -> None:
    action = target_spec().parse(np.array([0.35, -0.10, 0.0]))

    assert isinstance(action, TargetWeightAction)
    np.testing.assert_array_equal(action.weights, np.array([0.35, -0.10, 0.0]))
    np.testing.assert_array_equal(
        action.as_array(),
        np.array([0.35, -0.10, 0.0], dtype=np.float32),
    )


def test_zero_direct_target_means_flat_not_trend_baseline() -> None:
    spec = target_spec(count=2)
    action = spec.parse(np.zeros(2))
    trends = TrendTargets(
        fast=np.array([0.5, -0.5]),
        base=np.array([0.4, -0.4]),
        slow=np.array([0.2, -0.2]),
    )

    result = BaselineResidualComposer().compose(
        action,
        trends,
        np.zeros(2),
        alpha_enabled=False,
        max_gross=1.0,
    )

    np.testing.assert_array_equal(result.proposal, np.zeros(2))
    np.testing.assert_array_equal(result.baseline, trends.base)


@pytest.mark.parametrize(
    "value, message",
    [
        (np.array([0.1, 0.2]), "exactly 3"),
        (np.array([0.1, np.nan, 0.0]), "finite"),
    ],
)
def test_direct_target_action_rejects_invalid_vectors(
    value: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        target_spec().parse(value)


def test_target_action_mode_rejects_residual_controls() -> None:
    with pytest.raises(ValueError, match="residual controls"):
        ActionSpec(mode="target_weight", target_weight_count=3)
    with pytest.raises(ValueError, match="positive target_weight_count"):
        ActionSpec(
            mode="target_weight",
            risk_tilt_enabled=False,
            target_weight_count=0,
        )


def market() -> MarketDataset:
    n_bars = 40
    close = np.column_stack(
        [
            np.linspace(100.0, 120.0, n_bars),
            np.linspace(100.0, 90.0, n_bars),
        ]
    )
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=np.vstack([close[0], close[:-1]]),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def environment(spec: ActionSpec) -> ResidualMarketEnv:
    return ResidualMarketEnv(
        market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        action_spec=spec,
        config=ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            episode_bars=8,
            decision_every=1,
            reward=AbsoluteGrowthRewardConfig(),
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def test_environment_binds_target_actions_to_dataset_symbols() -> None:
    direct = environment(target_spec(count=2))
    residual = environment(ActionSpec())

    assert direct.action_names == (
        "target_weight:BTCUSDT",
        "target_weight:ETHUSDT",
    )
    assert direct.action_space.shape == (2,)
    assert direct.environment_digest != residual.environment_digest
    with pytest.raises(ValueError, match="target weight count"):
        environment(target_spec(count=3))
