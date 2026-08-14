from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import (
    ActionMode,
    ActionSpec,
    AlphaContract,
    AlphaSignalKind,
)
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.rewards import AbsoluteGrowthRewardConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


class _AlphaProvider:
    artifact_digest = "b" * 64

    def predict_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        del index
        return np.asarray([0.4, -0.2], dtype=np.float64)[: dataset.n_symbols]


def _market() -> MarketDataset:
    n_bars = 40
    close = np.column_stack(
        (
            np.linspace(100.0, 120.0, n_bars),
            np.linspace(100.0, 90.0, n_bars),
        )
    )
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=np.vstack((close[0], close[:-1])),
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


def _spec() -> ActionSpec:
    return ActionSpec(
        mode=ActionMode.ANCHORED_TARGET_RESIDUAL,
        alpha_enabled=True,
        risk_tilt_enabled=False,
        target_weight_count=2,
        residual_scale=0.1,
    )


def _environment(alpha_contract: AlphaContract) -> ResidualMarketEnv:
    return ResidualMarketEnv(
        _market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        alpha_provider=_AlphaProvider(),
        alpha_enabled=True,
        alpha_contract=alpha_contract,
        action_spec=_spec(),
        config=ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            episode_bars=8,
            decision_every=1,
            reward=AbsoluteGrowthRewardConfig(),
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def test_anchored_environment_requires_target_weight_alpha_semantics() -> None:
    with pytest.raises(ValueError, match="target-weight alpha"):
        _environment(AlphaContract(kind=AlphaSignalKind.EXPECTED_RETURN))


def test_anchored_environment_binds_policy_space_baseline_to_zero_residual() -> None:
    environment = _environment(AlphaContract(kind=AlphaSignalKind.TARGET_WEIGHT))
    environment.reset(options={"start_idx": 10, "initial_state_mode": "cash"})

    assert environment.action_names == (
        "anchored_residual:BTCUSDT",
        "anchored_residual:ETHUSDT",
    )
    assert environment.baseline_action().tolist() == pytest.approx([0.0, 0.0])
