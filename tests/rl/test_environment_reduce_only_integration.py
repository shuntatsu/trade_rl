from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import ActionMode, ActionSpec, AlphaContract, AlphaSignalKind
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


def _environment() -> ResidualMarketEnv:
    return ResidualMarketEnv(
        _market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        alpha_provider=_AlphaProvider(),
        alpha_enabled=True,
        alpha_contract=AlphaContract(kind=AlphaSignalKind.TARGET_WEIGHT),
        action_spec=ActionSpec(
            mode=ActionMode.ANCHORED_TARGET_RESIDUAL,
            alpha_enabled=True,
            risk_tilt_enabled=False,
            target_weight_count=2,
            residual_scale=0.1,
        ),
        config=ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            episode_bars=8,
            decision_every=1,
            signal_delay_decisions=1,
            reward=AbsoluteGrowthRewardConfig(),
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def test_environment_delays_reduce_only_mask_with_matching_hybrid_target() -> None:
    environment = _environment()
    environment.reset(options={"start_idx": 10, "initial_state_mode": "cash"})

    original_project = environment._risk_projector.project
    observed_masks: list[np.ndarray | None] = []

    def record_without_applying_reduce_only(request):
        observed_masks.append(
            None
            if request.reduce_only_mask is None
            else np.asarray(request.reduce_only_mask, dtype=np.bool_).copy()
        )
        return original_project(replace(request, reduce_only_mask=None))

    environment._risk_projector.project = record_without_applying_reduce_only  # type: ignore[method-assign]

    environment.set_next_hybrid_reduce_only_mask(np.asarray([True, False]))
    environment.step(np.zeros(2, dtype=np.float32))

    environment.set_next_hybrid_reduce_only_mask(np.asarray([False, False]))
    environment.step(np.zeros(2, dtype=np.float32))

    # Each step projects hybrid then shadow. The hybrid mask follows the target delay;
    # shadow never receives V10 reduce-only intent.
    assert observed_masks[0] is not None
    assert observed_masks[0].tolist() == [False, False]
    assert observed_masks[1] is None
    assert observed_masks[2] is not None
    assert observed_masks[2].tolist() == [True, False]
    assert observed_masks[3] is None
