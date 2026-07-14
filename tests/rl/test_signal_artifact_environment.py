from __future__ import annotations

from pathlib import Path

import numpy as np

from trade_rl.artifacts.signals import write_signal_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.integrations.signal_artifacts import load_alpha_artifact
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _dataset() -> MarketDataset:
    n_bars = 80
    close = (100.0 + np.arange(n_bars, dtype=np.float64))[:, None]
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTC",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_valid_episode_starts_respect_signal_fit_boundary(tmp_path: Path) -> None:
    dataset = _dataset()
    write_signal_artifact(
        tmp_path,
        kind="alpha",
        dataset_id=dataset.dataset_id,
        fit_start=0,
        fit_stop=20,
        names=dataset.symbols,
        values=np.zeros((dataset.n_bars, dataset.n_symbols)),
    )
    provider = load_alpha_artifact(
        tmp_path,
        dataset_id=dataset.dataset_id,
        expected_symbols=dataset.symbols,
    )
    env = ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=1, base_lookback=2, slow_lookback=3)
        ),
        alpha_provider=provider,
        alpha_enabled=True,
        action_spec=ActionSpec(alpha_enabled=True),
        config=ResidualMarketEnvConfig(
            episode_bars=4,
            decision_every=1,
            initial_capital=1_000.0,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )

    assert int(env._valid_starts(hours=4.0, bars=4).min()) == 20
