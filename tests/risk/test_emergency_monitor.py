from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.risk.emergency import CausalEmergencyRiskMonitor, EmergencyRiskConfig
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def market_with_last_bar_shock() -> MarketDataset:
    n_bars = 16
    close = np.full((n_bars, 2), 100.0)
    close[-1, 0] = 95.0
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="f" * 64,
        symbols=("BTC", "ETH"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def test_stop_loss_uses_only_completed_history_through_current_bar() -> None:
    dataset = market_with_last_bar_shock()
    monitor = CausalEmergencyRiskMonitor(
        EmergencyRiskConfig(stop_loss_return=0.03, stop_loss_hours=1.0)
    )

    before = monitor.assess(dataset, index=dataset.n_bars - 2, weights=np.array([0.4, 0.0]))
    after = monitor.assess(dataset, index=dataset.n_bars - 1, weights=np.array([0.4, 0.0]))

    np.testing.assert_array_equal(before.flatten_mask, np.array([False, False]))
    np.testing.assert_array_equal(after.flatten_mask, np.array([True, False]))
    assert after.reasons == ("stop_loss:BTC",)


def test_gap_and_untradable_checks_are_symbol_local() -> None:
    dataset = market_with_last_bar_shock()
    open_price = dataset.open.copy()
    tradable = dataset.tradable.copy()
    open_price[-1, 1] = 106.0
    tradable[-1, 0] = False
    shocked = replace(
        dataset,
        open=open_price,
        high=np.maximum(dataset.high, open_price),
        low=np.minimum(dataset.low, open_price),
        tradable=tradable,
    )
    monitor = CausalEmergencyRiskMonitor(
        EmergencyRiskConfig(gap_return=0.04, flatten_untradable=True)
    )

    result = monitor.assess(shocked, index=shocked.n_bars - 1, weights=np.array([0.3, -0.2]))

    np.testing.assert_array_equal(result.flatten_mask, np.array([True, True]))
    assert set(result.reasons) == {"untradable:BTC", "gap:ETH"}


def test_environment_emergency_exit_bypasses_ordinary_turnover_limit() -> None:
    dataset = market_with_last_bar_shock()
    env = ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        pre_trade_risk=PreTradeRisk(
            PreTradeRiskConfig(max_turnover=0.0, max_abs_weight=1.0)
        ),
        config=ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            episode_bars=4,
            decision_every=1,
            emergency_risk=EmergencyRiskConfig(
                stop_loss_return=0.03,
                stop_loss_hours=1.0,
            ),
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )
    env.current_index = dataset.n_bars - 1
    book = BookState.from_weights(
        weights=np.array([0.40, 0.0]),
        capital=100_000.0,
        prices=dataset.close[-1],
    )

    result = env._constrain_target(np.array([0.40, 0.0]), book)

    np.testing.assert_array_equal(result.weights, np.zeros(2))
    assert "stop_loss:BTC" in result.reasons
    assert result.turnover_overridden is True
