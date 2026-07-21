from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.execution_sensitivity import (
    default_execution_sensitivity_scenarios,
    evaluate_execution_sensitivity,
)
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig


def _dataset() -> MarketDataset:
    n = 8
    open_price = np.column_stack(
        (
            np.linspace(100.0, 114.0, n),
            np.linspace(50.0, 43.0, n),
        )
    )
    close = open_price * np.asarray((1.01, 0.99))
    high = np.maximum(open_price, close) * 1.003
    low = np.minimum(open_price, close) * 0.997
    tradable = np.ones((n, 2), dtype=np.bool_)
    tradable[0, 1] = False
    return MarketDataset(
        dataset_id="9" * 64,
        symbols=("UP", "DOWN"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=np.zeros((n, 2, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=np.full((n, 2), 20.0),
        funding_rate=np.zeros((n, 2)),
        tradable=tradable,
        feature_available=np.ones((n, 2, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_default_sensitivity_matrix_covers_requested_assumptions() -> None:
    names = {scenario.name for scenario in default_execution_sensitivity_scenarios()}
    assert {
        "fee_1x",
        "fee_2x",
        "fee_4x",
        "spread_1x",
        "spread_2x",
        "slippage_1x",
        "slippage_2x",
        "slippage_4x",
        "capacity_100pct",
        "capacity_50pct",
        "capacity_25pct",
        "signal_delay_0",
        "signal_delay_1",
        "signal_delay_2",
        "limit_optimistic",
        "limit_neutral",
        "limit_conservative",
        "tradability_delay_0",
        "tradability_delay_1",
    } <= names


def test_sensitivity_results_expose_cost_capacity_delay_and_fill_dependence() -> None:
    dataset = _dataset()
    results = evaluate_execution_sensitivity(
        dataset=dataset,
        initial_book=BookState.zero(2, 10_000.0, dataset.close[0]),
        target=np.asarray((0.6, -0.3)),
        start_index=0,
        horizon_bars=5,
        base_cost=ExecutionCostConfig(
            fee_rate=0.0005,
            spread_rate=0.0002,
            impact_rate=0.0001,
            max_participation_rate=0.5,
            slippage_std=0.0002,
            random_seed=42,
            max_leverage=1.0,
            maintenance_margin_rate=0.0,
        ),
    )
    by_name = {item.scenario.name: item for item in results}

    assert by_name["fee_4x"].interval_cost > by_name["fee_1x"].interval_cost
    assert by_name["fee_4x"].total_return < by_name["fee_1x"].total_return
    assert by_name["spread_2x"].interval_cost > by_name["spread_1x"].interval_cost
    assert by_name["slippage_4x"].interval_cost > by_name["slippage_1x"].interval_cost
    assert by_name["capacity_25pct"].fill_ratio < by_name["capacity_100pct"].fill_ratio
    assert by_name["capacity_25pct"].unfilled_turnover > by_name["capacity_100pct"].unfilled_turnover
    assert by_name["signal_delay_2"].total_return != by_name["signal_delay_0"].total_return
    assert by_name["limit_conservative"].fill_ratio <= by_name["limit_optimistic"].fill_ratio
    assert by_name["tradability_delay_1"].fill_ratio <= by_name["tradability_delay_0"].fill_ratio
