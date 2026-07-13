from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def market(
    *,
    quantity_step: float = 0.0,
    min_notional: float = 0.0,
    maintenance_margin_rate: float = 0.005,
) -> MarketDataset:
    close = np.array([[100.0], [100.0], [50.0], [50.0]])
    open_price = np.array([[100.0], [100.0], [100.0], [50.0]])
    return MarketDataset(
        dataset_id="c" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T01:00:00", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=np.zeros((4, 1, 1), dtype=np.float32),
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full((4, 1), 1_000_000.0),
        funding_rate=np.zeros((4, 1)),
        tradable=np.ones((4, 1), dtype=np.bool_),
        feature_available=np.ones((4, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        mark_price=close,
        quantity_step=np.array([quantity_step]),
        min_notional=np.array([min_notional]),
        maintenance_margin_rate=np.array([maintenance_margin_rate]),
    )


def test_quantity_step_rounds_fill_toward_zero() -> None:
    dataset = market(quantity_step=0.3)
    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
        BookState.zero(1, 100.0, dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )

    assert result.book.quantities[0] == pytest.approx(0.9)
    assert result.filled_turnover == pytest.approx(0.9)


def test_min_notional_blocks_new_tiny_position() -> None:
    dataset = market(min_notional=150.0)
    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).execute_interval(
        BookState.zero(1, 100.0, dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )

    assert result.fill_count == 0
    assert result.unfilled_turnover == pytest.approx(1.0)


def test_mark_price_margin_breach_forces_liquidation() -> None:
    dataset = market(maintenance_margin_rate=0.10)
    result = MarketExecutor(
        dataset,
        ExecutionCostConfig(
            fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            liquidation_fee_rate=0.01,
            max_participation_rate=1.0,
        ),
    ).execute_interval(
        BookState.zero(1, 100.0, dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=2,
    )

    assert result.liquidation_count == 1
    assert result.book.liquidation_count == 1
    np.testing.assert_allclose(result.book.quantities, np.zeros(1))
    assert result.book.total_cost == pytest.approx(0.5)


def test_execution_cost_randomization_is_seeded() -> None:
    dataset = market()
    config = ExecutionCostConfig(
        fee_rate=0.001,
        spread_rate=0.001,
        impact_rate=0.001,
        fee_multiplier_range=(0.5, 1.5),
        spread_multiplier_range=(0.5, 1.5),
        impact_multiplier_range=(0.5, 1.5),
        random_seed=7,
    )
    first = MarketExecutor(dataset, config)
    second = MarketExecutor(dataset, config)

    assert first.runtime_cost_multipliers == second.runtime_cost_multipliers
    first.reset_random_state(8)
    assert first.runtime_cost_multipliers != second.runtime_cost_multipliers
