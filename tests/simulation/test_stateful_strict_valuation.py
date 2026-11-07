"""Stateful account/funding valuation must use BookState's numerical authority."""

from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.orders.model import OrderBookState


def _subnormal_dataset() -> MarketDataset:
    rows = 3
    shape = (rows, 1)
    prices = np.full(shape, 1e-160)
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(rows) * np.timedelta64(1, "h"),
        features=np.zeros((rows, 1, 1), dtype=np.float32),
        global_features=np.zeros((rows, 1), dtype=np.float32),
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=np.full(shape, 100.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((rows, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        contract_multipliers=np.array([5e-164]),
        volume_units=(VolumeUnit.QUOTE_NOTIONAL,),
    )


@pytest.mark.parametrize("signed_position", (-1.0, 1.0))
def test_stateful_terminal_mark_uses_strict_safe_subnormal_valuation(
    signed_position: float,
) -> None:
    dataset = _subnormal_dataset()
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState(
        quantities=np.array([signed_position]),
        cash=1_000.0,
        mark_prices=dataset.close[0],
        peak_value=1_000.0,
        contract_multipliers=dataset.contract_multipliers,
    )
    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = executor.execute_orders(
            book, OrderBookState.empty(), (), start_index=0, bars=1
        )
        assert result.book.position_values[0] == signed_position * 5e-324
        assert result.book.portfolio_value == 1_000.0
    assert result.book.quantities.tolist() == [signed_position]
    assert result.book.cash == 1_000.0
    assert result.book.fill_count == 0
    assert result.book.returns_history == [0.0]
    assert book.quantities.tolist() == [signed_position]
    assert book.cash == 1_000.0
