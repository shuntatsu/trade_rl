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


@pytest.mark.parametrize("strict", (False, True))
def test_mutated_active_position_valuation_rejects_underflow_to_zero(
    strict: bool,
) -> None:
    book = BookState.zero(
        1,
        1_000.0,
        np.array([1e-200]),
        contract_multipliers=np.array([1e-200]),
    )
    book.quantities[0] = 1e-200
    with np.errstate(all="raise" if strict else "ignore"):
        with pytest.raises(ValueError, match="nonzero position value"):
            _ = book.position_values
        with pytest.raises(ValueError, match="nonzero position value"):
            _ = book.portfolio_value
    assert book.quantities.tolist() == [1e-200]
    assert book.cash == 1_000.0


def test_mutated_cash_and_position_cannot_silently_report_infinite_equity() -> None:
    book = BookState.zero(1, 1_000.0, np.ones(1))
    book.quantities[0] = 1e308
    book.cash = 1e308
    with np.errstate(all="ignore"):
        with pytest.raises(ValueError, match="portfolio value became non-finite"):
            _ = book.portfolio_value
    assert book.cash == 1e308
    assert book.quantities.tolist() == [1e308]


@pytest.mark.parametrize("quantity", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("strict", (False, True))
def test_mutated_nonfinite_position_read_fails_closed(
    quantity: float, strict: bool
) -> None:
    book = BookState.zero(1, 1_000.0, np.ones(1))
    book.quantities[0] = quantity
    with np.errstate(all="raise" if strict else "ignore"):
        with pytest.raises(
            ValueError, match="proposed account state became non-finite"
        ):
            _ = book.position_values
        with pytest.raises(ValueError):
            _ = book.portfolio_value
    assert book.cash == 1_000.0
