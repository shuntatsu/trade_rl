from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import OrderBookState


def market(**overrides: object) -> MarketDataset:
    n_bars = 4
    shape = (n_bars, 1)
    close = np.full(shape, 100.0, dtype=np.float64)
    values: dict[str, object] = {
        "dataset_id": "9" * 64,
        "symbols": ("BTCUSDT",),
        "timestamps": np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        "features": np.zeros((n_bars, 1, 1), dtype=np.float32),
        "global_features": np.zeros((n_bars, 1), dtype=np.float32),
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close,
        "volume": np.full(shape, 1_000.0, dtype=np.float64),
        "funding_rate": np.zeros(shape, dtype=np.float64),
        "tradable": np.ones(shape, dtype=np.bool_),
        "feature_available": np.ones((n_bars, 1, 1), dtype=np.bool_),
        "feature_names": ("signal",),
        "global_feature_names": ("regime",),
        "periods_per_year": 8_760,
        "contract_multipliers": np.array([1.0], dtype=np.float64),
    }
    values.update(overrides)
    return MarketDataset(**values)


def test_golden_01_cash() -> None:
    book = BookState.zero(1, 1_000.0, np.array([100.0]))
    net_return = book.mark_to_market(
        mark_prices=np.array([120.0]),
        funding_amount=0.0,
        period_start_value=1_000.0,
    )

    assert book.cash == pytest.approx(1_000.0)
    assert book.portfolio_value == pytest.approx(1_000.0)
    assert net_return == pytest.approx(0.0)


def test_golden_02_constant_long() -> None:
    book = BookState.from_weights(
        weights=np.array([1.0]),
        capital=1_000.0,
        prices=np.array([100.0]),
    )
    net_return = book.mark_to_market(
        mark_prices=np.array([110.0]),
        funding_amount=0.0,
        period_start_value=1_000.0,
    )

    assert book.quantities == pytest.approx([10.0])
    assert book.cash == pytest.approx(0.0)
    assert book.portfolio_value == pytest.approx(1_100.0)
    assert net_return == pytest.approx(0.10)


def test_golden_03_constant_short() -> None:
    book = BookState.from_weights(
        weights=np.array([-1.0]),
        capital=1_000.0,
        prices=np.array([100.0]),
    )
    net_return = book.mark_to_market(
        mark_prices=np.array([90.0]),
        funding_amount=0.0,
        period_start_value=1_000.0,
    )

    assert book.quantities == pytest.approx([-10.0])
    assert book.cash == pytest.approx(2_000.0)
    assert book.portfolio_value == pytest.approx(1_100.0)
    assert net_return == pytest.approx(0.10)


def test_golden_04_one_entry_one_exit() -> None:
    book = BookState.zero(1, 1_000.0, np.array([100.0]))
    book.execute(
        fill_prices=np.array([100.0]),
        target_quantities=np.array([5.0]),
        cost_amount=0.0,
        turnover=0.5,
    )
    book.mark_to_market(
        mark_prices=np.array([110.0]),
        funding_amount=0.0,
        period_start_value=1_000.0,
    )
    book.execute(
        fill_prices=np.array([110.0]),
        target_quantities=np.array([0.0]),
        cost_amount=0.0,
        turnover=0.5,
    )

    assert book.quantities == pytest.approx([0.0])
    assert book.cash == pytest.approx(1_050.0)
    assert book.portfolio_value == pytest.approx(1_050.0)
    assert book.n_trades == 2
    assert book.rebalance_events == 2


def test_golden_05_partial_fill() -> None:
    dataset = market(volume=np.full((4, 1), 4.0, dtype=np.float64))
    executor = MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), max_participation_rate=0.5),
    )
    result = executor.execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )

    assert result.requested_notional_by_symbol == pytest.approx([1_000.0])
    assert result.filled_notional_by_symbol == pytest.approx([200.0])
    assert result.book.quantities == pytest.approx([2.0])
    assert result.book.cash == pytest.approx(800.0)
    assert result.book.portfolio_value == pytest.approx(1_000.0)
    assert result.fill_ratio == pytest.approx(0.20)


def test_golden_06_no_fill() -> None:
    dataset = market(
        low=np.full((4, 1), 99.9, dtype=np.float64),
        high=np.full((4, 1), 100.1, dtype=np.float64),
    )
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            order_type="limit",
            limit_offset_rate=0.01,
        ),
    )
    result = executor.execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )

    assert result.filled_turnover == pytest.approx(0.0)
    assert result.book.quantities == pytest.approx([0.0])
    assert result.book.cash == pytest.approx(1_000.0)


def funding_market() -> MarketDataset:
    mark_price = np.full((4, 1), 100.0, dtype=np.float64)
    mark_price[1, 0] = 120.0
    funding_rate = np.zeros((4, 1), dtype=np.float64)
    funding_rate[1, 0] = 0.001
    funding_due = np.zeros((4, 1), dtype=np.bool_)
    funding_due[1, 0] = True
    return market(
        high=np.maximum(np.full((4, 1), 100.0), mark_price),
        low=np.minimum(np.full((4, 1), 100.0), mark_price),
        mark_price=mark_price,
        funding_rate=funding_rate,
        funding_due=funding_due,
    )


def funding_result(weight: float):
    dataset = funding_market()
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState.from_weights(
        weights=np.array([weight]),
        capital=1_000.0,
        prices=dataset.close[0],
        contract_multipliers=dataset.contract_multipliers,
    )
    return executor.execute_orders(
        book,
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=1,
    )


def test_golden_07_funding_payment() -> None:
    result = funding_result(1.0)

    assert result.interval_funding == pytest.approx(-1.2)
    assert result.book.funding_pnl == pytest.approx(-1.2)
    assert result.book.portfolio_value == pytest.approx(1_198.8)


def test_golden_08_funding_receipt() -> None:
    result = funding_result(-1.0)

    assert result.interval_funding == pytest.approx(1.2)
    assert result.book.funding_pnl == pytest.approx(1.2)
    assert result.book.portfolio_value == pytest.approx(801.2)


def test_golden_09_drawdown_emergency_reduction() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=0.0,
            drawdown_start=0.10,
            drawdown_stop=0.20,
        )
    )
    constrained = risk.constrain(
        np.array([0.5]),
        current=np.array([0.5]),
        drawdown=0.25,
    )
    book = BookState.from_weights(
        weights=np.array([0.5]),
        capital=1_000.0,
        prices=np.array([100.0]),
    )
    book.execute(
        fill_prices=np.array([100.0]),
        target_quantities=np.array([0.0]),
        cost_amount=0.0,
        turnover=0.5,
    )

    assert constrained.weights == pytest.approx([0.0])
    assert constrained.turnover_overridden is True
    assert book.quantities == pytest.approx([0.0])
    assert book.portfolio_value == pytest.approx(1_000.0)


def test_golden_10_terminal_mark_to_market_without_forced_close() -> None:
    book = BookState.zero(1, 1_000.0, np.array([100.0]))
    book.execute(
        fill_prices=np.array([100.0]),
        target_quantities=np.array([5.0]),
        cost_amount=0.0,
        turnover=0.5,
    )
    net_return = book.mark_to_market(
        mark_prices=np.array([120.0]),
        funding_amount=0.0,
        period_start_value=1_000.0,
    )

    assert book.quantities == pytest.approx([5.0])
    assert book.cash == pytest.approx(500.0)
    assert book.portfolio_value == pytest.approx(1_100.0)
    assert net_return == pytest.approx(0.10)
