from __future__ import annotations

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState


def test_zero_book_is_cash_only_at_initial_marks() -> None:
    book = BookState.zero(
        n_symbols=2,
        initial_capital=1_000.0,
        initial_prices=np.array([100.0, 200.0]),
    )

    np.testing.assert_allclose(book.quantities, np.zeros(2))
    np.testing.assert_allclose(book.weights, np.zeros(2))
    assert book.cash == pytest.approx(1_000.0)
    assert book.portfolio_value == pytest.approx(1_000.0)


def test_execution_and_mark_to_market_reconcile_cash_quantities_and_equity() -> None:
    book = BookState.zero(
        n_symbols=2,
        initial_capital=1_000.0,
        initial_prices=np.array([100.0, 200.0]),
    )

    book.execute(
        fill_prices=np.array([100.0, 200.0]),
        target_quantities=np.array([5.0, -1.0]),
        cost_amount=10.0,
        turnover=0.7,
    )

    assert book.cash == pytest.approx(690.0)
    assert book.portfolio_value == pytest.approx(990.0)
    assert book.max_drawdown == pytest.approx(0.01)
    assert book.fill_count == 2
    assert book.rebalance_events == 1

    book.mark_to_market(
        mark_prices=np.array([110.0, 180.0]),
        funding_amount=5.0,
    )

    assert book.cash == pytest.approx(695.0)
    assert book.portfolio_value == pytest.approx(1_065.0)
    np.testing.assert_allclose(
        book.weights,
        np.array([550.0 / 1_065.0, -180.0 / 1_065.0]),
    )
    assert book.returns_history[-1] == pytest.approx(1_065.0 / 990.0 - 1.0)
    assert book.funding_pnl == pytest.approx(5.0)


def test_execution_without_quantity_change_does_not_increment_counters() -> None:
    book = BookState.zero(
        n_symbols=1,
        initial_capital=100.0,
        initial_prices=np.array([10.0]),
    )

    book.execute(
        fill_prices=np.array([10.0]),
        target_quantities=np.array([0.0]),
        cost_amount=0.0,
        turnover=0.0,
    )

    assert book.fill_count == 0
    assert book.rebalance_events == 0
