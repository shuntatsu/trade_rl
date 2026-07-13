from __future__ import annotations

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState, EconomicTerminationReason


def test_book_can_represent_leveraged_signed_weights() -> None:
    book = BookState.from_weights(
        weights=np.array([1.2, -0.4]),
        capital=1_000.0,
        prices=np.array([100.0, 100.0]),
        max_gross=2.0,
    )
    assert book.gross_exposure == pytest.approx(1.6)
    assert book.portfolio_value == pytest.approx(1_000.0)


def test_economic_insolvency_is_state_not_uncaught_exception() -> None:
    book = BookState.zero(1, 100.0, np.array([10.0]))
    book.execute(
        fill_prices=np.array([10.0]),
        target_quantities=np.array([0.0]),
        cost_amount=100.0,
        turnover=0.0,
    )
    assert book.insolvent is True
    assert (
        book.termination_reason is EconomicTerminationReason.EXECUTION_COST_EXHAUSTION
    )


def test_maintenance_requirement_causes_margin_call() -> None:
    book = BookState.from_weights(
        weights=np.array([1.0]), capital=100.0, prices=np.array([100.0])
    )
    book.set_margin(
        margin_used=50.0, maintenance_margin=0.25, maintenance_requirement=120.0
    )
    assert book.termination_reason is EconomicTerminationReason.MARGIN_CALL


def test_split_dividend_and_delisting_settlement_reconcile() -> None:
    book = BookState.from_weights(
        weights=np.array([0.5]), capital=1_000.0, prices=np.array([100.0])
    )
    book.apply_split(np.array([2.0]))
    assert book.quantities[0] == pytest.approx(10.0)
    assert book.apply_dividend(np.array([1.0])) == pytest.approx(10.0)
    proceeds = book.settle_positions(
        mask=np.array([True]), prices=np.array([50.0]), recovery=np.array([0.8])
    )
    assert proceeds == pytest.approx(400.0)
    assert book.quantities[0] == 0.0
