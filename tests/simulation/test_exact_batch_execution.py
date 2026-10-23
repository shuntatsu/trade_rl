from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState


def _book_with_hidden_exact_tails() -> BookState:
    quantities = np.array([1.2345678901234567, -2.345678901234567])
    marks = np.array([100.0, 100.0])
    cash = 1_000.0 - float(np.sum(quantities * marks))
    book = BookState(
        quantities=quantities,
        cash=cash,
        mark_prices=marks,
        peak_value=1_000.0,
    )
    book.apply_split(np.array([1.1, 1.1]))
    book.refresh_drawdown()
    return book


def _snapshot(book: BookState) -> tuple[object, ...]:
    return (
        book.exact_quantities,
        book.cash,
        tuple(book.mark_prices),
        book.peak_value,
        book.max_drawdown,
        book.turnover_total,
        book.total_cost,
        book.funding_pnl,
        book.fill_count,
        book.rebalance_events,
        tuple(book.returns_history),
        book.borrow_cost,
        book.margin_used,
        book.maintenance_margin,
        book.maintenance_requirement,
        book.margin_deficit,
        book.insolvent,
        book.termination_reason,
    )


def test_exact_batch_deltas_preserve_tails_and_record_one_shared_execution() -> None:
    book = _book_with_hidden_exact_tails()
    before = book.exact_quantities
    assert any(
        Fraction(str(float(view))) != exact
        for exact, view in zip(before, book.quantities, strict=True)
    )
    starting_cash = book.cash
    deltas = (Fraction(-1, 4), Fraction(1, 2))

    book.execute_quantity_deltas(
        fill_prices=np.array([100.0, 200.0]),
        quantity_deltas=deltas,
        cost_amount=7.5,
        turnover=0.125,
        valuation_prices=np.array([101.0, 201.0]),
    )

    assert book.exact_quantities == tuple(
        quantity + delta for quantity, delta in zip(before, deltas, strict=True)
    )
    assert book.fill_count == 2
    assert book.rebalance_events == 1
    assert book.cash == pytest.approx(starting_cash - 82.5)
    assert book.total_cost == pytest.approx(7.5)
    assert book.turnover_total == pytest.approx(0.125)
    np.testing.assert_allclose(book.mark_prices, [101.0, 201.0])


def test_zero_batch_deltas_preserve_inventory_and_execution_counters() -> None:
    book = _book_with_hidden_exact_tails()
    before = _snapshot(book)

    book.execute_quantity_deltas(
        fill_prices=book.mark_prices.copy(),
        quantity_deltas=(Fraction(0), Fraction(0)),
        cost_amount=0.0,
        turnover=0.0,
        valuation_prices=book.mark_prices.copy(),
    )

    assert _snapshot(book) == before


@pytest.mark.parametrize(
    "quantity_deltas",
    [
        pytest.param([Fraction(1, 4), Fraction(1, 2)], id="list-not-tuple"),
        pytest.param(0.25, id="scalar-float"),
        pytest.param(True, id="scalar-bool"),
        pytest.param((Fraction(1, 4),), id="wrong-dimension"),
        pytest.param((Fraction(1, 4), 0.5), id="float-member"),
        pytest.param((Fraction(1, 4), True), id="bool-member"),
    ],
)
def test_invalid_batch_deltas_leave_book_unchanged(quantity_deltas: object) -> None:
    book = _book_with_hidden_exact_tails()
    before = _snapshot(book)

    with pytest.raises((TypeError, ValueError)):
        book.execute_quantity_deltas(
            fill_prices=np.array([100.0, 200.0]),
            quantity_deltas=quantity_deltas,
            cost_amount=7.5,
            turnover=0.125,
            valuation_prices=np.array([101.0, 201.0]),
        )

    assert _snapshot(book) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("cost_amount", float("nan"), id="nonfinite-cost"),
        pytest.param("cost_amount", -1.0, id="negative-cost"),
        pytest.param(
            "fill_prices", np.array([100.0, float("nan")]), id="nonfinite-fill-price"
        ),
        pytest.param(
            "valuation_prices", np.array([101.0, 0.0]), id="nonpositive-valuation-price"
        ),
    ],
)
def test_invalid_batch_accounting_inputs_leave_book_unchanged(
    field: str, value: object
) -> None:
    book = _book_with_hidden_exact_tails()
    before = _snapshot(book)
    arguments: dict[str, object] = {
        "fill_prices": np.array([100.0, 200.0]),
        "quantity_deltas": (Fraction(-1, 4), Fraction(1, 2)),
        "cost_amount": 7.5,
        "turnover": 0.125,
        "valuation_prices": np.array([101.0, 201.0]),
    }
    arguments[field] = value

    with pytest.raises((TypeError, ValueError)):
        book.execute_quantity_deltas(**arguments)

    assert _snapshot(book) == before
