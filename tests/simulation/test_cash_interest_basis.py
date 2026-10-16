from __future__ import annotations

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState


def _book(weight: float) -> BookState:
    return BookState.from_weights(
        weights=np.array([weight]),
        capital=1_000.0,
        prices=np.array([100.0]),
        max_gross=2.0,
    )


def _snapshot(book: BookState) -> tuple[object, ...]:
    return (
        book.cash,
        book.portfolio_value,
        book.peak_value,
        book.max_drawdown,
        book.exact_quantities,
        tuple(book.returns_history),
    )


def test_explicit_positive_cash_basis_changes_interest_not_account_cash_basis() -> None:
    book = _book(0.5)
    original_cash = book.cash
    initial_peak = book.peak_value
    initial_drawdown = book.max_drawdown
    expected_interest = 505.0 * 0.365 * (1.0 / (365.0 * 24.0))

    interest = book.apply_cash_interest(
        0.365,
        year_fraction=1.0 / (365.0 * 24.0),
        cash_basis=505.0,
    )

    assert interest == pytest.approx(expected_interest)
    assert book.cash == pytest.approx(original_cash + expected_interest)
    assert book.portfolio_value == pytest.approx(1_000.0 + expected_interest)
    assert book.peak_value == initial_peak
    assert book.max_drawdown == initial_drawdown


def test_zero_cash_basis_is_valid_and_has_no_cash_effect() -> None:
    book = _book(0.5)
    before = _snapshot(book)

    interest = book.apply_cash_interest(
        0.365,
        year_fraction=1.0 / (365.0 * 24.0),
        cash_basis=0.0,
    )

    assert interest == 0.0
    assert _snapshot(book) == before


def test_negative_explicit_basis_and_default_leveraged_cash_keep_signed_interest() -> (
    None
):
    explicit_basis_book = _book(1.5)
    default_basis_book = _book(1.5)
    assert explicit_basis_book.cash == pytest.approx(-500.0)
    year_fraction = 1.0 / (365.0 * 24.0)
    expected_explicit = -600.0 * 0.365 * year_fraction
    expected_default = -500.0 * 0.365 * year_fraction

    explicit_interest = explicit_basis_book.apply_cash_interest(
        0.365,
        year_fraction=year_fraction,
        cash_basis=-600.0,
    )
    default_interest = default_basis_book.apply_cash_interest(
        0.365,
        year_fraction=year_fraction,
    )

    assert explicit_interest == pytest.approx(expected_explicit)
    assert explicit_basis_book.cash == pytest.approx(-500.0 + expected_explicit)
    assert default_interest == pytest.approx(expected_default)
    assert default_basis_book.cash == pytest.approx(-500.0 + expected_default)


@pytest.mark.parametrize(
    "cash_basis",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_nonfinite_explicit_cash_basis_rejects_without_mutating_book(
    cash_basis: float,
) -> None:
    book = _book(0.5)
    before = _snapshot(book)

    with pytest.raises(ValueError):
        book.apply_cash_interest(
            0.365,
            year_fraction=1.0 / (365.0 * 24.0),
            cash_basis=cash_basis,
        )

    assert _snapshot(book) == before
