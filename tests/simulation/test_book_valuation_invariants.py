"""Independent value-domain and retained-debt drawdown counterexamples."""

from fractions import Fraction

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState, EconomicTerminationReason


@pytest.mark.parametrize("n_symbols", [1, 9])
@pytest.mark.parametrize("field", ["mark_prices", "contract_multipliers"])
@pytest.mark.parametrize("invalid", [0.0, -1.0])
@pytest.mark.parametrize("held", [False, True])
def test_public_book_valuation_refuses_nonpositive_marks_and_multipliers(
    n_symbols: int, field: str, invalid: float, held: bool
) -> None:
    book = BookState.zero(n_symbols, 1000.0)
    if held:
        book.quantities[0] = 1.0
    array = getattr(book, field)
    array[0] = invalid
    original_cash = book.cash
    # Public mutable arrays must not silently invert an asset/liability or
    # admit an invalid valuation domain even when the symbol is flat.
    with pytest.raises(ValueError, match="positive|nonzero position value"):
        _ = book.portfolio_value
    assert book.cash == original_cash
    assert book.quantities[0] == float(held)


@pytest.mark.parametrize("n_symbols", [1, 9])
def test_positive_signed_valuation_remains_literal(n_symbols: int) -> None:
    book = BookState.zero(n_symbols, 1000.0)
    book.quantities[0] = -2.0
    book.mark_prices[0] = 3.0
    assert book.contract_multipliers is not None
    book.contract_multipliers[0] = 4.0
    assert book.portfolio_value == float(Fraction(1000) - 2 * 3 * 4)


def test_unrepresentable_retained_debt_drawdown_does_not_corrupt_book() -> None:
    book = BookState.zero(1, 2.0**-20)
    book.charge_borrow(2.0**1023)
    assert book.termination_reason is EconomicTerminationReason.INSOLVENCY
    assert np.isfinite(book.portfolio_value)
    prior = book.clone()
    with pytest.raises(ValueError, match="drawdown.*non-finite"):
        book.refresh_drawdown(unbounded_loss=True)
    assert book.cash == prior.cash
    assert book.peak_value == prior.peak_value
    assert book.max_drawdown == prior.max_drawdown
    assert book.borrow_cost == prior.borrow_cost
    assert book.termination_reason is prior.termination_reason
    assert book.returns_history == prior.returns_history
    assert book.exact_quantities == prior.exact_quantities
    np.testing.assert_array_equal(book.mark_prices, prior.mark_prices)
    # Failed diagnostic calculation cannot make the whole book unclonable.
    assert book.clone().portfolio_value == prior.portfolio_value


def test_large_representable_retained_debt_drawdown_keeps_signed_loss() -> None:
    capital = Fraction(1, 2**20)
    debt_charge = Fraction(2**900)
    book = BookState.zero(1, float(capital))
    book.charge_borrow(float(debt_charge))
    book.refresh_drawdown(unbounded_loss=True)
    assert book.max_drawdown == float(debt_charge / capital)
    assert book.max_drawdown > 1
    assert book.clone().max_drawdown == book.max_drawdown


def test_bounded_loss_drawdown_still_floors_at_one() -> None:
    book = BookState.zero(1, 2.0**-20)
    book.charge_borrow(2.0**1023)
    book.refresh_drawdown()
    assert book.max_drawdown == 1.0
