"""Independent split oracles for representable marks and replacement atomicity."""

from fractions import Fraction

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState


@pytest.mark.parametrize("n_symbols", [1, 9])
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("mode", ["ignore", "raise"])
def test_split_preserves_representable_subnormal_mark(
    n_symbols: int, direction: int, mode: str
) -> None:
    mark = float.fromhex("0x1.0p-1022")
    book = BookState(
        quantities=np.full(n_symbols, float(direction)),
        cash=1000.0,
        mark_prices=np.full(n_symbols, mark),
        peak_value=1000.0,
    )
    expected_mark = float(Fraction.from_float(mark) / 3)
    assert 0.0 < expected_mark < mark
    with np.errstate(all=mode):
        book.apply_split(np.full(n_symbols, 3.0))
        values = book.position_values
    assert book.cash == 1000.0
    assert book.exact_quantities == (Fraction(3 * direction),) * n_symbols
    np.testing.assert_array_equal(book.quantities, [3.0 * direction] * n_symbols)
    np.testing.assert_array_equal(book.mark_prices, [expected_mark] * n_symbols)
    expected_value = float(3 * direction * Fraction.from_float(expected_mark))
    np.testing.assert_array_equal(values, [expected_value] * n_symbols)


@pytest.mark.parametrize("mode", ["ignore", "raise"])
def test_split_rejects_mark_rounded_to_zero_without_mutating_book(mode: str) -> None:
    mark = float.fromhex("0x0.0000000000001p-1022")
    book = BookState(
        quantities=np.array([1.0, -2.0]),
        cash=1000.0,
        mark_prices=np.array([10.0, mark]),
        peak_value=1010.0,
    )
    before = book.clone()
    with np.errstate(all=mode), pytest.raises(ValueError, match="split produced"):
        book.apply_split(np.array([2.0, 4.0]))
    assert book.exact_quantities == before.exact_quantities
    assert book.cash == before.cash
    np.testing.assert_array_equal(book.quantities, before.quantities)
    np.testing.assert_array_equal(book.mark_prices, before.mark_prices)
    assert book.portfolio_value == before.portfolio_value


def test_split_can_replace_read_only_mark_snapshot() -> None:
    book = BookState(
        quantities=np.array([2.0, -4.0]),
        cash=100.0,
        mark_prices=np.array([10.0, 5.0]),
        peak_value=100.0,
    )
    old_marks = book.mark_prices
    old_marks.flags.writeable = False

    book.apply_split(np.array([2.0, 4.0]))

    assert book.exact_quantities == (Fraction(4), Fraction(-16))
    np.testing.assert_array_equal(book.mark_prices, [5.0, 1.25])
    np.testing.assert_array_equal(old_marks, [10.0, 5.0])
    assert book.cash == book.portfolio_value == 100.0


def test_split_does_not_overwrite_aliased_contract_multipliers() -> None:
    book = BookState(
        quantities=np.array([2.0]),
        cash=100.0,
        mark_prices=np.array([10.0]),
        contract_multipliers=np.array([10.0]),
        peak_value=300.0,
    )
    # The public arrays may share storage. A split adjusts the mark, not the
    # contract multiplier or the independent inventory valuation.
    book.mark_prices = book.contract_multipliers
    book.apply_split(np.array([2.0]))

    assert book.exact_quantities == (Fraction(4),)
    np.testing.assert_array_equal(book.mark_prices, [5.0])
    np.testing.assert_array_equal(book.contract_multipliers, [10.0])
    assert book.cash == 100.0
    assert book.portfolio_value == 300.0
