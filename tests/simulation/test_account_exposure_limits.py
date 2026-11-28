"""Independent read-side exposure oracles for finite account values."""

from fractions import Fraction

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState


def _book(quantities: list[float], cash: float, margin: float = 0.0) -> BookState:
    return BookState(
        quantities=np.asarray(quantities, dtype=np.float64),
        cash=cash,
        mark_prices=np.ones(len(quantities)),
        peak_value=max(cash, 10.0),
        margin_used=margin,
    )


@pytest.mark.parametrize("mode", ["ignore", "raise"])
@pytest.mark.parametrize("metric", ["weights", "gross_exposure", "net_exposure"])
@pytest.mark.parametrize("n_symbols", [2, 9])
def test_nonfinite_normalized_holdings_refuse_without_mutating_book(
    mode: str, metric: str, n_symbols: int
) -> None:
    cash = float(Fraction(1, 2**1074))
    book = _book([1.0, -1.0] + [0.0] * (n_symbols - 2), cash)
    assert book.portfolio_value == cash
    before = book.quantities.copy()
    with np.errstate(all=mode), pytest.raises(ValueError, match="weights.*non-finite"):
        getattr(book, metric)
    np.testing.assert_array_equal(book.quantities, before)
    assert book.cash == cash
    assert book.portfolio_value == cash


@pytest.mark.parametrize("mode", ["ignore", "raise"])
def test_gross_reduction_refuses_overflow_of_finite_signed_weights(mode: str) -> None:
    cash = float(Fraction(1, 2**1023))
    book = _book([1.0, -1.0], cash)
    expected = float(Fraction(2**1023))
    np.testing.assert_array_equal(book.weights, [expected, -expected])
    assert book.net_exposure == 0.0
    with (
        np.errstate(all=mode),
        pytest.raises(ValueError, match="gross exposure.*non-finite"),
    ):
        _ = book.gross_exposure
    assert book.portfolio_value == cash


@pytest.mark.parametrize("mode", ["ignore", "raise"])
def test_net_reduction_preserves_finite_signed_cancellation(mode: str) -> None:
    cash = float(Fraction(1, 2**1023))
    book = _book([1.0, 1.0, -1.0, -1.0], cash)
    expected = float(Fraction(2**1023))
    np.testing.assert_array_equal(
        book.weights, [expected, expected, -expected, -expected]
    )
    # The ordinary binary64 reduction can overflow before cancelling. The
    # exact sum of the four represented weights is zero.
    assert sum(Fraction.from_float(float(x)) for x in book.weights) == 0
    with np.errstate(all=mode):
        assert book.net_exposure == 0.0
    assert book.portfolio_value == cash


@pytest.mark.parametrize("mode", ["ignore", "raise"])
def test_margin_ratio_refuses_overflow(mode: str) -> None:
    cash = float(Fraction(1, 2**1074))
    book = _book([1.0, -1.0], cash, margin=1.0)
    with (
        np.errstate(all=mode),
        pytest.raises(ValueError, match="margin utilization.*non-finite"),
    ):
        _ = book.margin_utilization
    assert book.margin_used == 1.0
    assert book.portfolio_value == cash


@pytest.mark.parametrize("mode", ["ignore", "raise"])
def test_representable_subnormal_weight_is_not_ambient_numpy_mode_dependent(
    mode: str,
) -> None:
    unit = float(Fraction(1, 2**1022))
    book = _book([unit, -unit], 3.0)
    expected = float(Fraction(1, 3 * 2**1022))
    assert 0.0 < expected < np.finfo(np.float64).tiny
    with np.errstate(all=mode):
        np.testing.assert_array_equal(book.weights, [expected, -expected])
        assert book.gross_exposure == expected * 2.0
        assert book.net_exposure == 0.0


@pytest.mark.parametrize("mode", ["ignore", "raise"])
def test_ordinary_signed_and_nonpositive_equity_contracts_preserved(mode: str) -> None:
    book = _book([3.0, -1.0], 2.0, margin=1.0)
    with np.errstate(all=mode):
        assert book.portfolio_value == 4.0
        np.testing.assert_array_equal(book.weights, [0.75, -0.25])
        assert book.cash_weight == 0.5
        assert book.gross_exposure == 1.0
        assert book.net_exposure == 0.5
        assert book.margin_utilization == 0.25
        terminal = _book([1.0, -1.0], 0.0, margin=1.0)
        np.testing.assert_array_equal(terminal.weights, [0.0, 0.0])
        assert terminal.cash_weight == 0.0
        assert terminal.gross_exposure == 0.0
        assert terminal.net_exposure == 0.0
        assert terminal.margin_utilization == 1.0


@pytest.mark.parametrize("mode", ["ignore", "raise"])
def test_scalar_ratios_preserve_valid_subnormals(mode: str) -> None:
    unit = np.float64(2.0**-1022)
    book = _book([3.0], unit, margin=unit)
    expected = float(Fraction(1, 3 * 2**1022))
    with np.errstate(all=mode):
        assert book.cash_weight == expected
        assert book.margin_utilization == expected


@pytest.mark.parametrize("mode", ["ignore", "raise"])
def test_net_overflow_recovery_preserves_nonzero_represented_residue(mode: str) -> None:
    cash = float(Fraction(1, 2**1023))
    book = _book([1.0, 1.0, cash, -1.0, -1.0], cash)
    # The current book equity uses its established binary64 valuation order.
    assert book.portfolio_value == cash
    expected = float(sum(Fraction.from_float(float(x)) for x in book.weights))
    assert expected == 1.0
    with np.errstate(all=mode):
        assert book.net_exposure == expected
