"""Result-blind UTC ordering and elapsed-time oracles for recorded depth."""

from datetime import UTC, datetime, timedelta, timezone, tzinfo
from fractions import Fraction
from typing import cast

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.depth import DepthOrderRules, execute_depth_order


class _RepeatedHour(tzinfo):
    """Model a repeated hour without depending on a host's timezone database."""

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        if dt is None:
            return None
        return timedelta(hours=-4 - dt.fold)


_LOCAL = _RepeatedHour()


def _local(second: int, *, fold: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, 1, 1, minute, second, tzinfo=_LOCAL, fold=fold)


def _utc(hour: int, second: int, *, minute: int = 0) -> datetime:
    return datetime(2026, 11, 1, hour, minute, second, tzinfo=UTC)


class _UnreachableBook:
    def __getattribute__(self, name: str) -> object:
        raise AssertionError(f"clock gate accessed book field {name}")


def _execute(book: BookState, clock: tuple[datetime, datetime, datetime], lots: int):
    return execute_depth_order(
        book=book,
        symbol_index=0,
        lot_count=lots,
        rules=DepthOrderRules(1, 1, 100, 0, None),
        bids=((99.0, 10.0), (98.0, 10.0)),
        asks=((101.0, 10.0), (102.0, 10.0)),
        decision_at=clock[0],
        quote_at=clock[1],
        execution_at=clock[2],
        fee_bps=10.0,
        adverse_bps=0.0,
        depth_fraction=0.1,
    )


_FOLD_CASES = [
    pytest.param(
        (_local(0, fold=1), _local(1, fold=0), _local(2, fold=0)),
        (_utc(6, 0), _utc(5, 1), _utc(5, 2)),
        "decision must precede",
        id="quote-before-decision",
    ),
    pytest.param(
        (_local(0, fold=0), _local(1, fold=1), _local(2, fold=0)),
        (_utc(5, 0), _utc(6, 1), _utc(5, 2)),
        "decision must precede",
        id="execution-before-quote",
    ),
    pytest.param(
        (_local(0, fold=0), _local(1, fold=0), _local(2, fold=1)),
        (_utc(5, 0), _utc(5, 1), _utc(6, 2)),
        "is stale",
        id="stale-quote-hidden-by-fold",
    ),
    pytest.param(
        (_local(0, fold=0), _local(1, fold=1), _local(2, fold=1)),
        (_utc(5, 0), _utc(6, 1), _utc(6, 2)),
        "is stale",
        id="stale-decision-with-fresh-quote",
    ),
    pytest.param(
        (_local(58, fold=0, minute=59), _local(0, fold=1), _local(1, fold=1)),
        (_utc(5, 58, minute=59), _utc(6, 0), _utc(6, 1)),
        "nonzero integer lot count",
        id="valid-repeated-hour-crossing",
    ),
]


@pytest.mark.parametrize(("clock", "expected_utc", "error"), _FOLD_CASES)
def test_fold_clock_gate(clock, expected_utc, error):
    # Literal UTC instants are the independent timeline oracle. A valid clock
    # reaches the intentionally invalid lot count, never the book or quantities.
    assert tuple(value.astimezone(UTC) for value in clock) == expected_utc
    with pytest.raises(ValueError, match=error):
        _execute(cast(BookState, _UnreachableBook()), clock, 0)


_START = _utc(5, 0)


@pytest.mark.parametrize(
    ("quote_delta", "execution_delta", "error"),
    [
        (0, 0, "decision must precede"),
        (1, 0.999999, "decision must precede"),
        (5, 10, "nonzero integer lot count"),
        (5, 10.000001, "is stale"),
        (9, 10.000001, "is stale"),
        (1, 6, "nonzero integer lot count"),
        (1, 6.000001, "is stale"),
        (5, 5, "nonzero integer lot count"),
    ],
)
def test_mixed_offset_clock_gate(quote_delta, execution_delta, error):
    clock = (
        _START,
        (_START + timedelta(seconds=quote_delta)).astimezone(
            timezone(timedelta(hours=9))
        ),
        (_START + timedelta(seconds=execution_delta)).astimezone(
            timezone(timedelta(hours=-3, minutes=-30))
        ),
    )
    with pytest.raises(ValueError, match=error):
        _execute(cast(BookState, _UnreachableBook()), clock, 0)


def test_out_of_range_utc_clock_gate():
    first = datetime.min.replace(tzinfo=timezone(timedelta(hours=1)))
    clock = (first, first + timedelta(seconds=1), first + timedelta(seconds=2))
    with pytest.raises(ValueError, match="representable in UTC"):
        _execute(cast(BookState, _UnreachableBook()), clock, 0)


def test_reversed_clock_keeps_actual_book_unchanged():
    book = BookState.zero(1, 1000, np.array([100.0]))
    clock = (_local(0, fold=1), _local(1, fold=0), _local(2, fold=0))
    with pytest.raises(ValueError, match="decision must precede"):
        _execute(book, clock, 2)
    assert book.cash == 1000
    assert book.exact_quantities == (Fraction(0),)
    assert book.mark_prices.tolist() == [100.0]
    assert book.fill_count == 0
    assert book.total_cost == 0
    assert book.turnover_total == 0


@pytest.mark.parametrize("lots", [2, -2])
def test_valid_fold_fill_matches_utc_and_independent_cash_ledger(lots):
    local = (_local(58, fold=0, minute=59), _local(0, fold=1), _local(1, fold=1))
    utc = (_utc(5, 58, minute=59), _utc(6, 0), _utc(6, 1))
    book = BookState.zero(1, 1000, np.array([100.0]))
    control = BookState.zero(1, 1000, np.array([100.0]))
    result = _execute(book, local, lots)
    assert result == _execute(control, utc, lots)
    price = Fraction(203 if lots > 0 else 197, 2)
    fee = abs(lots) * price / 1000
    expected_cash = Fraction(1000) - lots * price - fee
    assert result.filled_lots == lots and result.unfilled_lots == 0
    assert result.price == float(price)
    assert result.fee == pytest.approx(float(fee))
    assert book.cash == pytest.approx(float(expected_cash))
    assert book.exact_quantities == (Fraction(lots),)
    assert book.portfolio_value == pytest.approx(float(expected_cash + 100 * lots))
    assert book.total_cost == pytest.approx(float(fee))
    assert book.mark_prices.tolist() == [100.0]


class _ForeignTimestamp(datetime):
    conversion_calls = 0

    def astimezone(self, tz=None):
        type(self).conversion_calls += 1
        return _utc(5, 0)


class _ChangingOffset(tzinfo):
    def __init__(self, first: timedelta, later: timedelta) -> None:
        self.first = first
        self.later = later
        self.calls = 0

    def utcoffset(self, dt: datetime | None) -> timedelta:
        self.calls += 1
        return self.first if self.calls == 1 else self.later


def test_foreign_datetime_cannot_replace_decision_instant():
    _ForeignTimestamp.conversion_calls = 0
    book = BookState.zero(1, 1000, np.array([100.0]))
    decision = _ForeignTimestamp(2026, 11, 1, 6, tzinfo=UTC)
    clock = (decision, _utc(5, 1), _utc(5, 2))
    with pytest.raises(ValueError, match="native"):
        _execute(book, clock, 2)
    assert _ForeignTimestamp.conversion_calls == 0
    assert book.cash == 1000
    assert book.exact_quantities == (Fraction(0),)
    assert book.fill_count == 0
    assert book.total_cost == 0


@pytest.mark.parametrize("index", [0, 1, 2])
def test_foreign_datetime_refused_in_each_clock_position(index):
    class ForeignDatetime(datetime):
        def utcoffset(self):
            raise AssertionError("foreign datetime offset hook invoked")

    clock = [_utc(5, 0), _utc(5, 1), _utc(5, 2)]
    clock[index] = ForeignDatetime(2026, 11, 1, 5, 0, index, tzinfo=UTC)
    with pytest.raises(ValueError, match="native"):
        _execute(cast(BookState, _UnreachableBook()), tuple(clock), 0)


def test_changing_offset_cannot_turn_future_decision_into_past():
    zone = _ChangingOffset(timedelta(hours=-5), timedelta(hours=-4))
    decision = datetime(2026, 11, 1, 1, tzinfo=zone)
    book = BookState.zero(1, 1000, np.array([100.0]))
    # First sample declares 06:00 UTC. A second sample would turn it into 05:00.
    with pytest.raises(ValueError, match="decision must precede"):
        _execute(book, (decision, _utc(5, 1), _utc(5, 2)), 2)
    assert zone.calls == 1
    assert book.cash == 1000
    assert book.exact_quantities == (Fraction(0),)
    assert book.fill_count == 0
    assert book.total_cost == 0


@pytest.mark.parametrize("index", [0, 1, 2])
def test_each_utc_offset_is_sampled_once_for_valid_fill(index):
    zone = _ChangingOffset(timedelta(hours=-4), timedelta(hours=-5))
    clock = [_utc(5, 0), _utc(5, 1), _utc(5, 2)]
    clock[index] = datetime(2026, 11, 1, 1, 0, index, tzinfo=zone)
    book = BookState.zero(1, 1000, np.array([100.0]))
    control = BookState.zero(1, 1000, np.array([100.0]))
    assert _execute(book, tuple(clock), 2) == _execute(
        control, (_utc(5, 0), _utc(5, 1), _utc(5, 2)), 2
    )
    assert zone.calls == 1
    assert book.cash == control.cash
    assert book.exact_quantities == control.exact_quantities
    assert book.fill_count == 1


def test_foreign_offset_refused_without_arithmetic():
    class ForeignOffset(timedelta):
        def __rsub__(self, other):
            raise AssertionError("foreign offset arithmetic invoked")

    class ForeignOffsetZone(tzinfo):
        def utcoffset(self, dt):
            return ForeignOffset(hours=-4)

    book = BookState.zero(1, 1000, np.array([100.0]))
    decision = datetime(2026, 11, 1, 1, tzinfo=ForeignOffsetZone())
    with pytest.raises(ValueError, match="native timedelta"):
        _execute(book, (decision, _utc(5, 1), _utc(5, 2)), 2)
    assert book.cash == 1000
    assert book.exact_quantities == (Fraction(0),)
    assert book.fill_count == 0


def test_naive_clock_still_refused_before_book_mutation():
    with pytest.raises(ValueError, match="timezone-aware"):
        _execute(
            cast(BookState, _UnreachableBook()),
            (_utc(5, 0).replace(tzinfo=None), _utc(5, 1), _utc(5, 2)),
            2,
        )
