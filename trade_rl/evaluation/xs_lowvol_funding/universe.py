"""Monthly point-in-time universe of ``xs_lowvol_funding_v1`` (no economics, no I/O).

On each calendar month start ``M`` a symbol is *eligible* when its perp has a complete set of
1h bars over the 90 days strictly before ``M`` (a missing hour is allowed only inside a
declared halt) and at least 9 settled funding events in the 3 days before ``M``. The
*universe* of ``M`` is the 30 eligible symbols with the largest perp quote volume over the 30
days strictly before ``M`` (ties by symbol name). Membership is fixed for the month, uses no
row of ``M`` or later, and is empty (the book is flat) when fewer than 12 symbols are
eligible.

Everything works on plain arrays on one hourly grid: hour ``i`` opens at ``grid_start + i``
hours. A function that is handed a series that ends before ``M`` simply sees an incomplete
window and the symbol is ineligible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np

from trade_rl.evaluation.xs_lowvol_funding import protocol

HOUR: Final = timedelta(hours=1)
_HOUR_MS: Final = 3_600_000


class UniverseDataError(ValueError):
    """The hourly series cannot answer a universe question (history before the grid)."""


@dataclass(frozen=True, slots=True)
class UniverseSeries:
    """What the universe rules read of one symbol, on the stage's hourly grid.

    ``complete`` is True for an hour with a published row or one missing inside a declared
    halt; ``quote_volume`` is the published hourly quote volume (0 where absent);
    ``funding_ms`` are the sorted epoch-millisecond times of the settled funding events.
    """

    complete: np.ndarray
    quote_volume: np.ndarray
    funding_ms: np.ndarray

    def __post_init__(self) -> None:
        complete = np.asarray(self.complete, dtype=np.bool_)
        volume = np.asarray(self.quote_volume, dtype=np.float64)
        funding = np.asarray(self.funding_ms, dtype=np.int64)
        if complete.ndim != 1 or volume.shape != complete.shape:
            raise ValueError("complete and quote_volume must be aligned 1-d arrays")
        if not np.isfinite(volume).all() or np.any(volume < 0.0):
            raise ValueError("quote volume must be finite and non-negative")
        if funding.ndim != 1 or np.any(np.diff(funding) <= 0):
            raise ValueError("funding times must be strictly increasing")
        object.__setattr__(self, "complete", complete)
        object.__setattr__(self, "quote_volume", volume)
        object.__setattr__(self, "funding_ms", funding)


@dataclass(frozen=True, slots=True)
class MonthUniverse:
    """The universe of one month: eligible symbols, members and whether a book exists."""

    month_start: datetime
    eligible: tuple[str, ...]
    members: tuple[str, ...]
    book: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "month_start": self.month_start.isoformat(),
            "eligible": len(self.eligible),
            "members": list(self.members),
            "book": self.book,
        }


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("instants must be timezone-aware UTC")
    return value.astimezone(UTC)


def month_start_of(moment: datetime) -> datetime:
    stamp = _utc(moment)
    return stamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month(month_start: datetime) -> datetime:
    year, month = (
        (month_start.year + 1, 1)
        if month_start.month == 12
        else (month_start.year, month_start.month + 1)
    )
    return month_start.replace(year=year, month=month)


def months_of_stage(
    first_decision: datetime, last_decision_bar: datetime
) -> tuple[datetime, ...]:
    """Month starts of every calendar month that holds a weekly decision of the stage."""
    out: list[datetime] = []
    month = month_start_of(first_decision)
    last = month_start_of(last_decision_bar)
    while month <= last:
        out.append(month)
        month = next_month(month)
    return tuple(out)


def hour_index(moment: datetime, *, grid_start: datetime) -> int:
    delta = _utc(moment) - _utc(grid_start)
    if delta % HOUR != timedelta(0):
        raise ValueError("instant is not on the hourly grid")
    return int(delta // HOUR)


def eligibility(
    series: UniverseSeries, *, month_start: datetime, grid_start: datetime
) -> tuple[bool, float]:
    """``(eligible, 30-day quote volume)`` of one symbol for the month starting at ``month_start``."""
    stop = hour_index(month_start, grid_start=grid_start)
    window_90 = protocol.ELIGIBILITY_DAYS * 24
    window_30 = protocol.VOLUME_DAYS * 24
    if stop - window_90 < 0:
        raise UniverseDataError(
            f"the 90 days before {month_start.isoformat()} start before the hourly grid"
        )
    if stop > series.complete.size:
        raise UniverseDataError(
            f"the hourly series ends before the month {month_start.isoformat()}"
        )
    complete = bool(series.complete[stop - window_90 : stop].all())
    volume = float(series.quote_volume[stop - window_30 : stop].sum())
    start_ms = int(_utc(month_start).timestamp() * 1_000)
    lookback_ms = protocol.FUNDING_LOOKBACK_DAYS * 24 * _HOUR_MS
    lo = int(np.searchsorted(series.funding_ms, start_ms - lookback_ms, side="left"))
    hi = int(np.searchsorted(series.funding_ms, start_ms, side="left"))
    funded = hi - lo >= protocol.MIN_FUNDING_EVENTS
    return bool(complete and funded), volume


def select_universe(
    series: Mapping[str, UniverseSeries],
    *,
    month_start: datetime,
    grid_start: datetime,
) -> MonthUniverse:
    """The month's universe: the top 30 eligible by volume, empty below 12 eligible."""
    eligible: list[tuple[str, float]] = []
    for symbol in sorted(series):
        ok, volume = eligibility(
            series[symbol], month_start=month_start, grid_start=grid_start
        )
        if ok:
            eligible.append((symbol, volume))
    names = tuple(symbol for symbol, _ in eligible)
    if len(eligible) < protocol.MIN_ELIGIBLE:
        return MonthUniverse(
            month_start=_utc(month_start), eligible=names, members=(), book=False
        )
    ranked = sorted(eligible, key=lambda item: (-item[1], item[0]))
    members = tuple(symbol for symbol, _ in ranked[: protocol.UNIVERSE_SIZE])
    return MonthUniverse(
        month_start=_utc(month_start), eligible=names, members=members, book=True
    )


def stage_universe(
    series: Mapping[str, UniverseSeries],
    *,
    grid_start: datetime,
    first_decision: datetime,
    last_decision_bar: datetime,
) -> dict[datetime, MonthUniverse]:
    """The universe of every month that holds a weekly decision of the stage."""
    return {
        month: select_universe(series, month_start=month, grid_start=grid_start)
        for month in months_of_stage(first_decision, last_decision_bar)
    }


def ever_members(universe: Mapping[datetime, MonthUniverse]) -> tuple[str, ...]:
    """Every symbol that is a member in any month, sorted: the dataset's legs."""
    return tuple(
        sorted({name for month in universe.values() for name in month.members})
    )


def membership_mask(
    universe: Mapping[datetime, MonthUniverse],
    clock: np.ndarray,
    legs: Sequence[str],
) -> np.ndarray:
    """``(rows, legs)`` mask: True where the leg is a member of the month of the row close.

    A row belongs to the calendar month of its close timestamp; a decision row (a close at
    Monday 00:00) therefore uses the universe of the month that Monday falls in, which was
    fixed from rows before that month began.
    """
    stamps = np.asarray(clock).astype("datetime64[ns]")
    months = stamps.astype("datetime64[M]")
    mask = np.zeros((stamps.size, len(legs)), dtype=np.bool_)
    column = {name: index for index, name in enumerate(legs)}
    for start, month in universe.items():
        key = np.datetime64(start.replace(tzinfo=None), "M")
        rows = months == key
        for name in month.members:
            mask[rows, column[name]] = True
    return mask


__all__ = [
    "MonthUniverse",
    "UniverseDataError",
    "UniverseSeries",
    "eligibility",
    "ever_members",
    "hour_index",
    "membership_mask",
    "month_start_of",
    "months_of_stage",
    "next_month",
    "select_universe",
    "stage_universe",
]
