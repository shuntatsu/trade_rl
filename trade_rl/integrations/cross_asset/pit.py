"""Pure point-in-time rules for cross-asset daily observations.

An observation made at a venue's official session close only becomes *knowable*
after a publication lag. Everything here is a pure function of dates and rules:
no clock, no network, no filesystem.

Conventions
-----------
* ``session_close_utc`` converts the venue's local close on the observation date
  to UTC (daylight saving handled by the IANA database).
* ``knowable_at_utc`` is the later of that close and the *end* of the
  publication business day in the publisher's time zone. No intraday release
  time is assumed, so a value is never treated as known during the day it is
  published.
* Publication business days skip weekends and a deliberately conservative union
  of US federal and exchange holidays. A day that is a holiday in either calendar
  is never counted as a publication day, which can only delay knowability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np

PUBLISHER_TZ = "America/New_York"
_NS_PER_DAY = 86_400_000_000_000


@dataclass(frozen=True, slots=True)
class ReleaseRule:
    """How long after an observation date its value is published.

    ``lag_business_days`` counts publisher business days. With
    ``weekly_release_weekday`` set (Monday is 0) the value first appears in the
    first release on that weekday strictly after the observation date, rolled to
    the next business day, and ``lag_business_days`` is then added as a buffer.
    """

    lag_business_days: int
    weekly_release_weekday: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.lag_business_days, bool) or not isinstance(
            self.lag_business_days, int
        ):
            raise ValueError("lag_business_days must be an integer")
        if self.lag_business_days < 0:
            raise ValueError("lag_business_days must be non-negative")
        weekday = self.weekly_release_weekday
        if weekday is not None and (
            isinstance(weekday, bool)
            or not isinstance(weekday, int)
            or not 0 <= weekday <= 6
        ):
            raise ValueError("weekly_release_weekday must be 0 (Mon) through 6 (Sun)")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "lag_business_days": self.lag_business_days,
            "weekly_release_weekday": self.weekly_release_weekday,
        }


def _easter_sunday(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    weekday_offset = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * weekday_offset) // 451
    month, day = divmod(h + weekday_offset - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    following = date(year + (month == 12), month % 12 + 1, 1)
    last = following - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _with_observed(day: date) -> set[date]:
    result = {day}
    if day.weekday() == 5:
        result.add(day - timedelta(days=1))
    elif day.weekday() == 6:
        result.add(day + timedelta(days=1))
    return result


@lru_cache(maxsize=None)
def us_publication_holidays(year: int) -> frozenset[date]:
    """Return days on which no US series is treated as published.

    This is the union of US federal holidays (with weekend observance) and
    exchange-only closures (Good Friday). Columbus Day and Veterans Day are
    included although exchanges trade, which only delays knowability. Ad-hoc
    closures (state funerals, hurricanes) are not modelled.
    """

    days: set[date] = set()
    # A Saturday New Year's Day of the next year is observed on December 31.
    days |= _with_observed(date(year, 1, 1))
    days |= {d for d in _with_observed(date(year + 1, 1, 1)) if d.year == year}
    days.add(_nth_weekday(year, 1, 0, 3))  # Martin Luther King Jr. Day
    days.add(_nth_weekday(year, 2, 0, 3))  # Washington's Birthday
    days.add(_easter_sunday(year) - timedelta(days=2))  # Good Friday
    days.add(_last_weekday(year, 5, 0))  # Memorial Day
    if year >= 2021:
        days |= _with_observed(date(year, 6, 19))  # Juneteenth
    days |= _with_observed(date(year, 7, 4))
    days.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    days.add(_nth_weekday(year, 10, 0, 2))  # Columbus Day
    days |= _with_observed(date(year, 11, 11))  # Veterans Day
    days.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    days |= _with_observed(date(year, 12, 25))
    return frozenset(days)


def is_business_day(day: date) -> bool:
    return day.weekday() < 5 and day not in us_publication_holidays(day.year)


def add_business_days(day: date, count: int) -> date:
    """Return ``day`` advanced by ``count`` publisher business days.

    With ``count == 0`` a non-business ``day`` rolls forward to the next business
    day; the result is never earlier than ``day``.
    """

    if count < 0:
        raise ValueError("business day count must be non-negative")
    current = day
    while not is_business_day(current):
        current += timedelta(days=1)
    remaining = count
    while remaining > 0:
        current += timedelta(days=1)
        if is_business_day(current):
            remaining -= 1
    return current


def publication_date(obs_date: date, rule: ReleaseRule) -> date:
    """Return the publisher business day on which ``obs_date`` is released."""

    release = obs_date
    if rule.weekly_release_weekday is not None:
        release = obs_date + timedelta(days=1)
        while release.weekday() != rule.weekly_release_weekday:
            release += timedelta(days=1)
        release = add_business_days(release, 0)
    return add_business_days(release, rule.lag_business_days)


def session_close_utc(obs_date: date, venue_tz: str, close_local: time) -> datetime:
    """Convert the venue's local session close on ``obs_date`` to UTC."""

    if close_local.tzinfo is not None:
        raise ValueError("close_local must be a naive local wall-clock time")
    local = datetime.combine(obs_date, close_local, tzinfo=ZoneInfo(venue_tz))
    return local.astimezone(UTC)


def knowable_at_utc(
    obs_date: date,
    *,
    venue_tz: str,
    close_local: time,
    rule: ReleaseRule,
    publisher_tz: str = PUBLISHER_TZ,
) -> datetime:
    """Return the first UTC instant at which the observation may be used."""

    release_day = publication_date(obs_date, rule)
    release_end = datetime.combine(
        release_day + timedelta(days=1), time(0, 0), tzinfo=ZoneInfo(publisher_tz)
    ).astimezone(UTC)
    return max(session_close_utc(obs_date, venue_tz, close_local), release_end)


def _datetime64_ns(value: datetime) -> np.datetime64:
    return np.datetime64(value.astimezone(UTC).replace(tzinfo=None), "ns")


def session_close_array(
    obs_dates: np.ndarray, *, venue_tz: str, close_local: time
) -> np.ndarray:
    """Vectorised ``session_close_utc`` returning ``datetime64[ns]`` (UTC)."""

    days = np.asarray(obs_dates, dtype="datetime64[D]")
    return np.asarray(
        [
            _datetime64_ns(session_close_utc(day.astype(date), venue_tz, close_local))
            for day in days
        ],
        dtype="datetime64[ns]",
    )


def knowable_at_array(
    obs_dates: np.ndarray,
    *,
    venue_tz: str,
    close_local: time,
    rule: ReleaseRule,
    publisher_tz: str = PUBLISHER_TZ,
) -> np.ndarray:
    """Vectorised ``knowable_at_utc`` returning ``datetime64[ns]`` (UTC)."""

    days = np.asarray(obs_dates, dtype="datetime64[D]")
    return np.asarray(
        [
            _datetime64_ns(
                knowable_at_utc(
                    day.astype(date),
                    venue_tz=venue_tz,
                    close_local=close_local,
                    rule=rule,
                    publisher_tz=publisher_tz,
                )
            )
            for day in days
        ],
        dtype="datetime64[ns]",
    )


def align_asof(knowable_ns: np.ndarray, grid_ns: np.ndarray) -> np.ndarray:
    """Return, per decision time, the index of the latest row knowable by then.

    ``knowable_ns`` is the non-decreasing UTC nanosecond release time of each row
    and ``grid_ns`` the decision times. A row is usable at ``t`` iff
    ``knowable_at <= t``. ``-1`` marks decision times before the first release.
    Rows are only ever carried forward in time, never backward.
    """

    knowable = np.asarray(knowable_ns, dtype=np.int64)
    grid = np.asarray(grid_ns, dtype=np.int64)
    if knowable.ndim != 1 or grid.ndim != 1:
        raise ValueError("release times and decision times must be one-dimensional")
    if knowable.size > 1 and np.any(np.diff(knowable) < 0):
        raise ValueError("release times must be non-decreasing")
    return np.searchsorted(knowable, grid, side="right").astype(np.int64) - 1


__all__ = [
    "PUBLISHER_TZ",
    "ReleaseRule",
    "add_business_days",
    "align_asof",
    "is_business_day",
    "knowable_at_array",
    "knowable_at_utc",
    "publication_date",
    "session_close_array",
    "session_close_utc",
    "us_publication_holidays",
]
