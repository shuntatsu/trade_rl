"""Explicit native-bar boundary contract, separate from legacy Dataset clocks.

A grid of close timestamps is a structural assertion, not evidence that a
provider published a complete candle. Publication, OHLC completeness, and
execution/decision-clock admission belong to their respective consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

import numpy as np

from trade_rl.data._timestamps import (
    _MAX_DATETIME64_NS,
    _MIN_FINITE_DATETIME64_NS,
    _NANOSECONDS_PER_DAY,
    _datetime64_values_to_ns_exact,
)

_NS_PER_SECOND = 1_000_000_000
_WEEK_NS = 7 * _NANOSECONDS_PER_DAY
# Unix epoch was Thursday; the first following Monday is 1970-01-05 UTC.
_MONDAY_ANCHOR_NS = 4 * _NANOSECONDS_PER_DAY


class NativeClockKind(StrEnum):
    """Distinguish elapsed bars from explicitly UTC-aligned calendar bars."""

    FIXED_ELAPSED = "fixed_elapsed"
    UTC_WEEK = "utc_week"
    UTC_MONTH = "utc_month"


@dataclass(frozen=True, slots=True)
class NativeBarClock:
    """Validate consecutive half-open native intervals [start, close).

    Fixed intervals require an explicit grid anchor; a fixed seven-day
    interval is not necessarily a UTC week. UTC weeks close Monday 00:00;
    calendar months close on the first day of the next month at 00:00.
    NumPy datetime64 timestamps are interpreted as UTC, not local civil time.

    This additive contract does not admit a new MarketBuildConfig timeframe,
    change a Dataset/ReturnSeries schema, or grant a replay capability.
    """

    kind: NativeClockKind
    fixed_seconds: int | None = None
    fixed_anchor: np.datetime64 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, NativeClockKind):
            raise ValueError("unsupported native clock kind")
        if self.kind is NativeClockKind.FIXED_ELAPSED:
            seconds = self.fixed_seconds
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, int)
                or not 0 < seconds <= _MAX_DATETIME64_NS // _NS_PER_SECOND
            ):
                raise ValueError("fixed_seconds must be positive representable seconds")
            if not isinstance(self.fixed_anchor, np.datetime64):
                raise ValueError("fixed clock requires an explicit datetime64 anchor")
            anchor = _datetime64_values_to_ns_exact(np.asarray(self.fixed_anchor))[()]
            object.__setattr__(self, "fixed_anchor", anchor)
        elif self.fixed_seconds is not None or self.fixed_anchor is not None:
            raise ValueError("calendar clock does not accept fixed duration or anchor")

    def interval_bounds(self, closes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return lossless ns starts and closes for a nonempty complete grid.

        Reject missing, duplicate, reordered, off-grid or unrepresentable
        boundaries rather than infer or fill any missing bar.
        """
        normalized = _datetime64_values_to_ns_exact(np.asarray(closes))
        if normalized.ndim != 1 or normalized.size == 0:
            raise ValueError("native close grid must be nonempty and one-dimensional")
        if self.kind is NativeClockKind.UTC_MONTH:
            months = normalized.astype("datetime64[M]")
            month_closes = _datetime64_values_to_ns_exact(months)
            if not np.array_equal(normalized, month_closes):
                raise ValueError(
                    "calendar month closes must be at first-day UTC midnight"
                )
            if np.any(np.diff(months.astype(np.int64)) != 1):
                raise ValueError("native closes must be consecutive calendar months")
            starts = _datetime64_values_to_ns_exact(months - np.timedelta64(1, "M"))
            return starts, normalized
        if self.kind is NativeClockKind.UTC_WEEK:
            step_ns, anchor_ns = _WEEK_NS, _MONDAY_ANCHOR_NS
        else:
            assert self.fixed_seconds is not None and self.fixed_anchor is not None
            step_ns = self.fixed_seconds * _NS_PER_SECOND
            anchor_ns = int(self.fixed_anchor.astype(np.int64))

        ticks = normalized.astype(np.int64)
        previous = int(ticks[0]) - step_ns
        # Use Python integers: subtracting two valid int64 timestamps can wrap.
        for tick in ticks:
            end = int(tick)
            if (end - anchor_ns) % step_ns:
                raise ValueError("native close is off the declared grid anchor")
            if end - previous != step_ns:
                raise ValueError("native closes must be consecutive")
            previous = end
        if int(ticks[0]) - step_ns < _MIN_FINITE_DATETIME64_NS:
            raise ValueError("native start is outside the finite nanosecond range")
        starts = (ticks - np.int64(step_ns)).view("datetime64[ns]")
        return starts, normalized

    def elapsed_years(self, closes: np.ndarray) -> Fraction:
        """Exact ACT/365F sample duration, including the first complete bar.

        This is a duration convention, not a nominal calendar frequency or a
        statistical model for risk on irregularly spaced return observations.
        """
        starts, normalized = self.interval_bounds(closes)
        elapsed_ns = int(normalized[-1].astype(np.int64)) - int(
            starts[0].astype(np.int64)
        )
        return Fraction(elapsed_ns, 365 * _NANOSECONDS_PER_DAY)
