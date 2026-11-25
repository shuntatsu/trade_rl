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


def _require_unmasked_array_owners(
    value: np.ndarray | memoryview, *, field_name: str
) -> None:
    """Verify only closed native owner chains, before reading numeric values."""
    owner: object = value
    while True:
        if type(owner) is np.ndarray:
            owner = np.ndarray.base.__get__(owner)
        elif type(owner) is memoryview:
            owner = owner.obj
        elif owner is None or type(owner) is bytes or type(owner) is bytearray:
            return
        elif issubclass(type(owner), np.ma.MaskedArray):
            raise ValueError(f"{field_name} must not contain a masked array")
        else:
            # Never invoke foreign base or __class__ getters. A subclass or
            # opaque array-interface wrapper may retain missingness we cannot
            # authenticate through native owner links.
            raise ValueError(f"{field_name} has opaque or subclass array ownership")


def _native_unmasked_array(value: np.ndarray, *, field_name: str) -> np.ndarray:
    """Require native arrays before a conversion can erase missingness.

    Already discarded mask provenance cannot be recovered at this boundary.
    """
    if not issubclass(type(value), np.ndarray):
        raise ValueError(f"{field_name} must be a NumPy ndarray")
    _require_unmasked_array_owners(value, field_name=field_name)
    return np.asarray(value)


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
        normalized = _datetime64_values_to_ns_exact(
            _native_unmasked_array(closes, field_name="native closes")
        )
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

        ticks = normalized.view(np.int64)
        first = int(ticks[0])
        # Anchor the first bar in unbounded integer arithmetic. An exact
        # consecutive grid then inherits its phase without per-row modulo.
        if (first - anchor_ns) % step_ns:
            raise ValueError("native close is off the declared grid anchor")
        # For strictly increasing signed timestamps the unsigned modular
        # difference equals the mathematical positive distance, even across
        # the epoch. Check order independently: reversed timestamps can alias
        # a valid unsigned delta across the signed nanosecond boundary.
        invalid = np.diff(ticks.view(np.uint64)) != np.uint64(step_ns)
        invalid |= ticks[1:] <= ticks[:-1]
        if np.any(invalid):
            # The first invalid pair follows a valid anchored prefix; inspect
            # that close alone to preserve the original error precedence.
            end = int(ticks[int(np.argmax(invalid)) + 1])
            if (end - anchor_ns) % step_ns:
                raise ValueError("native close is off the declared grid anchor")
            raise ValueError("native closes must be consecutive")
        if first - step_ns < _MIN_FINITE_DATETIME64_NS:
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
