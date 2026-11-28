"""Return-series identity with explicit temporal semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from numbers import Integral, Real

import numpy as np

from trade_rl._validation import _ordered_ns_durations
from trade_rl.data._timestamps import (
    _MAX_DATETIME64_NS,
    _MIN_FINITE_DATETIME64_NS,
    _NANOSECONDS_PER_DAY,
)
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.native_execution_clock import calendar_month_execution_frequency


class ReturnKind(StrEnum):
    """Temporal unit represented by each return observation."""

    BASE_BAR = "base_bar"
    DECISION_STEP = "decision_step"


@dataclass(frozen=True, slots=True)
class ReturnSeries:
    """Immutable returns with legacy integer or explicit nonintegral fixed frequency.

    Native fixed return payloads bind the Fraction to the Dataset interval.
    Existing elapsed-year and persisted legacy result contracts are unchanged.
    """

    values: tuple[float, ...]
    kind: ReturnKind
    periods_per_year: int | Fraction
    elapsed_years: float | None = None

    def __post_init__(self) -> None:
        periods_per_year: object = self.periods_per_year
        normalized = (
            _normalize_duration_returns(self.values, label="native fixed")
            if type(periods_per_year) is Fraction
            else tuple(float(value) for value in self.values)
        )
        if not normalized:
            raise ValueError("return series must not be empty")
        if type(periods_per_year) is Fraction:
            try:
                frequency = float(periods_per_year)
            except OverflowError as exc:
                raise ValueError("exact annual frequency is not representable") from exc
            if (
                not math.isfinite(frequency)
                or frequency <= 0
                or periods_per_year.denominator == 1
                or self.elapsed_years is not None
            ):
                raise ValueError(
                    "exact fixed annual frequency cannot be combined with elapsed_years"
                )
        else:
            if (
                isinstance(periods_per_year, bool)
                or not isinstance(periods_per_year, Integral)
                or periods_per_year <= 0
            ):
                raise ValueError("periods_per_year must be a positive integer")
            object.__setattr__(self, "periods_per_year", int(periods_per_year))
        if self.elapsed_years is not None and (
            not math.isfinite(self.elapsed_years) or self.elapsed_years <= 0.0
        ):
            raise ValueError("elapsed_years must be finite and positive")
        for value in normalized:
            if not math.isfinite(value):
                raise ValueError("return series values must be finite")
            if value <= -1.0:
                raise ValueError("return series values must be greater than -1")
        object.__setattr__(self, "values", normalized)

    @property
    def annualization_periods_per_year(self) -> float:
        if self.elapsed_years is None:
            return float(self.periods_per_year)
        return len(self.values) / self.elapsed_years


@dataclass(frozen=True, slots=True)
class CalendarMonthReturnSeries(ReturnSeries):
    """Complete UTC calendar-month returns; frequency is descriptive, not fixed."""

    closes_ns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.kind is not ReturnKind.BASE_BAR
            or self.elapsed_years is not None
            or type(self.periods_per_year) is not Fraction
            or type(self.values) is not tuple
            or type(self.closes_ns) is not tuple
            or not self.closes_ns
            or len(self.closes_ns) != len(self.values)
            or any(
                type(x) is not int
                or not _MIN_FINITE_DATETIME64_NS <= x <= _MAX_DATETIME64_NS
                for x in self.closes_ns
            )
        ):
            raise ValueError("invalid calendar month return series")
        expected = calendar_month_execution_frequency(
            NativeBarClock(NativeClockKind.UTC_MONTH), self.closes
        )
        if self.periods_per_year != expected:
            raise ValueError(
                "calendar month return frequency does not match its intervals"
            )
        object.__setattr__(
            self,
            "values",
            _normalize_duration_returns(self.values, label="calendar month"),
        )

    @property
    def closes(self) -> np.ndarray:
        return np.asarray(self.closes_ns, dtype=np.int64).view("datetime64[ns]")

    @property
    def exact_elapsed_years(self) -> Fraction:
        return NativeBarClock(NativeClockKind.UTC_MONTH).elapsed_years(self.closes)

    @property
    def annualization_periods_per_year(self) -> float:
        raise ValueError("calendar month returns require duration-aware annualization")

    @classmethod
    def from_closes(
        cls, values: tuple[float, ...], closes: np.ndarray
    ) -> CalendarMonthReturnSeries:
        clock = NativeBarClock(NativeClockKind.UTC_MONTH)
        _, ends = clock.interval_bounds(closes)
        return cls(
            values,
            ReturnKind.BASE_BAR,
            calendar_month_execution_frequency(clock, ends),
            closes_ns=tuple(int(x) for x in ends.view(np.int64)),
        )


def _normalize_duration_returns(
    values: tuple[float, ...], *, label: str
) -> tuple[float, ...]:
    """Strict native-return boundary; never erase nonzero values on conversion."""
    normalized: list[float] = []
    for raw_item in values:
        item: object = raw_item
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise ValueError(f"{label} returns must be real numbers")
        try:
            value = float(item)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{label} return is not representable") from exc
        if not math.isfinite(value) or value <= -1.0:
            raise ValueError(f"{label} returns must be finite and greater than -1")
        if value == 0.0 and item != 0:
            raise ValueError(f"{label} nonzero return is not representable")
        normalized.append(value)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class SessionElapsedReturnSeries(ReturnSeries):
    """One close-to-close return per declared session interval, including gaps.

    The boundaries include the initial mark. Their identity does not attest
    a venue calendar or distinguish an independently omitted session.
    """

    boundaries_ns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        durations = _ordered_ns_durations(self.boundaries_ns)
        if (
            self.kind is not ReturnKind.BASE_BAR
            or self.elapsed_years is not None
            or type(self.periods_per_year) is not Fraction
            or type(self.values) is not tuple
            or len(self.values) != len(durations)
            or self.periods_per_year != len(durations) / self.exact_elapsed_years
        ):
            raise ValueError("invalid session elapsed return series")
        object.__setattr__(
            self, "values", _normalize_duration_returns(self.values, label="session")
        )

    @property
    def exact_elapsed_years(self) -> Fraction:
        return Fraction(
            self.boundaries_ns[-1] - self.boundaries_ns[0],
            365 * _NANOSECONDS_PER_DAY,
        )

    @property
    def annualization_periods_per_year(self) -> float:
        raise ValueError("session elapsed returns require duration-aware annualization")

    @classmethod
    def from_boundaries(
        cls, values: tuple[float, ...], *, boundaries_ns: tuple[int, ...]
    ) -> SessionElapsedReturnSeries:
        durations = _ordered_ns_durations(boundaries_ns)
        elapsed = Fraction(sum(durations), 365 * _NANOSECONDS_PER_DAY)
        return cls(
            values,
            ReturnKind.BASE_BAR,
            len(durations) / elapsed,
            boundaries_ns=boundaries_ns,
        )


def _finite_annualization(returns: ReturnSeries, *, label: str) -> float:
    """Validate the derived binary64 clock before performing evaluation."""
    try:
        frequency = returns.annualization_periods_per_year
    except OverflowError as exc:
        raise ValueError(f"{label} annualization must be finite and positive") from exc
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError(f"{label} annualization must be finite and positive")
    return frequency
