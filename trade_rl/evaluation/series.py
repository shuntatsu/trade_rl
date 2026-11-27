"""Return-series identity with explicit temporal semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from numbers import Integral


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
        normalized = tuple(float(value) for value in self.values)
        if not normalized:
            raise ValueError("return series must not be empty")
        periods_per_year: object = self.periods_per_year
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


def _finite_annualization(returns: ReturnSeries, *, label: str) -> float:
    """Validate the derived binary64 clock before performing evaluation."""
    try:
        frequency = returns.annualization_periods_per_year
    except OverflowError as exc:
        raise ValueError(f"{label} annualization must be finite and positive") from exc
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError(f"{label} annualization must be finite and positive")
    return frequency
