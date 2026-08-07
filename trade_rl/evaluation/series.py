"""Return-series identity with explicit temporal semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class ReturnKind(StrEnum):
    """Temporal unit represented by each return observation."""

    BASE_BAR = "base_bar"
    DECISION_STEP = "decision_step"


@dataclass(frozen=True, slots=True)
class ReturnSeries:
    """Immutable finite portfolio-return series with annualization metadata."""

    values: tuple[float, ...]
    kind: ReturnKind
    periods_per_year: int
    elapsed_years: float | None = None

    def __post_init__(self) -> None:
        normalized = tuple(float(value) for value in self.values)
        if not normalized:
            raise ValueError("return series must not be empty")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
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
