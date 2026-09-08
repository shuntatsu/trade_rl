"""AUM-capacity evaluation contracts for execution-aware strategies."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CapacityPoint:
    initial_capital: float
    total_return: float
    excess_total_return: float
    total_cost_fraction: float
    fill_ratio: float
    unfilled_turnover: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("initial_capital", self.initial_capital),
            ("total_return", self.total_return),
            ("excess_total_return", self.excess_total_return),
            ("total_cost_fraction", self.total_cost_fraction),
            ("fill_ratio", self.fill_ratio),
            ("unfilled_turnover", self.unfilled_turnover),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        if self.total_cost_fraction < 0.0 or self.unfilled_turnover < 0.0:
            raise ValueError(
                "capacity costs and unfilled turnover must be non-negative"
            )
        if not 0.0 <= self.fill_ratio <= 1.0:
            raise ValueError("fill_ratio must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class CapacityCurve:
    points: tuple[CapacityPoint, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("capacity curve must contain points")
        capitals = tuple(point.initial_capital for point in self.points)
        if tuple(sorted(capitals)) != capitals or len(set(capitals)) != len(capitals):
            raise ValueError("capacity points must have unique ascending capital")

    def maximum_viable_capital(
        self,
        *,
        minimum_fill_ratio: float = 0.95,
        minimum_excess_return: float = 0.0,
    ) -> float | None:
        viable = [
            point.initial_capital
            for point in self.points
            if point.fill_ratio >= minimum_fill_ratio
            and point.excess_total_return >= minimum_excess_return
        ]
        return None if not viable else max(viable)


def evaluate_capacity_grid(
    capitals: Iterable[float],
    evaluator: Callable[[float], CapacityPoint],
) -> CapacityCurve:
    resolved = tuple(sorted(float(value) for value in capitals))
    if not resolved or any(
        not math.isfinite(value) or value <= 0.0 for value in resolved
    ):
        raise ValueError("capitals must be finite and positive")
    if len(set(resolved)) != len(resolved):
        raise ValueError("capitals must be unique")
    points = tuple(evaluator(capital) for capital in resolved)
    if any(
        not math.isclose(point.initial_capital, capital, rel_tol=0.0, abs_tol=1e-9)
        for point, capital in zip(points, resolved, strict=True)
    ):
        raise ValueError("capacity evaluator returned a mismatched capital")
    return CapacityCurve(points=points)
