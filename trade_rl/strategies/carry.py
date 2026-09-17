"""Paired spot/perpetual carry policy, independent of execution and providers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from trade_rl.simulation.quantities import quantize_quantity


@dataclass(frozen=True)
class CarryConfig:
    gross_budget: float = 0.5
    common_lot: float = 0.001
    maximum_drawdown: float = 0.1

    def __post_init__(self) -> None:
        for name in ("gross_budget", "common_lot", "maximum_drawdown"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 < value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in (0, 1]")


class FundingCarryBot:
    """Monthly matched-quantity carry with a permanent emergency stop.

    Arrays alternate spot, perpetual for each pair. Only observed prices and
    realized positions are inputs; funding forecasts do not influence sizing.
    """

    def __init__(self, config: CarryConfig) -> None:
        self.config = config
        self.stop_reason: str | None = None
        self._month: tuple[int, int] | None = None
        self._target: np.ndarray | None = None

    def stop(self, reason: str) -> None:
        if not reason:
            raise ValueError("stop reason must not be empty")
        if self.stop_reason is None:
            self.stop_reason = reason

    def decide(
        self,
        *,
        timestamp: datetime,
        prices: np.ndarray,
        quantities: np.ndarray,
        equity: float,
        drawdown: float,
    ) -> np.ndarray:
        prices = np.asarray(prices, dtype=np.float64)
        quantities = np.asarray(quantities, dtype=np.float64)
        if (
            prices.ndim != 1
            or len(prices) == 0
            or len(prices) % 2
            or prices.shape != quantities.shape
            or not np.all(np.isfinite(prices) & (prices > 0))
            or not np.all(np.isfinite(quantities))
            or not math.isfinite(equity)
            or equity <= 0
            or not math.isfinite(drawdown)
            or not 0 <= drawdown <= 1
            or (self._target is not None and self._target.shape != prices.shape)
        ):
            raise ValueError("carry observation must contain valid aligned pairs")
        if np.any(quantities[::2] < -1e-9) or np.any(quantities[1::2] > 1e-9):
            self.stop("invalid_position_direction")
        if np.any(np.abs(quantities[::2] + quantities[1::2]) > 1e-9):
            self.stop("unmatched_hedge")
        if drawdown >= self.config.maximum_drawdown:
            self.stop("maximum_drawdown")
        if self.stop_reason:
            return np.zeros_like(prices)
        month = (timestamp.year, timestamp.month)
        if self._target is None or self._month != month:
            notional = equity * self.config.gross_budget / len(prices)
            units = notional / np.maximum(prices[::2], prices[1::2])
            units = np.asarray(
                [
                    quantize_quantity(float(value), self.config.common_lot)[0]
                    for value in units
                ]
            )
            self._target = np.empty_like(prices)
            self._target[::2] = units
            self._target[1::2] = -units
            self._month = month
        return self._target.copy()
