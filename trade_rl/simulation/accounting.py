"""Portfolio book state and base-bar accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class BookState:
    """Mutable simulation state isolated from policy and workflow logic."""

    weights: np.ndarray
    portfolio_value: float
    peak_value: float
    max_drawdown: float = 0.0
    turnover_total: float = 0.0
    total_cost: float = 0.0
    funding_pnl: float = 0.0
    n_trades: int = 0
    returns_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1).copy()
        if weights.size == 0 or not np.isfinite(weights).all():
            raise ValueError("book weights must be a non-empty finite vector")
        if float(np.abs(weights).sum()) > 1.0 + 1e-12:
            raise ValueError("book gross exposure cannot exceed one")
        if not np.isfinite(self.portfolio_value) or self.portfolio_value <= 0.0:
            raise ValueError("portfolio_value must be finite and positive")
        if not np.isfinite(self.peak_value) or self.peak_value <= 0.0:
            raise ValueError("peak_value must be finite and positive")
        if self.peak_value + 1e-12 < self.portfolio_value:
            raise ValueError("peak_value cannot be below portfolio_value")
        object.__setattr__(self, "weights", weights)

    @classmethod
    def zero(cls, n_symbols: int, initial_capital: float) -> BookState:
        if n_symbols <= 0:
            raise ValueError("n_symbols must be positive")
        if not np.isfinite(initial_capital) or initial_capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
        return cls(
            weights=np.zeros(n_symbols, dtype=np.float64),
            portfolio_value=float(initial_capital),
            peak_value=float(initial_capital),
        )

    def clone(self) -> BookState:
        return BookState(
            weights=self.weights.copy(),
            portfolio_value=self.portfolio_value,
            peak_value=self.peak_value,
            max_drawdown=self.max_drawdown,
            turnover_total=self.turnover_total,
            total_cost=self.total_cost,
            funding_pnl=self.funding_pnl,
            n_trades=self.n_trades,
            returns_history=list(self.returns_history),
        )

    def record_base_bar(
        self,
        *,
        net_return: float,
        next_weights: np.ndarray,
        funding_amount: float,
    ) -> None:
        if not np.isfinite(net_return) or net_return <= -1.0:
            raise ValueError("base-bar net return must be finite and greater than -1")
        weights = np.asarray(next_weights, dtype=np.float64).reshape(-1)
        if weights.shape != self.weights.shape or not np.isfinite(weights).all():
            raise ValueError("next_weights do not match the book")
        if float(np.abs(weights).sum()) > 1.0 + 1e-9:
            raise ValueError("next_weights gross exposure cannot exceed one")
        previous_value = self.portfolio_value
        self.portfolio_value *= 1.0 + net_return
        if self.portfolio_value <= 0.0 or not np.isfinite(self.portfolio_value):
            raise ValueError("portfolio value became non-positive or non-finite")
        self.weights = weights.copy()
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.max_drawdown = max(
            self.max_drawdown,
            1.0 - self.portfolio_value / self.peak_value,
        )
        self.funding_pnl += float(funding_amount)
        self.returns_history.append(float(net_return))
        if previous_value <= 0.0:
            raise RuntimeError("book contained an invalid previous value")

    def record_rebalance(self, *, turnover: float, cost_amount: float) -> None:
        if turnover < 0.0 or not np.isfinite(turnover):
            raise ValueError("turnover must be finite and non-negative")
        if cost_amount < 0.0 or not np.isfinite(cost_amount):
            raise ValueError("cost_amount must be finite and non-negative")
        self.turnover_total += turnover
        self.total_cost += cost_amount
        if turnover > 1e-12:
            self.n_trades += 1
