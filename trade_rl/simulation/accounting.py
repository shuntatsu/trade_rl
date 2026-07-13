"""Self-financing portfolio book state and base-bar accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_TOLERANCE = 1e-12


def _finite_vector(value: np.ndarray, *, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{field_name} must be a non-empty finite vector")
    return vector


@dataclass(slots=True)
class BookState:
    """Mutable signed-quantity account isolated from policy and workflow logic."""

    quantities: np.ndarray
    cash: float
    mark_prices: np.ndarray
    peak_value: float
    max_drawdown: float = 0.0
    turnover_total: float = 0.0
    total_cost: float = 0.0
    funding_pnl: float = 0.0
    fill_count: int = 0
    rebalance_events: int = 0
    returns_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        quantities = _finite_vector(self.quantities, field_name="quantities")
        marks = _finite_vector(self.mark_prices, field_name="mark_prices")
        if quantities.shape != marks.shape:
            raise ValueError("quantities and mark_prices must have identical shapes")
        if np.any(marks <= 0.0):
            raise ValueError("mark_prices must be strictly positive")
        for field_name, value in (
            ("cash", self.cash),
            ("peak_value", self.peak_value),
            ("max_drawdown", self.max_drawdown),
            ("turnover_total", self.turnover_total),
            ("total_cost", self.total_cost),
            ("funding_pnl", self.funding_pnl),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.peak_value <= 0.0:
            raise ValueError("peak_value must be positive")
        if self.fill_count < 0 or self.rebalance_events < 0:
            raise ValueError("execution counters must be non-negative")
        object.__setattr__(self, "quantities", quantities)
        object.__setattr__(self, "mark_prices", marks)
        self._validate_equity()
        if self.peak_value + _TOLERANCE < self.portfolio_value:
            raise ValueError("peak_value cannot be below portfolio_value")

    @classmethod
    def zero(
        cls,
        n_symbols: int,
        initial_capital: float,
        initial_prices: np.ndarray | None = None,
    ) -> BookState:
        if n_symbols <= 0:
            raise ValueError("n_symbols must be positive")
        if not np.isfinite(initial_capital) or initial_capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
        prices = (
            np.ones(n_symbols, dtype=np.float64)
            if initial_prices is None
            else _finite_vector(initial_prices, field_name="initial_prices")
        )
        if prices.shape != (n_symbols,) or np.any(prices <= 0.0):
            raise ValueError("initial_prices must match symbols and be positive")
        return cls(
            quantities=np.zeros(n_symbols, dtype=np.float64),
            cash=float(initial_capital),
            mark_prices=prices,
            peak_value=float(initial_capital),
        )

    @property
    def position_values(self) -> np.ndarray:
        return self.quantities * self.mark_prices

    @property
    def portfolio_value(self) -> float:
        return float(self.cash + self.position_values.sum())

    @property
    def weights(self) -> np.ndarray:
        value = self.portfolio_value
        if value <= 0.0:
            return np.zeros_like(self.quantities)
        return self.position_values / value

    @property
    def n_trades(self) -> int:
        """Compatibility alias: one trade means one filled symbol order."""

        return self.fill_count

    def clone(self) -> BookState:
        return BookState(
            quantities=self.quantities.copy(),
            cash=self.cash,
            mark_prices=self.mark_prices.copy(),
            peak_value=self.peak_value,
            max_drawdown=self.max_drawdown,
            turnover_total=self.turnover_total,
            total_cost=self.total_cost,
            funding_pnl=self.funding_pnl,
            fill_count=self.fill_count,
            rebalance_events=self.rebalance_events,
            returns_history=list(self.returns_history),
        )

    def revalue(self, mark_prices: np.ndarray) -> None:
        marks = _finite_vector(mark_prices, field_name="mark_prices")
        if marks.shape != self.quantities.shape or np.any(marks <= 0.0):
            raise ValueError("mark_prices must match the book and be positive")
        self.mark_prices = marks
        self._validate_equity()

    def execute(
        self,
        *,
        fill_prices: np.ndarray,
        target_quantities: np.ndarray,
        cost_amount: float,
        turnover: float,
    ) -> None:
        prices = _finite_vector(fill_prices, field_name="fill_prices")
        targets = _finite_vector(target_quantities, field_name="target_quantities")
        if (
            prices.shape != self.quantities.shape
            or targets.shape != self.quantities.shape
        ):
            raise ValueError("execution vectors must match the book")
        if np.any(prices <= 0.0):
            raise ValueError("fill_prices must be strictly positive")
        if not np.isfinite(cost_amount) or cost_amount < 0.0:
            raise ValueError("cost_amount must be finite and non-negative")
        if not np.isfinite(turnover) or turnover < 0.0:
            raise ValueError("turnover must be finite and non-negative")

        delta = targets - self.quantities
        filled = np.abs(delta) > _TOLERANCE
        self.cash -= float(np.dot(delta, prices)) + float(cost_amount)
        self.quantities = targets
        self.mark_prices = prices
        self.turnover_total += float(turnover)
        self.total_cost += float(cost_amount)
        fill_count = int(np.count_nonzero(filled))
        self.fill_count += fill_count
        if fill_count:
            self.rebalance_events += 1
        self._validate_equity()
        value = self.portfolio_value
        self.peak_value = max(self.peak_value, value)
        self.max_drawdown = max(
            self.max_drawdown,
            1.0 - value / self.peak_value,
        )

    def mark_to_market(
        self,
        *,
        mark_prices: np.ndarray,
        funding_amount: float,
        period_start_value: float | None = None,
    ) -> float:
        if not np.isfinite(funding_amount):
            raise ValueError("funding_amount must be finite")
        starting_value = (
            self.portfolio_value
            if period_start_value is None
            else float(period_start_value)
        )
        if not np.isfinite(starting_value) or starting_value <= 0.0:
            raise ValueError("period_start_value must be finite and positive")

        self.cash += float(funding_amount)
        self.revalue(mark_prices)
        value = self.portfolio_value
        net_return = value / starting_value - 1.0
        if not np.isfinite(net_return) or net_return <= -1.0:
            raise ValueError("base-bar net return must be finite and greater than -1")
        self.peak_value = max(self.peak_value, value)
        self.max_drawdown = max(
            self.max_drawdown,
            1.0 - value / self.peak_value,
        )
        self.funding_pnl += float(funding_amount)
        self.returns_history.append(float(net_return))
        return float(net_return)

    def _validate_equity(self) -> None:
        value = self.portfolio_value
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("portfolio value became non-positive or non-finite")
