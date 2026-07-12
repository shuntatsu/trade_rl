from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from mars_lite.trading.execution import ExecutionModel


@dataclass
class BookState:
    weights: np.ndarray
    portfolio_value: float = 1.0
    peak_value: float = 1.0
    max_drawdown: float = 0.0
    turnover_total: float = 0.0
    funding_pnl: float = 0.0
    total_cost: float = 0.0
    n_trades: int = 0
    returns_history: list[float] = field(default_factory=list)

    @classmethod
    def zero(cls, n_symbols: int, initial_capital: float = 1.0) -> "BookState":
        if n_symbols <= 0:
            raise ValueError("n_symbols must be positive")
        if not np.isfinite(initial_capital) or initial_capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
        return cls(
            weights=np.zeros(n_symbols, dtype=np.float64),
            portfolio_value=float(initial_capital),
            peak_value=float(initial_capital),
        )

    def copy(self) -> "BookState":
        return replace(
            self,
            weights=np.asarray(self.weights, dtype=np.float64).copy(),
            returns_history=list(self.returns_history),
        )


@dataclass(frozen=True)
class IntervalExecution:
    book: BookState
    bars_advanced: int
    next_t: int
    interval_turnover: float
    interval_cost: float
    interval_funding: float
    interval_gross_return: float
    interval_net_return: float
    interval_log_return: float
    bar_returns: tuple[float, ...]


class MarketExecutionCore:
    """Execute a fixed target over multiple base bars with one entry cost."""

    def __init__(self, fs, execution_model: ExecutionModel):
        self.fs = fs
        self.execution_model = execution_model

    def execute_interval(
        self,
        book: BookState,
        target: np.ndarray,
        *,
        start_t: int,
        bars: int,
    ) -> IntervalExecution:
        if bars <= 0:
            raise ValueError("bars must be positive")
        if start_t < 0:
            raise ValueError("start_t must be non-negative")

        target_weights = np.asarray(target, dtype=np.float64).reshape(-1)
        current_weights = np.asarray(book.weights, dtype=np.float64).reshape(-1)
        if target_weights.shape != current_weights.shape:
            raise ValueError("target shape must match book weights")
        if not np.all(np.isfinite(target_weights)):
            raise ValueError("target must be finite")

        last_start_exclusive = max(0, self.fs.n_bars - 2)
        bars_advanced = max(
            0, min(int(bars), last_start_exclusive - int(start_t))
        )
        if bars_advanced == 0:
            return IntervalExecution(
                book=book.copy(),
                bars_advanced=0,
                next_t=int(start_t),
                interval_turnover=0.0,
                interval_cost=0.0,
                interval_funding=0.0,
                interval_gross_return=0.0,
                interval_net_return=0.0,
                interval_log_return=0.0,
                bar_returns=(),
            )

        next_book = book.copy()
        value_before = float(next_book.portfolio_value)
        delta = target_weights - current_weights
        turnover = float(np.abs(delta).sum())
        cost = self.execution_model.cost_fraction(delta)
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError("execution cost must be finite and non-negative")

        interval_funding = 0.0
        gross_components: list[float] = []
        net_returns: list[float] = []
        for offset in range(bars_advanced):
            t = start_t + offset
            r_vec = self.fs.close[t + 1] / self.fs.close[t] - 1.0
            funding = float(np.sum(target_weights * self.fs.funding_rate[t + 1]))
            gross = float(np.dot(target_weights, r_vec))
            net = gross - funding - (cost if offset == 0 else 0.0)
            if not np.isfinite(gross) or not np.isfinite(funding) or not np.isfinite(net):
                raise ValueError("interval execution produced a non-finite value")
            next_book.portfolio_value *= 1.0 + net
            next_book.peak_value = max(next_book.peak_value, next_book.portfolio_value)
            if next_book.peak_value > 0.0:
                next_book.max_drawdown = max(
                    next_book.max_drawdown,
                    1.0 - next_book.portfolio_value / next_book.peak_value,
                )
            interval_funding += funding
            gross_components.append(gross)
            net_returns.append(net)

        next_book.weights = target_weights.copy()
        next_book.turnover_total += turnover
        next_book.funding_pnl += interval_funding
        next_book.total_cost += cost
        next_book.returns_history.extend(net_returns)
        if turnover > 0.0:
            next_book.n_trades += 1

        value_after = float(next_book.portfolio_value)
        interval_net_return = value_after / value_before - 1.0
        interval_log_return = (
            float(np.log(value_after / value_before))
            if value_before > 0.0 and value_after > 0.0
            else float("-inf")
        )
        interval_gross_return = float(
            np.prod(1.0 + np.asarray(gross_components, dtype=np.float64)) - 1.0
        )
        return IntervalExecution(
            book=next_book,
            bars_advanced=bars_advanced,
            next_t=start_t + bars_advanced,
            interval_turnover=turnover,
            interval_cost=cost,
            interval_funding=interval_funding,
            interval_gross_return=interval_gross_return,
            interval_net_return=interval_net_return,
            interval_log_return=interval_log_return,
            bar_returns=tuple(net_returns),
        )
