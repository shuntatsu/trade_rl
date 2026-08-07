"""Closed-position-cycle diagnostics reconstructed from execution fills."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class ClosedTradeDiagnostics:
    """Execution-cost-net PnL for flat-to-flat or sign-reversal cycles.

    Carry such as funding and borrow remains in the portfolio-level diagnostics because
    the execution results do not attribute it to individual position cycles.
    """

    closed_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    net_realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    open_positions: int = 0

    def __post_init__(self) -> None:
        for name in (
            "closed_trades",
            "winning_trades",
            "losing_trades",
            "breakeven_trades",
            "open_positions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.winning_trades + self.losing_trades + self.breakeven_trades != (
            self.closed_trades
        ):
            raise ValueError("closed trade outcome counts do not reconcile")
        for name in ("net_realized_pnl", "gross_profit", "gross_loss"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.gross_profit < 0.0 or self.gross_loss < 0.0:
            raise ValueError("closed trade gross profit/loss must be non-negative")
        if not math.isclose(
            self.net_realized_pnl,
            self.gross_profit - self.gross_loss,
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            raise ValueError("closed trade realized PnL does not reconcile")

    @property
    def win_rate(self) -> float | None:
        if self.closed_trades == 0:
            return None
        return self.winning_trades / self.closed_trades

    @property
    def average_realized_pnl(self) -> float | None:
        if self.closed_trades == 0:
            return None
        return self.net_realized_pnl / self.closed_trades

    @property
    def profit_factor(self) -> float | None:
        if self.gross_loss <= _TOLERANCE:
            return None
        return self.gross_profit / self.gross_loss

    def digest_payload(self) -> dict[str, object]:
        return {
            "average_realized_pnl": self.average_realized_pnl,
            "breakeven_trades": self.breakeven_trades,
            "closed_trades": self.closed_trades,
            "gross_loss": self.gross_loss,
            "gross_profit": self.gross_profit,
            "losing_trades": self.losing_trades,
            "net_realized_pnl": self.net_realized_pnl,
            "open_positions": self.open_positions,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "winning_trades": self.winning_trades,
        }

    @classmethod
    def combine(
        cls, values: tuple[ClosedTradeDiagnostics, ...]
    ) -> ClosedTradeDiagnostics:
        closed = sum(item.closed_trades for item in values)
        profit = sum(item.gross_profit for item in values)
        loss = sum(item.gross_loss for item in values)
        return cls(
            closed_trades=closed,
            winning_trades=sum(item.winning_trades for item in values),
            losing_trades=sum(item.losing_trades for item in values),
            breakeven_trades=sum(item.breakeven_trades for item in values),
            net_realized_pnl=profit - loss,
            gross_profit=profit,
            gross_loss=loss,
            open_positions=sum(item.open_positions for item in values),
        )


class ClosedTradeTracker:
    """Reconstruct signed position cycles from ordered execution fills."""

    def __init__(self, contract_multipliers: np.ndarray) -> None:
        multipliers = np.asarray(contract_multipliers, dtype=np.float64).reshape(-1)
        if multipliers.size == 0 or not np.isfinite(multipliers).all():
            raise ValueError("contract multipliers must be a non-empty finite vector")
        if np.any(multipliers <= 0.0):
            raise ValueError("contract multipliers must be positive")
        self._multipliers = multipliers.copy()
        self._quantities = np.zeros_like(multipliers)
        self._average_prices = np.zeros_like(multipliers)
        self._entry_costs = np.zeros_like(multipliers)
        self._cycle_pnl = np.zeros_like(multipliers)
        self._closed_pnls: list[float] = []

    @property
    def quantities(self) -> np.ndarray:
        return self._quantities.copy()

    def seed_positions(self, *, quantities: object, prices: object) -> None:
        """Seed positions that already exist at the evaluation boundary.

        The boundary prices become the diagnostic cost basis, so realized PnL is
        measured only over the evaluated path.  Seeding is deliberately allowed
        only before any fill has been ingested.
        """

        resolved_quantities = np.asarray(quantities, dtype=np.float64).reshape(-1)
        resolved_prices = np.asarray(prices, dtype=np.float64).reshape(-1)
        if (
            resolved_quantities.shape != self._quantities.shape
            or resolved_prices.shape != self._quantities.shape
        ):
            raise ValueError("seeded positions do not match the tracker universe")
        if (
            not np.isfinite(resolved_quantities).all()
            or not np.isfinite(resolved_prices).all()
        ):
            raise ValueError("seeded positions must be finite")
        open_mask = np.abs(resolved_quantities) > _TOLERANCE
        if np.any(resolved_prices[open_mask] <= 0.0):
            raise ValueError("seeded open positions require positive prices")
        if (
            np.any(np.abs(self._quantities) > _TOLERANCE)
            or self._closed_pnls
            or np.any(np.abs(self._entry_costs) > _TOLERANCE)
            or np.any(np.abs(self._cycle_pnl) > _TOLERANCE)
        ):
            raise RuntimeError("closed trade tracker positions are already initialized")
        self._quantities = resolved_quantities.copy()
        self._average_prices = np.where(open_mask, resolved_prices, 0.0)

    def _close_cycle(self, symbol: int) -> None:
        self._closed_pnls.append(float(self._cycle_pnl[symbol]))
        self._cycle_pnl[symbol] = 0.0

    def record_fill(
        self,
        *,
        symbol: int,
        quantity: float,
        price: float,
        execution_cost: float,
    ) -> None:
        if not 0 <= symbol < len(self._quantities):
            raise ValueError("fill symbol is outside the tracker universe")
        for name, value in (
            ("quantity", quantity),
            ("price", price),
            ("execution_cost", execution_cost),
        ):
            if not math.isfinite(value):
                raise ValueError(f"fill {name} must be finite")
        if price <= 0.0 or execution_cost < 0.0:
            raise ValueError("fill price/cost is invalid")
        if abs(quantity) <= _TOLERANCE:
            if execution_cost > _TOLERANCE:
                raise ValueError("zero-quantity fill cannot carry execution cost")
            return

        current = float(self._quantities[symbol])
        if abs(current) <= _TOLERANCE or math.copysign(1.0, current) == math.copysign(
            1.0, quantity
        ):
            old_abs = abs(current)
            fill_abs = abs(quantity)
            new_quantity = current + quantity
            self._average_prices[symbol] = (
                price
                if old_abs <= _TOLERANCE
                else (old_abs * self._average_prices[symbol] + fill_abs * price)
                / (old_abs + fill_abs)
            )
            self._entry_costs[symbol] += execution_cost
            self._quantities[symbol] = new_quantity
            return

        current_abs = abs(current)
        fill_abs = abs(quantity)
        closing_abs = min(current_abs, fill_abs)
        entry_cost = self._entry_costs[symbol] * closing_abs / current_abs
        closing_cost = execution_cost * closing_abs / fill_abs
        gross_pnl = (
            closing_abs
            * self._multipliers[symbol]
            * (price - self._average_prices[symbol])
            * math.copysign(1.0, current)
        )
        self._cycle_pnl[symbol] += gross_pnl - entry_cost - closing_cost
        self._entry_costs[symbol] -= entry_cost

        remaining_current = current_abs - closing_abs
        remaining_fill = fill_abs - closing_abs
        if remaining_current > _TOLERANCE:
            self._quantities[symbol] = math.copysign(remaining_current, current)
            return

        self._quantities[symbol] = 0.0
        self._average_prices[symbol] = 0.0
        self._entry_costs[symbol] = 0.0
        self._close_cycle(symbol)
        if remaining_fill > _TOLERANCE:
            self._quantities[symbol] = math.copysign(remaining_fill, quantity)
            self._average_prices[symbol] = price
            self._entry_costs[symbol] = execution_cost - closing_cost

    def ingest_stateful(self, result: Any) -> None:
        events = tuple(
            event
            for event in result.order_events
            if abs(float(event.filled_quantity)) > _TOLERANCE
        )
        cost_by_symbol = np.asarray(result.cost_by_symbol, dtype=np.float64)
        notional_totals = np.zeros_like(self._multipliers)
        for event in events:
            notional_totals[event.symbol_index] += abs(float(event.filled_notional))
        for event in events:
            symbol = int(event.symbol_index)
            if event.execution_price is None:
                raise ValueError("filled order event is missing its execution price")
            denominator = notional_totals[symbol]
            cost = (
                0.0
                if denominator <= _TOLERANCE
                else cost_by_symbol[symbol]
                * abs(float(event.filled_notional))
                / denominator
            )
            self.record_fill(
                symbol=symbol,
                quantity=float(event.filled_quantity),
                price=float(event.execution_price),
                execution_cost=float(cost),
            )
        self._assert_quantities(result.book.quantities)

    def ingest_liquidation(self, result: Any) -> None:
        resulting = np.asarray(result.book.quantities, dtype=np.float64)
        deltas = resulting - self._quantities
        notionals = np.asarray(result.filled_notional_by_symbol, dtype=np.float64)
        costs = np.asarray(result.cost_by_symbol, dtype=np.float64)
        for symbol, quantity in enumerate(deltas):
            if abs(quantity) <= _TOLERANCE:
                continue
            denominator = abs(quantity) * self._multipliers[symbol]
            if denominator <= _TOLERANCE or notionals[symbol] <= 0.0:
                raise ValueError("liquidation fill lacks executable notional")
            self.record_fill(
                symbol=symbol,
                quantity=float(quantity),
                price=float(notionals[symbol] / denominator),
                execution_cost=float(costs[symbol]),
            )
        self._assert_quantities(resulting)

    def _assert_quantities(self, expected: np.ndarray) -> None:
        resolved = np.asarray(expected, dtype=np.float64)
        if resolved.shape != self._quantities.shape or not np.allclose(
            resolved, self._quantities, rtol=1e-9, atol=1e-9
        ):
            raise ValueError("closed trade tracker quantities diverged from the book")

    def diagnostics(self) -> ClosedTradeDiagnostics:
        tolerance = 1e-8
        wins = sum(value > tolerance for value in self._closed_pnls)
        losses = sum(value < -tolerance for value in self._closed_pnls)
        breakeven = len(self._closed_pnls) - wins - losses
        profit = sum(max(value, 0.0) for value in self._closed_pnls)
        loss = sum(max(-value, 0.0) for value in self._closed_pnls)
        return ClosedTradeDiagnostics(
            closed_trades=len(self._closed_pnls),
            winning_trades=wins,
            losing_trades=losses,
            breakeven_trades=breakeven,
            net_realized_pnl=profit - loss,
            gross_profit=profit,
            gross_loss=loss,
            open_positions=int(np.count_nonzero(np.abs(self._quantities) > _TOLERANCE)),
        )


__all__ = ["ClosedTradeDiagnostics", "ClosedTradeTracker"]
