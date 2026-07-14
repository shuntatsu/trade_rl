"""Immutable execution evidence carried with evaluation return series."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionDiagnostics:
    """Economic and execution diagnostics for one evaluated range."""

    turnover_total: float = 0.0
    total_cost: float = 0.0
    funding_pnl: float = 0.0
    borrow_cost: float = 0.0
    n_trades: int = 0
    rebalance_events: int = 0
    termination_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name, value in (
            ("turnover_total", self.turnover_total),
            ("total_cost", self.total_cost),
            ("borrow_cost", self.borrow_cost),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if not math.isfinite(self.funding_pnl):
            raise ValueError("funding_pnl must be finite")
        for field_name, value in (
            ("n_trades", self.n_trades),
            ("rebalance_events", self.rebalance_events),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if any(not reason for reason in self.termination_reasons):
            raise ValueError("termination reasons must be non-empty")

    @property
    def termination_count(self) -> int:
        return len(self.termination_reasons)

    def digest_payload(self) -> dict[str, object]:
        return {
            "borrow_cost": self.borrow_cost,
            "funding_pnl": self.funding_pnl,
            "n_trades": self.n_trades,
            "rebalance_events": self.rebalance_events,
            "termination_reasons": self.termination_reasons,
            "total_cost": self.total_cost,
            "turnover_total": self.turnover_total,
        }

    @classmethod
    def combine(cls, values: Iterable[ExecutionDiagnostics]) -> ExecutionDiagnostics:
        items = tuple(values)
        return cls(
            turnover_total=sum(item.turnover_total for item in items),
            total_cost=sum(item.total_cost for item in items),
            funding_pnl=sum(item.funding_pnl for item in items),
            borrow_cost=sum(item.borrow_cost for item in items),
            n_trades=sum(item.n_trades for item in items),
            rebalance_events=sum(item.rebalance_events for item in items),
            termination_reasons=tuple(
                reason for item in items for reason in item.termination_reasons
            ),
        )
