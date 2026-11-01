"""Equal-weight buy-and-hold control implementing the existing ``PortfolioPolicy``."""

from __future__ import annotations

import math

from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)


class BuyAndHoldPortfolioPolicy:
    """Hold equal weights: trade at the first decision, then keep exact quantities.

    ``gross_budget`` is split equally and long across the roster at the first
    snapshot. Every later bar proposes the current weights with all symbols marked
    HOLD-current-quantity, so no further orders are generated (weights drift with
    prices). ``rebalance_every_bars`` optionally re-issues the equal-weight target
    on a fixed bar calendar counted from the first decision; it is off by default.
    The default budget leaves 1% cash because entry costs are paid from equity: a
    book invested at exactly the pre-trade gross limit would sit slightly above it
    and the shared risk layer would keep trimming it instead of holding.
    After account termination (insolvency or a termination reason) the policy holds
    every quantity instead of raising, matching the allocator's fail-closed hold.
    The policy owns no account; the shared portfolio session owns fills and P&L.
    Instances carry replay state and must not be reused across replays.
    """

    __slots__ = (
        "_symbols",
        "_gross_budget",
        "_rebalance_every_bars",
        "_first_index",
        "_last_timestamp_ns",
    )

    def __init__(
        self,
        *,
        symbols: tuple[str, ...],
        gross_budget: float = 0.99,
        rebalance_every_bars: int | None = None,
    ) -> None:
        if type(symbols) is not tuple or not symbols:
            raise TypeError("symbols must be a non-empty tuple")
        if any(not isinstance(value, str) or not value.strip() for value in symbols):
            raise ValueError("symbols must contain non-empty names")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique")
        if isinstance(gross_budget, bool) or not isinstance(gross_budget, (int, float)):
            raise TypeError("gross_budget must be a real number")
        if not math.isfinite(gross_budget) or not 0.0 < gross_budget <= 1.0:
            raise ValueError("gross_budget must be within (0, 1]")
        if rebalance_every_bars is not None and (
            type(rebalance_every_bars) is not int or rebalance_every_bars < 1
        ):
            raise ValueError("rebalance_every_bars must be a positive integer or None")
        self._symbols = symbols
        self._gross_budget = float(gross_budget)
        self._rebalance_every_bars = rebalance_every_bars
        self._first_index: int | None = None
        self._last_timestamp_ns: int | None = None

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if type(snapshot) is not PortfolioDecisionSnapshot:
            raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
        if snapshot.symbols != self._symbols:
            raise ValueError("snapshot symbols do not match the buy-and-hold roster")
        if (
            self._last_timestamp_ns is not None
            and snapshot.timestamp_ns <= self._last_timestamp_ns
        ):
            raise ValueError("snapshot timestamps must increase between decisions")
        self._last_timestamp_ns = snapshot.timestamp_ns
        account = snapshot.account
        n_symbols = len(self._symbols)
        hold_all = PortfolioTargetProposal(
            target_weights=account.weights,
            hold_current_quantity_symbols=tuple(range(n_symbols)),
        )
        if account.insolvent or account.termination_reason is not None:
            # Fail closed like the allocator: never open or add after termination.
            return hold_all
        if self._first_index is None:
            self._first_index = snapshot.index
            return self._equal_weights(n_symbols)
        if (
            self._rebalance_every_bars is not None
            and (snapshot.index - self._first_index) % self._rebalance_every_bars == 0
        ):
            return self._equal_weights(n_symbols)
        return hold_all

    def _equal_weights(self, n_symbols: int) -> PortfolioTargetProposal:
        return PortfolioTargetProposal(
            target_weights=(self._gross_budget / n_symbols,) * n_symbols
        )


__all__ = ["BuyAndHoldPortfolioPolicy"]
