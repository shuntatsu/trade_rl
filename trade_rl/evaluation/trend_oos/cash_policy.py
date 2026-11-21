"""Spot cash-account wrapper for the proposed trend-OOS v2 software revision.

The unchanged lowturn target may demand spot gross 2.0 while its replay permits free
cash financing. An actual spot cash account needs an execution-cost reserve as well as
a gross cap. At each trade decision, let ``g0`` be the current spot gross and ``r``
the largest *configured* deterministic per-notional fill cost across the frozen
scenarios. The next effective spot gross ``g1`` obeys
``g1 <= max(0, (gross_limit - r*g0)/(1+r))``. In addition, gross
spending on *new buys alone* plus the fee reserve must not exceed cash already
available before any sell fills. Future sell proceeds cannot finance earlier buys.

A cap that binds drops every exact-quantity hold; otherwise the old deadband can
leave excessive exposure. A pure HOLD or exact risk-scale pre-image creates no new
orders and is left untouched. Every retargeted effective weight is divided by the
pretrade risk scale once. The wrapper is causal and tracks pre-cap gross per rebalance.

This is not a guarantee of nonnegative *execution-time* cash: delayed price changes,
partial fills, pending orders and additional Dataset fee rates are not bounded by this
decision-mark estimate. The ledger audits ordered fills as well as interval closes
and invalidates a cell rather than reusing historical fees as fictional cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trade_rl.evaluation.lowturn_screen.policies import LowturnPolicy, risk_scale_of
from trade_rl.evaluation.trend_oos import protocol
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)

_NO_TRADE_ATOL = 1e-12
_GROSS_EPS = 1e-9


@dataclass(slots=True)
class CapStatistics:
    """Reported (never gating) statistics of the spot gross cap over rebalance rows."""

    rebalance_decisions: int = 0
    bound_decisions: int = 0
    pre_cap_gross: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        n = self.rebalance_decisions
        gross = self.pre_cap_gross
        return {
            "gross_limit": protocol.SPOT_GROSS_LIMIT,
            "rebalance_decisions": n,
            "cap_bound_decisions": self.bound_decisions,
            "cap_bound_fraction": (self.bound_decisions / n) if n else None,
            "mean_pre_cap_gross": (float(np.mean(gross)) if gross else None),
            "max_pre_cap_gross": (float(np.max(gross)) if gross else None),
        }


class CashAccountPolicy(PortfolioPolicy):
    """``LowturnPolicy`` plus the spot cash-account gross cap (see module docstring)."""

    def __init__(
        self,
        inner: LowturnPolicy,
        *,
        start_index: int,
        gross_limit: float = protocol.SPOT_GROSS_LIMIT,
        rebalance_every_bars: int = protocol.REBALANCE_EVERY_BARS,
    ) -> None:
        if not isinstance(inner, LowturnPolicy):
            raise TypeError("inner must be a LowturnPolicy")
        if not 0.0 < gross_limit <= 1.0:
            raise ValueError("a spot cash account has a gross limit within (0, 1]")
        self._inner = inner
        self._start = start_index
        self._limit = gross_limit
        self._every = rebalance_every_bars
        self.stats = CapStatistics()

    @property
    def family(self) -> str:
        return self._inner.family

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        proposal = self._inner.decide_portfolio(snapshot)
        proposed = np.asarray(proposal.target_weights, dtype=np.float64)
        n_legs = proposed.size
        spot = np.arange(n_legs // 2) * 2
        held = np.zeros(n_legs, dtype=bool)
        if proposal.hold_current_quantity_symbols:
            held[list(proposal.hold_current_quantity_symbols)] = True
        current = np.asarray(snapshot.account.weights, dtype=np.float64)
        scale = risk_scale_of(snapshot)
        effective = np.where(held, current, scale * proposed)
        gross = float(np.abs(effective[spot]).sum())
        # At one mark, total fill notional never exceeds current spot gross plus
        # target spot gross. Reserve the highest configured scenario cost against
        # that worst-case turnover before taking any new spot exposure.
        current_gross = float(np.abs(current[spot]).sum())
        fee_rate = protocol.cash_reserve_rate()
        cash_gross_limit = max(
            0.0, (self._limit - fee_rate * current_gross) / (1.0 + fee_rate)
        )
        trades = not (
            held.all() or np.allclose(effective, current, atol=_NO_TRADE_ATOL, rtol=0.0)
        )
        rebalance = (snapshot.index - self._start) % self._every == 0
        # The native processor fills symbols by ascending index, so all new
        # lower-index buys may execute before higher-index sells. Netting their
        # eventual proceeds at decision time would be unsecured borrowing.
        # Keep only cash that is already owned; never count sell proceeds here.
        value = float(snapshot.account.portfolio_value)
        cash_fraction = max(
            0.0, min(float(snapshot.account.cash) / value, 1.0 - current_gross)
        )
        current_spot = current[spot]
        target_spot = effective[spot]

        def buy_spend(factor: float) -> float:
            return float(np.maximum(factor * target_spot - current_spot, 0.0).sum())

        max_fraction = min(1.0, cash_gross_limit / gross) if gross > 0.0 else 1.0
        if (
            trades
            and scale > 0.0
            and (
                buy_spend(max_fraction) * (1.0 + fee_rate) > cash_fraction + _GROSS_EPS
            )
        ):
            # Buy principal and buy fees must be payable before any sell.
            # Sells finance their own fees; they never require upfront cash.
            # The separate gross cap covers worst-case *total* fill costs.
            # Monotonic in the uniform target factor.
            lower, upper = 0.0, max_fraction
            for _ in range(64):
                middle = (lower + upper) / 2.0
                if buy_spend(middle) * (1.0 + fee_rate) <= cash_fraction:
                    lower = middle
                else:
                    upper = middle
            max_fraction = lower
        binds = trades and max_fraction < 1.0 - _GROSS_EPS and scale > 0.0
        if rebalance:
            self.stats.rebalance_decisions += 1
            self.stats.pre_cap_gross.append(gross)
            self.stats.bound_decisions += int(binds)
        if not binds:
            return proposal
        capped = effective * max_fraction / scale
        return PortfolioTargetProposal(
            target_weights=tuple(float(w) for w in capped),
            hold_current_quantity_symbols=(),
        )


__all__ = ["CapStatistics", "CashAccountPolicy"]
