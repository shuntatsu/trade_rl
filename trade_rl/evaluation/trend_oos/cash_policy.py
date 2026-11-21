"""Spot cash-account wrapper of the unchanged lowturn policy.

``lowturn_family_screen_v1`` allowed a spot gross of 2.0 while the engine finances negative
cash at 0%. ``trend_long_spot_oos_v1`` trades on a spot cash account with no borrowing:
after the frozen lowturn caps (which this wrapper does not touch), a proposal whose
*effective* spot gross would exceed ``SPOT_GROSS_LIMIT`` is rescaled proportionally to
exactly that gross.

Why the wrapper looks at effective weights and drops holds when it binds:

* ``PreTradeRisk`` multiplies a proposal by ``risk_scale`` once, and an exact-quantity
  HOLD keeps the current weight. The weight a proposal will actually execute is therefore
  ``current`` for a held leg and ``risk_scale * proposal`` for every other leg; that is the
  vector whose gross is capped (gross is measured over the spot legs; no perpetual is ever available, which
  the replay tests pin).
* The lowturn deadband holds a leg whose *uncapped* target is within 0.02 of its current
  weight. A capped target is lower, so a held leg could keep more than its capped share
  and the mixture of held and re-targeted legs could creep above the limit. When the cap
  binds the wrapper therefore drops every hold and re-targets the whole effective vector
  scaled to gross exactly ``SPOT_GROSS_LIMIT``: no leg keeps weight the cap took from
  the others, and the resulting book cannot exceed the limit through the deadband.
* The lowturn pre-image proposal (``current / risk_scale`` while ``0 < risk_scale < 1``)
  executes to exactly the current weights, and a pure hold executes to the current book.
  Neither trades, and a trade-free proposal cannot change cash (a price move revalues
  positions and equity together), so both pass through untouched even if the current gross
  sits a hair above the limit from execution costs; capping them would only create
  churn.
* The rescaled proposal is divided by ``risk_scale`` again so that risk applies the scale
  exactly once. Risk only ever reduces a proposal further (clip, gross, turnover), so the
  limit holds after risk too.

The wrapper is causal: it reads only the snapshot of the decision. It records, per
rebalance decision, the pre-cap effective gross and whether the cap bound.
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
        # Relative allclose can classify real buys as holds while gross exceeds the cap.
        trades = not (
            held.all() or np.allclose(effective, current, atol=_NO_TRADE_ATOL, rtol=0.0)
        )
        rebalance = (snapshot.index - self._start) % self._every == 0
        binds = trades and gross > self._limit + _GROSS_EPS and scale > 0.0
        if rebalance:
            self.stats.rebalance_decisions += 1
            self.stats.pre_cap_gross.append(gross)
            self.stats.bound_decisions += int(binds)
        if not binds:
            return proposal
        capped = effective * (self._limit / gross) / scale
        return PortfolioTargetProposal(
            target_weights=tuple(float(w) for w in capped),
            hold_current_quantity_symbols=(),
        )


__all__ = ["CapStatistics", "CashAccountPolicy"]
