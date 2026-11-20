"""Cash-account wrapper of the anchor (Amendment 1 of ``agy_anchor_realistic_v1``).

After the signal's own caps, a proposal whose *effective* cash use would exceed
``protocol.CASH_LIMIT`` is rescaled proportionally to exactly that limit, where the cash use is

    spot long gross + INITIAL_MARGIN_RATE * perp gross

(perp gross is the sum of absolute perp weights; shorts count). ``INITIAL_MARGIN_RATE`` is
the engine's own ``1 / max_leverage`` (``trade_rl/simulation/execution.py:756``).

The mechanics are those of :class:`trade_rl.evaluation.trend_oos.cash_policy.CashAccountPolicy`
(reasoned there at length) with the measure above instead of the spot gross:

* the capped vector is the *effective* weight (held legs at their current weight, every other
  leg times ``risk_scale``), because that is what ``PreTradeRisk`` will execute;
* when the cap binds, every hold is dropped and the whole effective vector is re-targeted, so
  a deadband cannot let the measure creep above the limit;
* a proposal that does not trade (pure hold, or the pre-image of an unchanged risk scale)
  passes through untouched;
* the rescaled proposal is divided by ``risk_scale`` so risk applies the scale exactly once.

The wrapper reads only the snapshot of the decision (causal) and records, per rebalance
decision, the pre-cap measure and whether the cap bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trade_rl.evaluation.agy_anchor_oos import protocol
from trade_rl.strategies.agy_anchor.policy import risk_scale_of
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)

_NO_TRADE_ATOL = 1e-12
_EPS = 1e-9


def cash_measure(
    weights: np.ndarray, *, rate: float = protocol.INITIAL_MARGIN_RATE
) -> float:
    """Spot long gross plus ``rate`` times the perp gross of an alternating weight vector."""
    w = np.asarray(weights, dtype=np.float64)
    spot_long = float(np.maximum(w[0::2], 0.0).sum())
    perp_gross = float(np.abs(w[1::2]).sum())
    return spot_long + rate * perp_gross


@dataclass(slots=True)
class CashStatistics:
    """Reported (never gating) statistics of the cap over rebalance rows."""

    rebalance_decisions: int = 0
    bound_decisions: int = 0
    pre_cap_measure: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        n = self.rebalance_decisions
        m = self.pre_cap_measure
        return {
            "cash_limit": protocol.CASH_LIMIT,
            "initial_margin_rate": protocol.INITIAL_MARGIN_RATE,
            "rebalance_decisions": n,
            "cap_bound_decisions": self.bound_decisions,
            "cap_bound_fraction": (self.bound_decisions / n) if n else None,
            "mean_pre_cap_measure": (float(np.mean(m)) if m else None),
            "max_pre_cap_measure": (float(np.max(m)) if m else None),
        }


class AnchorCashAccountPolicy(PortfolioPolicy):
    """Any alternating-leg policy plus the cash-account cap (see module docstring)."""

    def __init__(
        self,
        inner: PortfolioPolicy,
        *,
        start_index: int,
        rebalance_every_bars: int = protocol.REBALANCE_EVERY_BARS,
        limit: float = protocol.CASH_LIMIT,
        rate: float = protocol.INITIAL_MARGIN_RATE,
    ) -> None:
        if not 0.0 < limit <= 1.0:
            raise ValueError("a cash account has a limit within (0, 1]")
        if not 0.0 < rate <= 1.0:
            raise ValueError("the initial margin rate must lie within (0, 1]")
        if start_index < 0 or rebalance_every_bars < 1:
            raise ValueError("invalid start index or stride")
        self._inner = inner
        self._start = start_index
        self._every = rebalance_every_bars
        self._limit = limit
        self._rate = rate
        self.stats = CashStatistics()

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        proposal = self._inner.decide_portfolio(snapshot)
        proposed = np.asarray(proposal.target_weights, dtype=np.float64)
        held = np.zeros(proposed.size, dtype=bool)
        if proposal.hold_current_quantity_symbols:
            held[list(proposal.hold_current_quantity_symbols)] = True
        current = np.asarray(snapshot.account.weights, dtype=np.float64)
        scale = risk_scale_of(snapshot)
        effective = np.where(held, current, scale * proposed)
        measure = cash_measure(effective, rate=self._rate)
        trades = not (
            held.all() or np.allclose(effective, current, atol=_NO_TRADE_ATOL)
        )
        binds = trades and measure > self._limit + _EPS and scale > 0.0
        if (snapshot.index - self._start) % self._every == 0:
            self.stats.rebalance_decisions += 1
            self.stats.pre_cap_measure.append(measure)
            self.stats.bound_decisions += int(binds)
        if not binds:
            return proposal
        capped = effective * (self._limit / measure) / scale
        return PortfolioTargetProposal(
            target_weights=tuple(float(w) for w in capped),
            hold_current_quantity_symbols=(),
        )


__all__ = ["AnchorCashAccountPolicy", "CashStatistics", "cash_measure"]
