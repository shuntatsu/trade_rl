"""Shared long-only spot proposal budget using the native account snapshot.

This is a pretrade estimate; the native executor remains the only book owner.
The wrapper does not promise a nonnegative fill-time cash balance when
prices, liquidity, fee terms or order state change between decision and fill.
"""

from __future__ import annotations

import math

from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.risk.spot_cash import spot_cash_target_factor
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)


class SpotCashBudgetPolicy:
    """Apply the common decision-mark fee reserve to an all-spot long-only policy.

    On a binding cap, release holds so native risk can reduce the whole
    vector. Divide by the native drawdown scale once, not twice. A no-trade
    request remains a quantity hold; native risk may still override it.
    The caller must use an all-spot, long-only roster and ensure the declared
    reserve covers its configured deterministic costs. Active orders are
    not supported by this decision-only budget.
    """

    def __init__(
        self, inner: PortfolioPolicy, *, reserve_rate: float, gross_limit: float = 1.0
    ) -> None:
        # Validate the same numeric contract even before the first snapshot.
        spot_cash_target_factor(
            (0.0,),
            (1.0,),
            cash_fraction=1.0,
            gross_limit=gross_limit,
            reserve_rate=reserve_rate,
        )
        self._inner = inner
        self._rate = float(reserve_rate)
        self._limit = float(gross_limit)

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        account = snapshot.account
        current = account.weights
        if account.insolvent or account.termination_reason is not None:
            return PortfolioTargetProposal(current, tuple(range(len(current))))
        if account.portfolio_value <= 0 or not math.isfinite(account.portfolio_value):
            raise ValueError("portfolio value must be positive and finite")
        if any(w < 0 for w in current) or any(q < 0 for q in account.quantities):
            raise ValueError("spot holdings must be nonnegative")
        if snapshot.active_orders:
            raise ValueError("spot decision budget does not admit active orders")
        proposal = self._inner.decide_portfolio(snapshot)
        weights = proposal.target_weights
        held = set(proposal.hold_current_quantity_symbols)
        if len(weights) != len(current) or any(i >= len(current) for i in held):
            raise ValueError("spot target/hold roster mismatch")
        if any(weights[i] != current[i] for i in held):
            raise ValueError("held spot target must equal the current native weight")
        if any(w < 0 for w in weights):
            raise ValueError("spot targets must be nonnegative")
        scale = PreTradeRisk(
            PreTradeRiskConfig(
                drawdown_start=snapshot.risk.drawdown_start,
                drawdown_stop=snapshot.risk.drawdown_stop,
            )
        ).risk_scale(account.max_drawdown)
        if scale == 0:
            return proposal
        effective = tuple(
            current[i] if i in held else scale * w for i, w in enumerate(weights)
        )
        if len(held) == len(current) or all(
            a == b for a, b in zip(effective, current, strict=True)
        ):
            return proposal
        factor = spot_cash_target_factor(
            current,
            effective,
            cash_fraction=account.cash / account.portfolio_value,
            gross_limit=self._limit,
            reserve_rate=self._rate,
        )
        if factor == 1.0:
            return proposal
        return PortfolioTargetProposal(tuple(w * factor / scale for w in effective))
