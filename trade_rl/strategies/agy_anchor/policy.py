"""``AgyAnchorPolicy``: the agy anchor as a ``PortfolioPolicy`` on the 22-leg layout.

Decision rule (lab ``run_r8_4h_sim`` + ``synthesize_aw_pa``, 4H L1): every
``rebalance_every_bars`` bars (6 = daily) the policy proposes the synthesized anchor
target from signal row ``index - signal_lag_bars`` (L1 = 1, so a decision never reads the
current row). If the total turnover to that target is below ``deadband`` the whole book is
held; legs that are not valid on the current row keep their current quantity (lab
``ws_exec = where(valid, target, current)``). Between rebalance bars the book is held.

Risk-scale handling (same contract as ``ShadowAnchorMtfPolicy``): ``PreTradeRisk``
multiplies the whole proposal by ``risk_scale`` and an exact-quantity HOLD survives only
when risk leaves its weight intact, so a HOLD while ``risk_scale < 1`` would be scaled
again every bar (compounding). The policy therefore

* re-proposes the last intended targets whenever ``risk_scale`` decreases, so risk
  applies the new scale to the intended book exactly once;
* while ``0 < risk_scale < 1`` is unchanged, re-proposes the pre-image
  ``current / risk_scale`` that risk maps back onto the current weights (no trade);
* otherwise holds exact quantities.
"""

from __future__ import annotations

import math

import numpy as np

from trade_rl.strategies.agy_anchor.signals import AnchorSignals
from trade_rl.strategies.agy_anchor.targets import (
    DEADBAND,
    REBALANCE_EVERY_BARS,
    interleave_legs,
    synthesize_row,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)

_SCALE_EPS = 1e-12


def risk_scale_of(snapshot: PortfolioDecisionSnapshot) -> float:
    """``PreTradeRisk.risk_scale`` of the snapshot's drawdown (same linear ramp)."""
    peak = max(float(snapshot.account.peak_value), 1e-12)
    current_dd = max(0.0, 1.0 - float(snapshot.account.portfolio_value) / peak)
    drawdown = min(1.0, max(current_dd, float(snapshot.account.max_drawdown)))
    start = float(snapshot.risk.drawdown_start)
    stop = float(snapshot.risk.drawdown_stop)
    if drawdown <= start:
        return 1.0
    if drawdown >= stop or stop == start:
        return 0.0
    return 1.0 - (drawdown - start) / (stop - start)


class AgyAnchorPolicy(PortfolioPolicy):
    """Daily-rebalanced agy anchor over alternating ``<pair>:spot, <pair>:perp`` legs."""

    def __init__(
        self,
        *,
        signals: AnchorSignals,
        start_index: int,
        signal_lag_bars: int = 1,
        rebalance_every_bars: int = REBALANCE_EVERY_BARS,
        deadband: float = DEADBAND,
    ) -> None:
        if not isinstance(signals, AnchorSignals):
            raise TypeError("signals must be AnchorSignals")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if signal_lag_bars < 0:
            raise ValueError("signal_lag_bars must be non-negative")
        if rebalance_every_bars < 1:
            raise ValueError("rebalance_every_bars must be >= 1")
        if not math.isfinite(deadband) or deadband < 0.0:
            raise ValueError("deadband must be finite and non-negative")
        self._signals = signals
        self._start = start_index
        self._lag = signal_lag_bars
        self._every = rebalance_every_bars
        self._deadband = deadband
        self._expected_symbols = tuple(
            leg for pair in signals.symbols for leg in (f"{pair}:spot", f"{pair}:perp")
        )
        self._last_index: int | None = None
        self._intended: np.ndarray | None = None
        self._last_risk_scale = 1.0

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        proposal = self._decide(snapshot)
        self._last_risk_scale = risk_scale_of(snapshot)
        return proposal

    def _validate(self, snapshot: PortfolioDecisionSnapshot) -> None:
        if snapshot.symbols != self._expected_symbols:
            raise ValueError(
                "snapshot symbols do not match the alternating anchor legs"
            )
        if not 0 <= snapshot.index < self._signals.n_rows:
            raise ValueError(f"snapshot.index {snapshot.index} outside the signal rows")
        if snapshot.index < self._start:
            raise ValueError("snapshot.index precedes the segment start")
        if self._last_index is not None and snapshot.index <= self._last_index:
            raise ValueError("snapshot.index must advance monotonically")
        self._last_index = snapshot.index

    @staticmethod
    def _held_proposal(
        snapshot: PortfolioDecisionSnapshot,
        scale: float,
        held: np.ndarray,
        intended: np.ndarray,
    ) -> PortfolioTargetProposal:
        """Proposal that keeps ``held`` legs where they are and re-targets the rest."""
        current = np.asarray(snapshot.account.weights, dtype=np.float64)
        if scale >= 1.0:
            weights = np.where(held, current, intended)
            holds = tuple(int(i) for i in np.flatnonzero(held))
            return PortfolioTargetProposal(
                target_weights=tuple(float(w) for w in weights),
                hold_current_quantity_symbols=holds,
            )
        if scale > 0.0 and bool(held.any()):
            pre_image = np.where(held, current / scale, intended)
            if float(np.max(np.abs(pre_image), initial=0.0)) <= float(
                snapshot.risk.max_abs_weight
            ) and float(np.sum(np.abs(pre_image))) <= float(snapshot.risk.max_gross):
                return PortfolioTargetProposal(
                    target_weights=tuple(float(w) for w in pre_image)
                )
        return PortfolioTargetProposal(target_weights=tuple(float(w) for w in intended))

    def _decide(self, snapshot: PortfolioDecisionSnapshot) -> PortfolioTargetProposal:
        self._validate(snapshot)
        scale = risk_scale_of(snapshot)
        current = np.asarray(snapshot.account.weights, dtype=np.float64)

        if (snapshot.index - self._start) % self._every != 0:
            if self._intended is None:
                raise ValueError("the first decision must be a rebalance bar")
            decreased = scale < self._last_risk_scale - _SCALE_EPS
            if decreased or scale <= 0.0:
                # risk applies the (lower) scale to the intended book exactly once
                held = np.zeros(current.shape, dtype=bool)
            else:
                held = np.ones(current.shape, dtype=bool)
            return self._held_proposal(snapshot, scale, held, self._intended)

        row = max(0, snapshot.index - self._lag)
        spot, perp = synthesize_row(self._signals, row)
        target = interleave_legs(spot, perp)
        self._intended = target
        valid = interleave_legs(
            self._signals.spot_valid[snapshot.index].astype(np.float64),
            self._signals.perp_valid[snapshot.index].astype(np.float64),
        ).astype(bool)
        executed = np.where(valid, target * scale, current)
        turnover = float(np.abs(executed - current).sum())
        if turnover < self._deadband:
            held = np.ones(current.shape, dtype=bool)
        else:
            held = ~valid
        return self._held_proposal(snapshot, scale, held, target)


__all__ = ["AgyAnchorPolicy", "risk_scale_of"]
