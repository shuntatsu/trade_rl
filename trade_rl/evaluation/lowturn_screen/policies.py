"""One ``PortfolioPolicy`` parametrised by low-turnover family.

Weekly rebalance, per-symbol deadband and exact-quantity HOLD between rebalances. The
risk-scale handling mirrors ``ShadowAnchorMtfPolicy``: ``PreTradeRisk`` multiplies the
whole proposal by ``risk_scale`` and an exact-quantity HOLD survives only when risk
leaves its weight intact, so a HOLD while ``risk_scale < 1`` would be scaled again on
every bar (compounding). The policy therefore

* re-proposes the last rebalance targets whenever ``risk_scale`` decreases, so risk
  applies the new scale to the intended book exactly once;
* while ``0 < risk_scale < 1`` is unchanged, re-proposes the pre-image
  ``current / risk_scale`` that risk maps back onto the current weights (no trade);
* otherwise holds exact quantities.

The engine's drawdown is the running maximum drawdown, so ``risk_scale`` never rises and
a 20% stop leaves the book flat for the rest of the segment.
"""

from __future__ import annotations

import math

import numpy as np

from trade_rl.evaluation.lowturn_screen import protocol
from trade_rl.evaluation.lowturn_screen.signals import LowturnSignals, signal_row
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)

_SCALE_EPS = 1e-12
_SPOT_SHORT_TOLERANCE = 1e-12


def apply_caps(weights: np.ndarray, *, gross_limit: float) -> np.ndarray:
    """Clip each ``|w|`` to the per-name cap, then scale down to ``gross_limit``."""
    clipped = np.clip(
        np.asarray(weights, dtype=np.float64),
        -protocol.PER_NAME_CAP,
        protocol.PER_NAME_CAP,
    )
    gross = float(np.abs(clipped).sum())
    if gross > gross_limit:
        clipped = clipped * (gross_limit / gross)
    return clipped


def _pair_legs(signals: LowturnSignals) -> tuple[np.ndarray, np.ndarray]:
    n_pairs = len(signals.pair_names)
    idx = np.arange(n_pairs)
    return 2 * idx, 2 * idx + 1  # spot legs, perp legs


def _pair_index(signals: LowturnSignals, name: str) -> int | None:
    try:
        return signals.pair_names.index(name)
    except ValueError:
        return None


def _trend_strength(
    signals: LowturnSignals, row: int, legs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(s, risk_weight)`` per leg; NaN where an input is unavailable."""
    signs = (
        np.sign(signals.r30[row, legs])
        + np.sign(signals.r90[row, legs])
        + np.sign(signals.r180[row, legs])
    ) / 3.0
    return signs, protocol.VOL_TARGET / signals.vol60[row, legs]


def _place(n_legs: int, legs: np.ndarray, per_pair: np.ndarray) -> np.ndarray:
    out = np.zeros(n_legs, dtype=np.float64)
    out[legs] = np.nan_to_num(per_pair, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def family_targets(family: str, signals: LowturnSignals, row: int) -> np.ndarray:
    """Capped target weight of every leg for ``family`` from signal row ``row``."""
    if family not in protocol.FAMILIES:
        raise ValueError(f"unknown family: {family!r}")
    n_legs = len(signals.symbols)
    spot, perp = _pair_legs(signals)
    gross_limit = protocol.GROSS_CAP
    raw = np.zeros(n_legs, dtype=np.float64)

    if family == "CASH":
        pass
    elif family == "EW_HOLD_SPOT":
        usable = np.asarray(signals.usable[row, spot], dtype=bool)
        raw[spot[usable]] = 1.0 / protocol.N_PAIRS
    elif family == "VT_HOLD_PERP":
        risk_weight = protocol.VOL_TARGET / signals.vol60[row, perp]
        raw = _place(n_legs, perp, risk_weight / protocol.N_PAIRS)
    elif family == "TS_TREND_LS":
        strength, risk_weight = _trend_strength(signals, row, perp)
        raw = _place(n_legs, perp, strength * risk_weight / protocol.N_PAIRS)
    elif family == "TS_TREND_LONG_SPOT":
        strength, risk_weight = _trend_strength(signals, row, spot)
        raw = _place(
            n_legs, spot, np.clip(strength, 0.0, 1.0) * risk_weight / protocol.N_PAIRS
        )
    elif family == "TS_TREND_BTCETH":
        strength, risk_weight = _trend_strength(signals, row, perp)
        per_pair = strength * risk_weight / len(protocol.BTC_ETH_PAIRS)
        keep = np.zeros(per_pair.shape, dtype=bool)
        for name in protocol.BTC_ETH_PAIRS:
            position = _pair_index(signals, name)
            if position is not None:
                keep[position] = True
        raw = _place(n_legs, perp, np.where(keep, per_pair, 0.0))
    elif family == "XS_MOM":
        gross_limit = protocol.XS_GROSS_CAP
        score = signals.xs_score[row, perp]
        risk_weight = protocol.VOL_TARGET / signals.vol60[row, perp]
        available = np.flatnonzero(np.isfinite(score) & np.isfinite(risk_weight))
        side = protocol.XS_SIDE_COUNT
        if available.size >= 2 * side:
            # one ranking (highest score first, ties by ascending pair index) so the
            # long and short sides are disjoint even when every score is tied
            order = available[np.argsort(-score[available], kind="stable")]
            top = order[:side]
            bottom = order[-side:]
            per_pair = np.zeros(perp.shape, dtype=np.float64)
            per_pair[top] = risk_weight[top] / (2 * side)
            per_pair[bottom] = -risk_weight[bottom] / (2 * side)
            raw = _place(n_legs, perp, per_pair)
    else:  # MA_BTCETH_SPOT
        per_pair = np.zeros(spot.shape, dtype=np.float64)
        for name in protocol.BTC_ETH_PAIRS:
            position = _pair_index(signals, name)
            if position is None:
                continue
            leg = int(spot[position])
            fast = signals.sma_fast[row, leg]
            slow = signals.sma_slow[row, leg]
            if np.isfinite(fast) and np.isfinite(slow) and fast > slow:
                per_pair[position] = protocol.MA_LEG_WEIGHT
        raw = _place(n_legs, spot, per_pair)

    return apply_caps(raw, gross_limit=gross_limit)


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


class LowturnPolicy(PortfolioPolicy):
    """Weekly-rebalanced low-turnover family policy over the alternating spot/perp legs."""

    def __init__(
        self,
        *,
        family: str,
        signals: LowturnSignals,
        start_index: int,
        signal_lag_bars: int = 0,
        rebalance_every_bars: int = protocol.REBALANCE_EVERY_BARS,
        deadband: float = protocol.DEADBAND,
    ) -> None:
        if family not in protocol.FAMILIES:
            raise ValueError(f"unknown family: {family!r}")
        if not isinstance(signals, LowturnSignals):
            raise TypeError("signals must be LowturnSignals")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if signal_lag_bars not in (0, 1):
            raise ValueError("signal_lag_bars must be 0 or 1")
        if rebalance_every_bars < 1:
            raise ValueError("rebalance_every_bars must be >= 1")
        if not math.isfinite(deadband) or deadband < 0.0:
            raise ValueError("deadband must be finite and non-negative")
        self._family = family
        self._signals = signals
        self._start = start_index
        self._lag = signal_lag_bars
        self._every = rebalance_every_bars
        self._deadband = deadband
        spot, _ = _pair_legs(signals)
        self._spot_legs = spot
        self._last_index: int | None = None
        self._intended: np.ndarray | None = None
        self._last_risk_scale = 1.0

    @property
    def family(self) -> str:
        return self._family

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        proposal = self._decide(snapshot)
        self._last_risk_scale = risk_scale_of(snapshot)
        return proposal

    def _validate(self, snapshot: PortfolioDecisionSnapshot) -> None:
        if snapshot.symbols != self._signals.symbols:
            raise ValueError("snapshot symbols do not match the signal legs")
        if not 0 <= snapshot.index < self._signals.n_rows:
            raise ValueError(f"snapshot.index {snapshot.index} outside the signal rows")
        if snapshot.index < self._start:
            raise ValueError("snapshot.index precedes the segment start")
        if self._last_index is not None and snapshot.index <= self._last_index:
            raise ValueError("snapshot.index must advance monotonically")
        self._last_index = snapshot.index

    def _held_proposal(
        self,
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

        row = signal_row(snapshot.index, self._lag)
        target = family_targets(self._family, self._signals, row)
        if float(np.min(target[self._spot_legs], initial=0.0)) < -_SPOT_SHORT_TOLERANCE:
            raise ValueError("negative spot target: spot legs are long-only")
        self._intended = target
        held = np.abs(target * scale - current) < self._deadband
        return self._held_proposal(snapshot, scale, held, target)


__all__ = [
    "LowturnPolicy",
    "apply_caps",
    "family_targets",
    "risk_scale_of",
]
