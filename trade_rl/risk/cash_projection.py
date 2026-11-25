"""Core-owned scalar decision-mark budgets for external portfolio studies.

These are target projections, not a mutable account, a venue margin model, or
a fill-time guarantee. The native engine still owns risk, orders, cash,
positions, funding, fees, and margin. In particular, signed perpetual notional
is booked by the existing native accounting semantics, not exchange P&L-only
collateral. No historical execution profile changes when this API is imported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    try:
        v = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be representable") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


@dataclass(frozen=True, slots=True)
class MixedCashProjection:
    """Pre-cap measures and uniform factors, all expressed as equity fractions."""

    margin_measure: float
    net_exposure: float
    factor: float
    margin_factor: float
    net_factor: float


def project_mixed_cash_budget(
    effective_weights: tuple[float, ...],
    *,
    spot_indices: tuple[int, ...],
    initial_margin_rate: float,
    cash_limit: float,
    net_exposure_limit: float,
    tolerance: float = 0.0,
) -> MixedCashProjection:
    """Project a long-only spot / signed perpetual target at a single mark.

    Every weight is a fraction of the same positive account equity. Explicit
    sorted unique spot indices designate long-only principal-funded positions;
    every other position is a perpetual with the declared initial margin rate
    (normally 1 / native execution max_leverage). Empty spot indices mean
    all-perpetual; no alternating-leg or symbol-name assumption is made.
    The margin measure is spot principal + rate * absolute perpetual gross,
    and the second, one-sided cap is on the signed sum of all weights.

    This reproduces the legacy AGY/XS decision projection for valid inputs
    when their declared tolerance is supplied. It is not an exchange
    certification, pending-order reservation, free-cash estimate or strict
    mixed-instrument execution guard. A later native risk or fill can change
    the actual portfolio. No orders, prices or account states are mutated.
    """
    if type(effective_weights) is not tuple or not effective_weights:
        raise ValueError("nonempty effective weight tuple required")
    if type(spot_indices) is not tuple or any(type(i) is not int for i in spot_indices):
        raise TypeError("spot indices must be a tuple of integer coordinates")
    if tuple(sorted(set(spot_indices))) != spot_indices or any(
        i < 0 or i >= len(effective_weights) for i in spot_indices
    ):
        raise ValueError("spot indices must be sorted unique and within the roster")
    w = np.array(
        [_finite(v, "effective weight") for v in effective_weights], dtype=np.float64
    )
    rate = _finite(initial_margin_rate, "initial margin rate")
    cap = _finite(cash_limit, "cash limit")
    net_cap = _finite(net_exposure_limit, "net exposure limit")
    eps = _finite(tolerance, "projection tolerance")
    if (
        not 0 < rate <= 1
        or not 0 < cap <= 1
        or not 0 < net_cap <= 1
        or not 0 <= eps <= 1e-8
    ):
        raise ValueError("rates and limits must be in (0,1], tolerance in [0,1e-8]")
    mask = np.zeros(w.size, dtype=np.bool_)
    mask[list(spot_indices)] = True
    if np.any(w[mask] < 0):
        raise ValueError("spot effective weights must be nonnegative")
    # Keep the original reduction order on valid frozen alternating/all-perp
    # rosters. Fail on nonrepresentable intermediate measures, not on genuine
    # cancellation in the signed net.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        spot = float(w[mask].sum())
        gross = float(np.abs(w[~mask]).sum())
        net = float(w.sum())
    margin = rate * gross
    measure = spot + margin
    if not all(math.isfinite(v) for v in (spot, gross, net, margin, measure)):
        raise ValueError("cash budget measures must be finite")
    if gross != 0 and margin == 0:
        raise ValueError("nonzero margin cannot be represented")
    mf = cap / measure if measure > cap + eps else 1.0
    nf = net_cap / net if net > net_cap + eps else 1.0
    factor = min(mf, nf)
    if factor <= 0:
        raise ValueError("nonzero projection factor cannot be represented")
    return MixedCashProjection(measure, net, factor, mf, nf)


def snapshot_drawdown_scale(
    *,
    portfolio_value: float,
    peak_value: float,
    max_drawdown: float,
    drawdown_start: float,
    drawdown_stop: float,
) -> float:
    """Legacy-compatible snapshot-to-risk ramp using canonical PreTradeRisk.

    The snapshot carries the native observed account values. Retain the
    historical 1e-12 denominator floor and [0,1] drawdown clamp used by
    external policies, including retained-debt snapshots, without copying
    the ramp itself. This does not revalue or update a book.
    """
    value = _finite(portfolio_value, "portfolio value")
    peak = _finite(peak_value, "peak value")
    saved = _finite(max_drawdown, "maximum drawdown")
    start = _finite(drawdown_start, "drawdown start")
    stop = _finite(drawdown_stop, "drawdown stop")
    if peak < 0 or saved < 0:
        raise ValueError("peak value and maximum drawdown must be nonnegative")
    observed = min(1.0, max(0.0, 1.0 - value / max(peak, 1e-12), saved))
    return PreTradeRisk(
        PreTradeRiskConfig(drawdown_start=start, drawdown_stop=stop)
    ).risk_scale(observed)
