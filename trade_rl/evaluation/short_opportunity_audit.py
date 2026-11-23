"""Read-only, completed-day candidate short returns conditioned on R1/R2 gates.

The sampled 24h price change is a FORWARD LABEL for post-hoc diagnostics
only; NEVER feed it into a trading policy, feature roster, portfolio replay,
or any account P&L/economic winner assertion. Only known-used data permitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from trade_rl.data.features.cloud_regime_context import CloudRegimeContext
from trade_rl.data.features.divergence_mtf import DivergenceCloudMtfPack

ShortGroup = Literal[
    "G0_momentum_nonnegative",
    "G1_momentum_negative_slope_nonnegative",
    "G2_both_negative",
]
_SHORT_GROUPS: tuple[ShortGroup, ...] = (
    "G0_momentum_nonnegative",
    "G1_momentum_negative_slope_nonnegative",
    "G2_both_negative",
)
_NS_HOUR = 3_600_000_000_000
_NS_DAY = 24 * _NS_HOUR
_HORIZON_HOURS = 24
_ENTRY_CONFIRM_HOURS = 6


@dataclass(frozen=True, slots=True)
class ShortOpportunityObservation:
    """One completed UTC day short candidate, evaluated offline one day later."""

    index: int
    next_index: int
    timestamp_ns: int
    group: ShortGroup
    signed_24h_return_pct: float
    normalized_24h_short_move: float | None


@dataclass(frozen=True, slots=True)
class ShortOpportunityGroupStats:
    count: int
    mean_signed_24h_pct: float | None
    median_signed_24h_pct: float | None
    favorable_fraction: float | None
    mean_vol_normalized_move: float | None


def audit_24h_short_opportunities(
    *,
    close: np.ndarray,
    signals: DivergenceCloudMtfPack,
    context: CloudRegimeContext,
    start_index: int,
    stop_index: int,
) -> tuple[ShortOpportunityObservation, ...]:
    """Evaluate all *observed* daily C2 short candidates, never the sealed future.

    Uses only t's **completed** context to assign the group. The next t+24
    trade price is read solely as an offline response label inside the
    caller's explicit, already-consumed study window.
    """
    c = np.asarray(close, dtype=np.float64)
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("close must be one-dimensional, positive and finite")
    n = len(c)
    if type(start_index) is not int or type(stop_index) is not int:
        raise TypeError("explicit study bounds must be integers")
    if not (0 <= start_index < stop_index < n):
        raise ValueError("study bounds exceed supplied historical data")

    if (
        np.asarray(signals.timestamps).shape != (n,)
        or np.asarray(context.timestamps).shape != (n,)
        or not np.array_equal(signals.timestamps, context.timestamps)
    ):
        raise ValueError("short opportunity context and signal clocks must align")
    time = signals.timestamps.astype("datetime64[ns]").astype(np.int64)
    if np.any(time % _NS_HOUR != 0) or (n > 1 and np.any(np.diff(time) != _NS_HOUR)):
        raise ValueError("regular UTC one-hour completion clock required")

    state = np.asarray(signals.cloud_state)
    cloud_ready = np.asarray(signals.cloud_ready)
    ready = np.asarray(context.ready)
    momentum = np.asarray(context.momentum_30d, dtype=np.float64)
    cloud_slope = np.asarray(context.cloud_mid_slope_7d, dtype=np.float64)
    annual_vol = np.asarray(context.realized_vol_30d, dtype=np.float64)
    if (
        any(
            a.shape != (n,)
            for a in (state, cloud_ready, ready, momentum, cloud_slope, annual_vol)
        )
        or cloud_ready.dtype != np.bool_
        or ready.dtype != np.bool_
        or np.any(~np.isin(state, (-1, 0, 1)))
    ):
        raise ValueError("short opportunity arrays must use valid aligned masks")
    if (
        not np.isfinite(momentum[ready]).all()
        or not np.isfinite(cloud_slope[ready]).all()
        or not np.isfinite(annual_vol[ready]).all()
        or np.any(annual_vol[ready] < 0)
    ):
        raise ValueError("ready context must contain finite valid causal values")

    rows: list[ShortOpportunityObservation] = []
    # Exclude observations lacking a complete 24h response by study end.
    # One daily timestamp per asset avoids mechanical overlap of 24h labels.
    for i in range(
        max(start_index, _ENTRY_CONFIRM_HOURS - 1), stop_index - _HORIZON_HOURS + 1
    ):
        if time[i] % _NS_DAY != 0 or not ready[i]:
            continue
        begin = i - _ENTRY_CONFIRM_HOURS + 1
        if not (
            np.all(cloud_ready[begin : i + 1]) and np.all(state[begin : i + 1] == -1)
        ):
            continue
        m = float(momentum[i])
        slope = float(cloud_slope[i])
        if m >= 0:
            group: ShortGroup = "G0_momentum_nonnegative"
        elif slope >= 0:
            group = "G1_momentum_negative_slope_nonnegative"
        else:
            group = "G2_both_negative"
        label = float(100 * (1 - c[i + _HORIZON_HOURS] / c[i]))
        if not math.isfinite(label):
            raise ValueError("offline 24h label became nonfinite")
        vol = float(annual_vol[i])
        normalized = (
            label / (100 * vol * math.sqrt(_HORIZON_HOURS / 8760)) if vol > 0 else None
        )
        if normalized is not None and not math.isfinite(normalized):
            raise ValueError("offline normalized label became nonfinite")
        rows.append(
            ShortOpportunityObservation(
                index=i,
                next_index=i + _HORIZON_HOURS,
                timestamp_ns=int(time[i]),
                group=group,
                signed_24h_return_pct=label,
                normalized_24h_short_move=normalized,
            )
        )
    return tuple(rows)


def summarize_short_opportunities(
    observations: tuple[ShortOpportunityObservation, ...],
) -> dict[ShortGroup, ShortOpportunityGroupStats]:
    """Complete, fixed-roster group summaries without dropping failures."""
    outputs: dict[ShortGroup, ShortOpportunityGroupStats] = {}
    for group in _SHORT_GROUPS:
        selected = [x for x in observations if x.group == group]
        moves = [x.signed_24h_return_pct for x in selected]
        normalized = [
            x.normalized_24h_short_move
            for x in selected
            if x.normalized_24h_short_move is not None
        ]
        count = len(moves)
        outputs[group] = ShortOpportunityGroupStats(
            count=count,
            mean_signed_24h_pct=math.fsum(moves) / count if count else None,
            median_signed_24h_pct=float(np.median(moves)) if count else None,
            favorable_fraction=sum(x > 0 for x in moves) / count if count else None,
            mean_vol_normalized_move=(
                math.fsum(normalized) / len(normalized) if normalized else None
            ),
        )
    return outputs


__all__ = [
    "ShortOpportunityObservation",
    "ShortOpportunityGroupStats",
    "audit_24h_short_opportunities",
    "summarize_short_opportunities",
]
