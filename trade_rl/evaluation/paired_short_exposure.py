"""Read-only overlap/disagreement of actual short-position hours on used data.

Read the canonical shared account's exact *pre-interval* quantities. Future
traded-close price moves are OFFLINE directional labels only, never tradable
signals, fill prices or replacement account P&L. No exposure-count matching,
market-based model fitting, or synthetic reconstructed book is performed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from trade_rl.evaluation.divergence_cloud_diagnostics import (
    LedgerInterval,
    attribute_single_symbol_ledger,
)
from trade_rl.simulation.quantities import parse_quantity

GroupName = Literal["both", "r2_only", "control_only", "neither"]
_GROUPS: tuple[GroupName, ...] = ("both", "r2_only", "control_only", "neither")
_HOUR_NS = 3_600_000_000_000


@dataclass(frozen=True, slots=True)
class ShortHourGroupStats:
    count: int
    mean_signed_market_pct: float | None
    median_signed_market_pct: float | None
    fraction_favorable: float | None


@dataclass(frozen=True, slots=True)
class PairedShortHourAudit:
    total_hour_count: int
    r2_short_hour_count: int
    control_short_hour_count: int
    groups: dict[GroupName, ShortHourGroupStats]
    r2_only_minus_control_only_mean_pct: float | None


def _is_short(raw: str) -> bool:
    return parse_quantity(raw) < 0


def audit_paired_short_hours(
    *,
    close: np.ndarray,
    timestamps: np.ndarray,
    r2_intervals: Sequence[LedgerInterval],
    control_intervals: Sequence[LedgerInterval],
    start_index: int,
    stop_index: int,
) -> PairedShortHourAudit:
    """Partition all completed [t,t+1] intervals by observed short status.

    Both strategies' *decisions and fills* come from canonical native replay;
    only the contemporaneous next-hour directional **market** label is
    computed here, after that hour completes, for descriptive analysis.
    """
    if type(start_index) is not int or type(stop_index) is not int:
        raise ValueError("study interval bounds must be native integer indexes")
    c = np.asarray(close, dtype=np.float64)
    time = np.asarray(timestamps)
    if c.ndim != 1 or time.shape != c.shape:
        raise ValueError("one-dimensional price and timestamp clocks must align")
    if not np.issubdtype(time.dtype, np.datetime64) or np.isnat(time).any():
        raise ValueError("timestamps must be valid completed UTC datetime64 hours")
    if not (0 <= start_index < stop_index < len(c)):
        raise ValueError("study interval bounds outside available price history")
    window = c[start_index : stop_index + 1]
    if not np.isfinite(window).all() or np.any(window <= 0):
        raise ValueError("selected historical prices must be positive finite")
    native_time = time.astype("datetime64[ns]").astype(np.int64)
    if np.any(native_time[start_index : stop_index + 1] % _HOUR_NS != 0) or np.any(
        np.diff(native_time[start_index : stop_index + 1]) != _HOUR_NS
    ):
        raise ValueError("regular UTC 1h completed candle clock is required")

    expected = stop_index - start_index
    if len(r2_intervals) != expected or len(control_intervals) != expected:
        raise ValueError("native ledger interval lengths do not match")
    # Reuse the already-tested native-ledger chain validator; this only
    # validates native equity/quantity continuity and never recalculates it.
    r2_account = attribute_single_symbol_ledger(r2_intervals)
    ctrl_account = attribute_single_symbol_ledger(control_intervals)
    if r2_account.reversal_intervals or ctrl_account.reversal_intervals:
        raise ValueError("direct reversal inside a native interval is ambiguous")

    labels: dict[GroupName, list[float]] = {key: [] for key in _GROUPS}
    r2_hours = 0
    ctrl_hours = 0
    for j, (a, b) in enumerate(zip(r2_intervals, control_intervals, strict=True)):
        i = start_index + j
        if (
            a.start_index != i
            or a.next_index != i + 1
            or b.start_index != i
            or b.next_index != i + 1
        ):
            raise ValueError("native ledger interval indexes are not the study clock")
        in_r2 = _is_short(a.exact_quantities_before[0])
        in_control = _is_short(b.exact_quantities_before[0])
        if in_r2 and in_control:
            kind: GroupName = "both"
        elif in_r2:
            kind = "r2_only"
        elif in_control:
            kind = "control_only"
        else:
            kind = "neither"
        r2_hours += int(in_r2)
        ctrl_hours += int(in_control)
        label = float(100 * (1 - c[i + 1] / c[i]))
        if not math.isfinite(label):
            raise ValueError("nonfinite post-hour directional market label")
        labels[kind].append(label)

    groups: dict[GroupName, ShortHourGroupStats] = {}
    for name in _GROUPS:
        series = labels[name]
        count = len(series)
        groups[name] = ShortHourGroupStats(
            count=count,
            mean_signed_market_pct=math.fsum(series) / count if count else None,
            median_signed_market_pct=float(np.median(series)) if count else None,
            fraction_favorable=sum(v > 0 for v in series) / count if count else None,
        )
    if sum(s.count for s in groups.values()) != expected:
        raise ValueError("paired group accounting does not cover study hours")
    if groups["both"].count + groups["r2_only"].count != r2_hours:
        raise ValueError("R2 short-hour count does not reconcile")
    if groups["both"].count + groups["control_only"].count != ctrl_hours:
        raise ValueError("control short-hour count does not reconcile")
    left = groups["r2_only"].mean_signed_market_pct
    right = groups["control_only"].mean_signed_market_pct
    return PairedShortHourAudit(
        total_hour_count=expected,
        r2_short_hour_count=r2_hours,
        control_short_hour_count=ctrl_hours,
        groups=groups,
        r2_only_minus_control_only_mean_pct=(
            left - right if left is not None and right is not None else None
        ),
    )


__all__ = ["PairedShortHourAudit", "ShortHourGroupStats", "audit_paired_short_hours"]
