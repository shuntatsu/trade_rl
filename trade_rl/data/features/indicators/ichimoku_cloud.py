"""Visible Ichimoku cloud calculated without chart-only future displacement.

A span computed at source bar s is conventionally *drawn* at s+26. At decision
bar t the visible cloud therefore comes from s=t-26, never from a later bar.
The existing core cloud-position feature instead uses unshifted spans for
decision-time analysis; this opt-in series intentionally has different semantics.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from trade_rl.data.contracts import InstrumentContract
from trade_rl.data.features.multitimeframe import (
    _align_event_values,
    _clocked_lifetime,
    _complete_lifetime_mask,
    _decision_grid,
    _dependency_available_at,
)
from trade_rl.data.native_clock import NativeBarClock
from trade_rl.data.source import NativeRawMarketSeries, RawMarketSeries


@dataclass(frozen=True, slots=True)
class VisibleCloud:
    top: np.ndarray
    bottom: np.ndarray
    span_a: np.ndarray
    span_b: np.ndarray
    ready: np.ndarray


def _midpoint(upper: float, lower: float) -> float:
    total = upper + lower
    return 0.5 * total if math.isfinite(total) else 0.5 * upper + 0.5 * lower


def _rolling_midpoints(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Compute trailing midpoints in amortized linear time."""
    result = np.full(len(high), np.nan)
    upper: deque[tuple[int, float]] = deque()
    lower: deque[tuple[int, float]] = deque()
    for i, (hi, lo) in enumerate(zip(high, low, strict=True)):
        high_value, low_value = float(hi), float(lo)
        while upper and upper[0][0] <= i - period:
            upper.popleft()
        while lower and lower[0][0] <= i - period:
            lower.popleft()
        while upper and upper[-1][1] <= high_value:
            upper.pop()
        while lower and lower[-1][1] >= low_value:
            lower.pop()
        upper.append((i, high_value))
        lower.append((i, low_value))
        if i >= period - 1:
            result[i] = _midpoint(upper[0][1], lower[0][1])
    return result


def visible_ichimoku_cloud(
    raw: RawMarketSeries,
    *,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> VisibleCloud:
    """Compute the visible top/bottom from past native bars in linear time.

    Readiness describes completed native-row arithmetic, not publication
    availability or grid completeness. Those remain caller-owned boundaries.
    """
    return _visible_cloud(
        raw,
        raw.tradable,
        tenkan_period=tenkan_period,
        kijun_period=kijun_period,
        span_b_period=span_b_period,
        displacement=displacement,
    )


def _visible_cloud(
    raw: RawMarketSeries,
    tradable: np.ndarray,
    *,
    tenkan_period: int,
    kijun_period: int,
    span_b_period: int,
    displacement: int,
) -> VisibleCloud:
    periods = (tenkan_period, kijun_period, span_b_period, displacement)
    if any(type(value) is not int or value < 1 for value in periods):
        raise ValueError("cloud periods and displacement must be positive integers")
    n = len(raw.close)
    a = np.full(n, np.nan)
    b = np.full(n, np.nan)
    ready = np.zeros(n, dtype=np.bool_)
    dependency_period = max(tenkan_period, kijun_period, span_b_period)
    first_visible = dependency_period - 1 + displacement
    if first_visible < n:
        # Each unique trailing window is computed once. The displacement
        # never requires the price suffix beyond the final source row.
        source_count = n - displacement
        midpoints = {
            period: _rolling_midpoints(
                raw.high[:source_count], raw.low[:source_count], period
            )
            for period in {tenkan_period, kijun_period, span_b_period}
        }
        # Prefix counts preserve the complete tradable window through the
        # visible row, including the displacement interval, without rescans.
        blocked = np.empty(n + 1, dtype=np.int64)
        blocked[0] = 0
        np.cumsum(~tradable, dtype=np.int64, out=blocked[1:])
        for t in range(first_visible, n):
            s = t - displacement
            if blocked[t + 1] != blocked[s - dependency_period + 1]:
                continue
            a[t] = _midpoint(
                float(midpoints[tenkan_period][s]), float(midpoints[kijun_period][s])
            )
            b[t] = midpoints[span_b_period][s]
            ready[t] = True
    top = np.maximum(a, b)
    bottom = np.minimum(a, b)
    for array in (a, b, top, bottom, ready):
        array.setflags(write=False)
    return VisibleCloud(top=top, bottom=bottom, span_a=a, span_b=b, ready=ready)


@dataclass(frozen=True, slots=True)
class VisibleCloudEvents:
    """Native close-indexed cloud prices, validity and full-window receipt times."""

    top: np.ndarray
    bottom: np.ndarray
    span_a: np.ndarray
    span_b: np.ndarray
    valid: np.ndarray
    available_at: np.ndarray


@dataclass(frozen=True, slots=True)
class AlignedVisibleCloud:
    """Cloud values on decision timestamps; unavailable prices are NaN."""

    top: np.ndarray
    bottom: np.ndarray
    span_a: np.ndarray
    span_b: np.ndarray
    available: np.ndarray
    age_hours: np.ndarray
    staleness: np.ndarray


def visible_ichimoku_events(
    raw: NativeRawMarketSeries,
    *,
    clock: NativeBarClock,
    contract: InstrumentContract,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> VisibleCloudEvents:
    """Apply the native grid, lifetime and complete dependency publication contract.

    Periods and displacement count native bars. For visible row t, the
    dependency window begins at t-displacement-max(periods)+1 and extends
    through t, including tradability and the latest declared publication.
    Invalid events have NaN prices and NaT availability. This neither
    authenticates the input source nor changes any Dataset feature identity.
    """
    if type(raw) is not NativeRawMarketSeries:
        raise ValueError("visible cloud requires an explicit NativeRawMarketSeries")
    if type(clock) is not NativeBarClock or type(contract) is not InstrumentContract:
        raise ValueError(
            "visible cloud requires a native clock and instrument contract"
        )
    starts, closes = clock.interval_bounds(raw.timestamps)
    listed, delisted = _clocked_lifetime(contract)
    active = _complete_lifetime_mask(starts, closes, listed, delisted)
    cloud = _visible_cloud(
        raw,
        raw.tradable & active,
        tenkan_period=tenkan_period,
        kijun_period=kijun_period,
        span_b_period=span_b_period,
        displacement=displacement,
    )
    # Only subtract the width for valid indices: arbitrarily long legal
    # positive periods can exceed int64 when the input has no ready events.
    valid_indices = np.flatnonzero(cloud.ready)
    source_start = np.full(len(raw.timestamps), -1, dtype=np.int64)
    if valid_indices.size:
        width = max(tenkan_period, kijun_period, span_b_period) + displacement - 1
        source_start[valid_indices] = valid_indices - width
    assert raw.available_at is not None
    available_at = _dependency_available_at(raw.available_at, cloud.ready, source_start)
    available_at.setflags(write=False)
    return VisibleCloudEvents(
        cloud.top, cloud.bottom, cloud.span_a, cloud.span_b, cloud.ready, available_at
    )


def align_visible_ichimoku_cloud(
    raw: NativeRawMarketSeries,
    *,
    clock: NativeBarClock,
    contract: InstrumentContract,
    base_timestamps: np.ndarray,
    base_active: np.ndarray,
    max_staleness_hours: float,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> AlignedVisibleCloud:
    """Calculate once, then align both visible spans to one causal frontier.

    Staleness is elapsed hours from the visible native close, not receipt
    time or the displaced source close. Never extrapolate or fill an
    unavailable cloud price; output NaN with an explicit false mask instead.
    The caller owns raw-source identity, provider finality and permission.
    """
    if type(max_staleness_hours) not in (int, float):
        raise ValueError("max_staleness_hours must be finite and positive")
    try:
        finite = math.isfinite(max_staleness_hours)
    except OverflowError:
        finite = False
    if not finite or max_staleness_hours <= 0:
        raise ValueError("max_staleness_hours must be finite and positive")
    base, active = _decision_grid(base_timestamps, base_active)
    events = visible_ichimoku_events(
        raw,
        clock=clock,
        contract=contract,
        tenkan_period=tenkan_period,
        kijun_period=kijun_period,
        span_b_period=span_b_period,
        displacement=displacement,
    )
    listed, delisted = _clocked_lifetime(contract)
    decision_active = active & (base >= listed)
    if delisted is not None:
        decision_active &= base < delisted
    spans, available, age, stale = _align_event_values(
        float(max_staleness_hours),
        raw,
        (
            np.column_stack((events.span_a, events.span_b)),
            events.valid,
            events.available_at,
        ),
        base,
        decision_active,
    )
    spans[~available] = np.nan
    spans.setflags(write=False)
    a, b = spans[:, 0], spans[:, 1]
    top, bottom = np.maximum(a, b), np.minimum(a, b)
    for array in (top, bottom, a, b, available, age, stale):
        array.setflags(write=False)
    return AlignedVisibleCloud(top, bottom, a, b, available, age, stale)


__all__ = [
    "AlignedVisibleCloud",
    "VisibleCloud",
    "VisibleCloudEvents",
    "align_visible_ichimoku_cloud",
    "visible_ichimoku_cloud",
    "visible_ichimoku_events",
]
