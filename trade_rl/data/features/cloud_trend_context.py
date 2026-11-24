"""As-of completed 4h ADX and ATR context for opt-in cloud research.

Canonical indicator numerics are reused. This module owns only availability,
units, and the hourly alignment; no market loading or portfolio decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.features.core import calculate_feature_events
from trade_rl.data.features.divergence_mtf import (
    _aggregate,
    _time_ns,
    _validate_native_hourly,
)
from trade_rl.data.source import RawMarketSeries, _readonly, _timestamp_ns_array

_PERIOD = 14
_HOUR_NS = 3_600_000_000_000


@dataclass(frozen=True, slots=True)
class CloudTrendContext:
    """Immutable per-hour context; both features are fractions, not 0–100 values."""

    timestamps: np.ndarray
    adx_fraction: np.ndarray
    atr_fraction: np.ndarray
    ready: np.ndarray

    def __post_init__(self) -> None:
        if np.asarray(self.timestamps).dtype.kind != "M":
            raise ValueError("context timestamps must use datetime64")
        time = _timestamp_ns_array(self.timestamps, field_name="timestamps")
        if time.ndim != 1 or not len(time):
            raise ValueError("context clock must be nonempty and one-dimensional")
        ns = time.astype(np.int64)
        if np.any(ns % _HOUR_NS) or np.any(np.diff(ns) != _HOUR_NS):
            raise ValueError("regular hourly context clock required")
        ready = np.asarray(self.ready)
        if ready.dtype != np.bool_ or ready.shape != time.shape:
            raise ValueError("context ready must be a matching boolean mask")
        adx = np.asarray(self.adx_fraction, dtype=np.float64)
        atr = np.asarray(self.atr_fraction, dtype=np.float64)
        if adx.shape != time.shape or atr.shape != time.shape:
            raise ValueError("context values must match timestamps")
        if (
            not np.isfinite(adx[ready]).all()
            or np.any((adx[ready] < 0) | (adx[ready] > 1))
            or not np.isfinite(atr[ready]).all()
            or np.any(atr[ready] <= 0)
            or not np.isnan(adx[~ready]).all()
            or not np.isnan(atr[~ready]).all()
        ):
            raise ValueError("invalid ADX/ATR values or readiness mask")
        for name, value in (
            ("timestamps", time),
            ("adx_fraction", adx),
            ("atr_fraction", atr),
            ("ready", ready),
        ):
            object.__setattr__(self, name, _readonly(value))


def build_cloud_trend_context(raw: RawMarketSeries) -> CloudTrendContext:
    """Align only the latest completed and available 4h feature pair at each hour."""
    _validate_native_hourly(raw)
    stamps = _time_ns(raw)
    adx = np.full(len(stamps), np.nan, dtype=np.float64)
    atr = np.full(len(stamps), np.nan, dtype=np.float64)
    ready = np.zeros(len(stamps), dtype=np.bool_)
    four = _aggregate(raw, 4)
    if four is None:
        return CloudTrendContext(raw.timestamps, adx, atr, ready)
    assert four.funding_available is not None
    values = []
    for kind in (FeatureKind.ADX, FeatureKind.ATR_PCT):
        values.append(
            calculate_feature_events(
                FeatureSpec(name=kind.value, kind=kind, lookback=_PERIOD),
                open_price=four.open,
                high=four.high,
                low=four.low,
                close=four.close,
                volume=four.volume,
                funding_rate=four.funding_rate,
                funding_available=four.funding_available,
                row_present=np.ones(len(four.close), dtype=np.bool_),
                active=four.tradable,
            )
        )
    adx_native, adx_valid, adx_start = values[0]
    atr_native, atr_valid, _ = values[1]
    # Canonical ADX requires 2*period observations in a contiguous segment,
    # although its first seeded DX average can be written one row earlier.
    # Suppress that row so adding a future bar cannot enable an earlier value.
    valid = (
        adx_valid
        & atr_valid
        & (np.arange(len(four.close)) - adx_start >= 2 * _PERIOD - 1)
    )
    if not (
        np.isfinite(adx_native[valid]).all() and np.isfinite(atr_native[valid]).all()
    ):
        raise ValueError("nonfinite canonical ADX/ATR")
    # Permit only rounding-scale excursion beyond the theoretical ADX range.
    if np.any((adx_native[valid] < -1e-12) | (adx_native[valid] > 1 + 1e-12)):
        raise ValueError("canonical ADX outside its fraction range")
    valid &= atr_native > 0
    assert raw.available_at is not None and four.available_at is not None
    native_time = _time_ns(four)
    known = np.maximum.accumulate(four.available_at.astype(np.int64))
    latest = np.searchsorted(native_time, stamps, side="right") - 1
    for i, j in enumerate(latest):
        if (
            j < 0
            or not raw.tradable[i]
            or raw.available_at[i] > raw.timestamps[i]
            or known[j] > stamps[i]
            or stamps[i] - native_time[j] >= 4 * _HOUR_NS
            or not valid[j]
        ):
            continue
        adx[i] = np.clip(adx_native[j], 0.0, 1.0)
        atr[i] = atr_native[j]
        ready[i] = True
    return CloudTrendContext(raw.timestamps, adx, atr, ready)


__all__ = ["CloudTrendContext", "build_cloud_trend_context"]
