"""Read-only, same-source feature adapter for the cloud regime switch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.features.cloud_trend_context import build_cloud_trend_context
from trade_rl.data.features.core import calculate_feature_events
from trade_rl.data.features.divergence_mtf import build_divergence_cloud_mtf
from trade_rl.data.features.price_channels import CHANNEL_NAMES, with_price_channels
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import RawMarketSeries, _readonly, _timestamp_ns_array

_HOUR_NS = 3_600_000_000_000


@dataclass(frozen=True, slots=True)
class CloudRegimeSwitchContext:
    """Immutable hourly inputs; no economic result or execution ownership."""

    symbol: str
    timestamps: np.ndarray
    ready: np.ndarray
    adx_fraction: np.ndarray
    atr_fraction: np.ndarray
    rsi_centered: np.ndarray
    bollinger_position: np.ndarray
    cloud_state: np.ndarray
    entry_breakout: np.ndarray
    exit_breakout: np.ndarray
    bull_1h: np.ndarray
    bear_1h: np.ndarray

    def __post_init__(self) -> None:
        if type(self.symbol) is not str or not self.symbol.strip():
            raise ValueError("symbol must be nonempty")
        if np.asarray(self.timestamps).dtype.kind != "M":
            raise ValueError("timestamps must use datetime64")
        t = _timestamp_ns_array(self.timestamps, field_name="timestamps")
        if t.ndim != 1 or not len(t):
            raise ValueError("context requires a nonempty hourly clock")
        ns = t.astype(np.int64)
        if np.any(ns % _HOUR_NS) or np.any(np.diff(ns) != _HOUR_NS):
            raise ValueError("regular hourly context clock required")
        ready = np.asarray(self.ready)
        if ready.shape != t.shape or ready.dtype != np.bool_:
            raise ValueError("ready must be a matching boolean mask")
        object.__setattr__(self, "timestamps", t)
        object.__setattr__(self, "ready", _readonly(ready))
        for name in (
            "adx_fraction",
            "atr_fraction",
            "rsi_centered",
            "bollinger_position",
        ):
            a = np.asarray(getattr(self, name), dtype=np.float64)
            if (
                a.shape != t.shape
                or not np.isfinite(a[ready]).all()
                or not np.isnan(a[~ready]).all()
            ):
                raise ValueError(f"invalid {name} readiness or value")
            if name == "adx_fraction" and np.any((a[ready] < 0) | (a[ready] > 1)):
                raise ValueError("ADX must be a fraction")
            if name == "atr_fraction" and np.any(a[ready] <= 0):
                raise ValueError("ATR fraction must be positive")
            if name == "rsi_centered" and np.any(np.abs(a[ready]) > 1):
                raise ValueError("RSI must be centered within [-1,1]")
            object.__setattr__(self, name, _readonly(a))
        for name in (
            "cloud_state",
            "entry_breakout",
            "exit_breakout",
            "bull_1h",
            "bear_1h",
        ):
            a = np.asarray(getattr(self, name))
            if a.shape != t.shape or a.dtype.kind not in "iu":
                raise ValueError(f"invalid {name} shape or type")
            if name in ("bull_1h", "bear_1h"):
                if np.any(a < 0):
                    raise ValueError("divergence counts must be nonnegative")
            elif np.any(~np.isin(a, (-1, 0, 1))):
                raise ValueError("invalid directional state")
            object.__setattr__(self, name, _readonly(a))


def build_cloud_regime_switch_context(
    dataset: MarketDataset,
) -> CloudRegimeSwitchContext:
    """Derive every input from the same completed one-symbol source dataset.

    Exclude unavailable-at-close rows conservatively, including from indicator
    recurrence. The channel-augmented Dataset is temporary, never persisted.
    """
    if not isinstance(dataset, MarketDataset) or dataset.n_symbols != 1:
        raise ValueError("a one-symbol MarketDataset is required")
    if dataset.bar_hours != 1.0:
        raise ValueError("native hourly source required")
    t = dataset.timestamps
    usable = (
        dataset.tradable
        & dataset.resolved_array("symbol_active")
        & dataset.resolved_array("information_available")
        & (dataset.resolved_array("available_at") <= t[:, None])
    )[:, 0]
    raw = RawMarketSeries(
        timestamps=t,
        open=dataset.open[:, 0],
        high=dataset.high[:, 0],
        low=dataset.low[:, 0],
        close=dataset.close[:, 0],
        volume=dataset.volume[:, 0],
        funding_rate=np.zeros(dataset.n_bars),
        funding_available=np.zeros(dataset.n_bars, dtype=bool),
        available_at=dataset.resolved_array("available_at")[:, 0],
        tradable=usable,
    )
    trend = build_cloud_trend_context(raw)
    cloud = build_divergence_cloud_mtf(raw)
    assert raw.funding_available is not None
    rsi, rsi_ok, _ = calculate_feature_events(
        FeatureSpec(name="switch_rsi", kind=FeatureKind.RSI, lookback=14),
        open_price=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
        funding_rate=raw.funding_rate,
        funding_available=raw.funding_available,
        row_present=np.ones(dataset.n_bars, dtype=bool),
        active=usable,
    )
    bb, bb_ok, _ = calculate_feature_events(
        FeatureSpec(name="switch_bb", kind=FeatureKind.BOLLINGER_POSITION, lookback=20),
        open_price=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
        funding_rate=raw.funding_rate,
        funding_available=raw.funding_available,
        row_present=np.ones(dataset.n_bars, dtype=bool),
        active=usable,
    )
    entry = np.zeros(dataset.n_bars, dtype=np.int8)
    exit_ = np.zeros_like(entry)
    channel_ok = np.zeros(dataset.n_bars, dtype=bool)
    if dataset.n_bars > 20:
        augmented = with_price_channels(dataset, entry_bars=20, exit_bars=10)
        indexes = [augmented.feature_names.index(name) for name in CHANNEL_NAMES]
        values = augmented.features[:, 0, indexes]
        channel_ok = augmented.feature_available[:, 0, indexes].all(axis=1)
        entry = np.where(values[:, 0] > 0, 1, np.where(values[:, 1] < 0, -1, 0)).astype(
            np.int8
        )
        exit_ = np.where(values[:, 2] > 0, 1, np.where(values[:, 3] < 0, -1, 0)).astype(
            np.int8
        )
    # Also exclude any unresolved earlier arrival, even after a local warmup.
    assert raw.available_at is not None
    prefix_known = np.maximum.accumulate(raw.available_at.astype(np.int64))
    ready = (
        usable
        & trend.ready
        & cloud.cloud_ready
        & rsi_ok
        & bb_ok
        & channel_ok
        & (prefix_known <= t.astype(np.int64))
    )
    return CloudRegimeSwitchContext(
        symbol=dataset.symbols[0],
        timestamps=t,
        ready=ready,
        adx_fraction=np.where(ready, trend.adx_fraction, np.nan),
        atr_fraction=np.where(ready, trend.atr_fraction, np.nan),
        rsi_centered=np.where(ready, rsi, np.nan),
        bollinger_position=np.where(ready, bb, np.nan),
        cloud_state=cloud.cloud_state,
        entry_breakout=entry,
        exit_breakout=exit_,
        bull_1h=cloud.bull_1h,
        bear_1h=cloud.bear_1h,
    )


__all__ = ["CloudRegimeSwitchContext", "build_cloud_regime_switch_context"]
