"""1h/4h/1d divergence-and-cloud signal pack on a causal 1h decision clock.

All candles are native completed bars. The 4h/1d aggregation closes on UTC
boundaries. The current visible cloud uses the most recent *available* daily
bar, and events appear once at or after the publishing availability timestamp.
An incomplete or irregular native 1h stream is refused rather than padded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from trade_rl.data.features.divergence import (
    DivergenceConfig,
    DivergenceEvent,
    detect_divergences,
)
from trade_rl.data.features.indicators.divergence_bank import build_divergence_bank
from trade_rl.data.features.indicators.ichimoku_cloud import visible_ichimoku_cloud
from trade_rl.data.source import RawMarketSeries

_NS_HOUR = 3_600_000_000_000


@dataclass(frozen=True, slots=True)
class DivergenceCloudMtfPack:
    timestamps: np.ndarray
    bull_1h: np.ndarray
    bear_1h: np.ndarray
    bull_4h: np.ndarray
    bear_4h: np.ndarray
    cloud_state: np.ndarray  # 1 = above, -1 = below, 0 = inside/unknown
    cloud_ready: np.ndarray
    events_1h: tuple[DivergenceEvent, ...]
    events_4h: tuple[DivergenceEvent, ...]


def _time_ns(raw: RawMarketSeries) -> np.ndarray:
    return raw.timestamps.astype("datetime64[ns]").astype(np.int64)


def _validate_native_hourly(raw: RawMarketSeries) -> None:
    stamps = _time_ns(raw)
    if len(stamps) > 1 and not np.all(np.diff(stamps) == _NS_HOUR):
        raise ValueError("native 1h bars must be regular and gap-free")
    if np.any(stamps % _NS_HOUR != 0):
        raise ValueError("native 1h bars must close on UTC whole-hour boundaries")


def _aggregate(raw: RawMarketSeries, hours: int) -> RawMarketSeries | None:
    stamp = _time_ns(raw)
    ends = np.flatnonzero(stamp % (hours * _NS_HOUR) == 0)
    ends = ends[ends >= hours - 1]
    if not len(ends):
        return None
    rows = [slice(int(t) - hours + 1, int(t) + 1) for t in ends]
    v = np.asarray([float(np.sum(raw.volume[s], dtype=np.float64)) for s in rows])
    if not np.isfinite(v).all():
        raise ValueError("aggregated volume overflow")
    assert raw.available_at is not None
    return RawMarketSeries(
        timestamps=raw.timestamps[ends],
        available_at=np.asarray([raw.available_at[s].max() for s in rows]),
        open=np.asarray([raw.open[s.start] for s in rows]),
        high=np.asarray([raw.high[s].max() for s in rows]),
        low=np.asarray([raw.low[s].min() for s in rows]),
        close=raw.close[ends],
        volume=v,
        funding_rate=np.zeros(len(ends), dtype=np.float64),
        tradable=np.asarray([raw.tradable[s].all() for s in rows]),
    )


def _aligned_events(
    raw: RawMarketSeries,
    base_ns: np.ndarray,
    base_tradable: np.ndarray,
    *,
    config: DivergenceConfig,
    indicators: tuple[str, ...],
    timeframe_hours: int,
) -> tuple[np.ndarray, np.ndarray, tuple[DivergenceEvent, ...]]:
    bank = build_divergence_bank(raw, names=indicators)
    signals = detect_divergences(
        high=raw.high,
        low=raw.low,
        close=raw.close,
        indicators=bank,
        config=config,
    )
    bull = np.zeros(len(base_ns), dtype=np.int16)
    bear = np.zeros(len(base_ns), dtype=np.int16)
    assert raw.available_at is not None
    stamps = _time_ns(raw)
    available_ns = raw.available_at.astype("datetime64[ns]").astype(np.int64)
    # A signal uses every historical pivot/indicator sample up through its
    # observation bar. One older bar can publish later than the current bar.
    prefix_known_at = np.maximum.accumulate(available_ns)
    for j in np.flatnonzero(signals.bull_total + signals.bear_total):
        effective_at = int(prefix_known_at[j])
        # A stale late packet must not be mistaken for a fresh entry signal.
        if effective_at - stamps[j] >= timeframe_hours * _NS_HOUR:
            continue
        decision = int(np.searchsorted(base_ns, effective_at))
        if decision >= len(base_ns) or not base_tradable[decision]:
            continue
        bull[decision] += signals.bull_total[j]
        bear[decision] += signals.bear_total[j]
    return bull, bear, signals.events


def build_divergence_cloud_mtf(
    raw_1h: RawMarketSeries,
    *,
    indicators: Iterable[str] = ("RSI", "MACD_HIST", "MFI"),
    config: DivergenceConfig | None = None,
) -> DivergenceCloudMtfPack:
    """Generate per-hour causal research signals without changing Dataset schema."""
    _validate_native_hourly(raw_1h)
    cfg = config or DivergenceConfig()
    chosen = tuple(indicators)
    base_ns = _time_ns(raw_1h)
    assert raw_1h.available_at is not None
    observed_at_ns = raw_1h.available_at.astype("datetime64[ns]").astype(np.int64)
    base_usable = raw_1h.tradable & (observed_at_ns <= base_ns)
    bull1, bear1, events1 = _aligned_events(
        raw_1h,
        base_ns,
        base_usable,
        config=cfg,
        indicators=chosen,
        timeframe_hours=1,
    )
    four = _aggregate(raw_1h, 4)
    daily = _aggregate(raw_1h, 24)
    bull4 = np.zeros(len(base_ns), dtype=np.int16)
    bear4 = np.zeros(len(base_ns), dtype=np.int16)
    events4: tuple[DivergenceEvent, ...] = ()
    if four is not None:
        bull4, bear4, events4 = _aligned_events(
            four,
            base_ns,
            base_usable,
            config=cfg,
            indicators=chosen,
            timeframe_hours=4,
        )
    state = np.zeros(len(base_ns), dtype=np.int8)
    ready = np.zeros(len(base_ns), dtype=bool)
    if daily is not None:
        cloud = visible_ichimoku_cloud(daily)
        assert daily.available_at is not None
        available = daily.available_at.astype("datetime64[ns]").astype(np.int64)
        # Cloud spans depend on an entire preceding 78-day native window,
        # not merely on the most recent daily close.
        prefix_known_at = np.maximum.accumulate(available)
        newest = -1
        ordered = sorted(
            range(len(available)), key=lambda idx: (prefix_known_at[idx], idx)
        )
        pointer = 0
        for i, t in enumerate(base_ns):
            while pointer < len(ordered) and prefix_known_at[ordered[pointer]] <= t:
                newest = max(newest, ordered[pointer])
                pointer += 1
            if newest < 0 or not base_usable[i] or not cloud.ready[newest]:
                continue
            # Stale daily data older than 24h cannot authorize new risk.
            if t - int(_time_ns(daily)[newest]) >= 24 * _NS_HOUR:
                continue
            ready[i] = True
            if raw_1h.close[i] > cloud.top[newest]:
                state[i] = 1
            elif raw_1h.close[i] < cloud.bottom[newest]:
                state[i] = -1

    for arr in (bull1, bear1, bull4, bear4, state, ready):
        arr.setflags(write=False)
    return DivergenceCloudMtfPack(
        timestamps=raw_1h.timestamps,
        bull_1h=bull1,
        bear_1h=bear1,
        bull_4h=bull4,
        bear_4h=bear4,
        cloud_state=state,
        cloud_ready=ready,
        events_1h=events1,
        events_4h=events4,
    )


__all__ = ["DivergenceCloudMtfPack", "build_divergence_cloud_mtf"]
