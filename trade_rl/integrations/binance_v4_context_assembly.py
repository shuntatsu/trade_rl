"""Decision-clock assembly for Causal Alpha V4 Binance context sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import numpy as np

from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.data.v4_context import V4CrossMarketInputs
from trade_rl.integrations.binance_v4_context import (
    AlignedFuturesMetrics,
    BinanceFundingEventSeries,
    BinanceV4KlineSeries,
)

_BINANCE_VISION_ROOT: Final = "https://data.binance.vision/data/"
_ALIGNED_KLINE_SCHEMA: Final = "binance_v4_aligned_kline_v1"
_ALIGNED_FUNDING_SCHEMA: Final = "binance_v4_aligned_funding_events_v1"
_ASSEMBLED_SOURCE_SCHEMA: Final = "binance_v4_cross_market_inputs_source_v1"


def _require_sha256(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lower-case SHA-256 digest")
    return value


def _decision_ns(value: object) -> np.ndarray:
    result = np.asarray(value, dtype="datetime64[ns]").reshape(-1)
    if result.size == 0 or np.any(np.isnat(result)):
        raise ValueError("V4 decision timestamps must be non-empty and finite")
    raw = result.astype(np.int64)
    if raw.size > 1 and np.any(raw[1:] <= raw[:-1]):
        raise ValueError("V4 decision timestamps must be strictly increasing")
    return raw


@dataclass(frozen=True, slots=True)
class AlignedV4KlineSeries:
    close: np.ndarray
    quote_volume: np.ndarray
    taker_buy_quote_volume: np.ndarray
    available: np.ndarray
    source_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for field, value, dtype in (
            ("close", self.close, np.float64),
            ("quote_volume", self.quote_volume, np.float64),
            ("taker_buy_quote_volume", self.taker_buy_quote_volume, np.float64),
            ("available", self.available, np.bool_),
        ):
            array = np.asarray(value, dtype=dtype).reshape(-1).copy(order="C")
            if array.size == 0:
                raise ValueError(f"V4 aligned kline {field} must not be empty")
            if dtype is not np.bool_ and not np.isfinite(array).all():
                raise ValueError(f"V4 aligned kline {field} must be finite")
            array.setflags(write=False)
            arrays[field] = array
        size = arrays["available"].size
        if any(array.shape != (size,) for array in arrays.values()):
            raise ValueError("V4 aligned kline arrays must be row aligned")
        if np.any(arrays["close"][arrays["available"]] <= 0.0):
            raise ValueError("V4 aligned kline available close must be positive")
        if np.any(arrays["quote_volume"] < 0.0) or np.any(
            arrays["taker_buy_quote_volume"] < 0.0
        ):
            raise ValueError("V4 aligned kline volumes must be non-negative")
        if np.any(
            arrays["taker_buy_quote_volume"]
            > arrays["quote_volume"] + np.finfo(np.float64).eps
        ):
            raise ValueError("V4 aligned kline taker buy quote exceeds quote volume")
        source = _require_sha256(
            self.source_digest, field="V4 aligned kline source_digest"
        )
        for field, array in arrays.items():
            object.__setattr__(self, field, array)
        object.__setattr__(self, "source_digest", source)
        expected = content_and_arrays_digest(
            {
                "schema_version": _ALIGNED_KLINE_SCHEMA,
                "source_digest": source,
            },
            tuple((field, array) for field, array in arrays.items()),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 aligned kline digest mismatch")
        object.__setattr__(self, "digest", expected)


def vision_mark_price_kline_url(symbol: str, interval: str, day: datetime) -> str:
    """Return one immutable Binance Vision USD-M mark-price kline URL."""

    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("mark-price symbol must be non-empty")
    if not isinstance(interval, str) or not interval.strip():
        raise ValueError("mark-price interval must be non-empty")
    if day.tzinfo is None or day.utcoffset() is None:
        raise ValueError("mark-price day must be timezone-aware")
    resolved_symbol = symbol.strip().upper()
    resolved_interval = interval.strip()
    date = day.astimezone(UTC).strftime("%Y-%m-%d")
    return (
        f"{_BINANCE_VISION_ROOT}futures/um/daily/markPriceKlines/"
        f"{resolved_symbol}/{resolved_interval}/"
        f"{resolved_symbol}-{resolved_interval}-{date}.zip"
    )


def align_v4_kline_to_decisions(
    decision_timestamps: object,
    series: BinanceV4KlineSeries,
    *,
    interval_minutes: int,
) -> AlignedV4KlineSeries:
    """Align only the exact fully closed source bar to each decision boundary."""

    if (
        isinstance(interval_minutes, bool)
        or not isinstance(interval_minutes, int)
        or interval_minutes <= 0
    ):
        raise ValueError("interval_minutes must be a positive integer")
    if not isinstance(series, BinanceV4KlineSeries):
        raise TypeError("series must be BinanceV4KlineSeries")
    decisions_ns = _decision_ns(decision_timestamps)
    interval_ms = interval_minutes * 60 * 1_000
    decision_ms = decisions_ns // 1_000_000
    required_open_ms = decision_ms - interval_ms

    close = np.zeros(decision_ms.shape, dtype=np.float64)
    quote = np.zeros(decision_ms.shape, dtype=np.float64)
    taker = np.zeros(decision_ms.shape, dtype=np.float64)
    available = np.zeros(decision_ms.shape, dtype=np.bool_)

    source_positions = {
        int(open_time): index for index, open_time in enumerate(series.open_time_ms)
    }
    for row, expected_open in enumerate(required_open_ms):
        source_index = source_positions.get(int(expected_open))
        if source_index is None:
            continue
        if int(series.close_time_ms[source_index]) > int(decision_ms[row]):
            continue
        close[row] = series.close[source_index]
        quote[row] = series.quote_volume[source_index]
        taker[row] = series.taker_buy_quote_volume[source_index]
        available[row] = True

    digest = content_and_arrays_digest(
        {
            "interval_minutes": interval_minutes,
            "schema_version": _ALIGNED_KLINE_SCHEMA,
            "source_series_digest": series.digest,
        },
        (("decision_timestamps_ns", decisions_ns),),
    )
    return AlignedV4KlineSeries(
        close=close,
        quote_volume=quote,
        taker_buy_quote_volume=taker,
        available=available,
        source_digest=digest,
    )


def align_funding_events_to_decisions(
    decision_timestamps: object,
    events: BinanceFundingEventSeries,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Place each actual funding event on its first observable decision exactly once."""

    if not isinstance(events, BinanceFundingEventSeries):
        raise TypeError("events must be BinanceFundingEventSeries")
    decisions_ns = _decision_ns(decision_timestamps)
    decision_ms = decisions_ns // 1_000_000
    rate = np.zeros(decision_ms.shape, dtype=np.float64)
    available = np.zeros(decision_ms.shape, dtype=np.bool_)
    for event_time, event_rate in zip(events.event_time_ms, events.rate, strict=True):
        row = int(np.searchsorted(decision_ms, int(event_time), side="left"))
        if row >= len(decision_ms):
            continue
        if available[row]:
            raise ValueError("multiple funding events map to one V4 decision")
        rate[row] = float(event_rate)
        available[row] = True
    rate.setflags(write=False)
    available.setflags(write=False)
    digest = content_and_arrays_digest(
        {
            "events_digest": events.digest,
            "schema_version": _ALIGNED_FUNDING_SCHEMA,
        },
        (
            ("decision_timestamps_ns", decisions_ns),
            ("rate", rate),
            ("available", available),
        ),
    )
    return rate, available, digest


def _aligned_size(value: AlignedV4KlineSeries, *, field: str, size: int) -> None:
    if len(value.available) != size:
        raise ValueError(f"{field} does not align with decisions")


def assemble_v4_cross_market_inputs(
    *,
    decision_indices: object,
    decision_timestamps: object,
    spot: AlignedV4KlineSeries,
    perp: AlignedV4KlineSeries,
    mark: AlignedV4KlineSeries,
    funding_event_rate: object,
    funding_event_available: object,
    metrics: AlignedFuturesMetrics | None,
) -> V4CrossMarketInputs:
    """Combine causally aligned exchange sources into one V4 input contract."""

    decisions = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
    timestamps = np.asarray(decision_timestamps, dtype="datetime64[ns]").reshape(-1)
    if decisions.size == 0 or timestamps.shape != decisions.shape:
        raise ValueError("V4 decision indices/timestamps must align and be non-empty")
    if np.any(decisions < 0) or (
        decisions.size > 1 and np.any(decisions[1:] <= decisions[:-1])
    ):
        raise ValueError(
            "V4 decision indices must be strictly increasing and non-negative"
        )
    _decision_ns(timestamps)
    size = decisions.size
    for field, value in (("spot", spot), ("perp", perp), ("mark", mark)):
        if not isinstance(value, AlignedV4KlineSeries):
            raise TypeError(f"{field} must be AlignedV4KlineSeries")
        _aligned_size(value, field=field, size=size)

    funding_rate = np.asarray(funding_event_rate, dtype=np.float64).reshape(-1)
    funding_available = np.asarray(funding_event_available, dtype=np.bool_).reshape(-1)
    if funding_rate.shape != (size,) or funding_available.shape != (size,):
        raise ValueError("V4 funding event arrays must align with decisions")
    if not np.isfinite(funding_rate).all():
        raise ValueError("V4 funding event rate must be finite")

    spot_available = np.asarray(spot.available, dtype=np.bool_)
    perp_available = np.asarray(perp.available & mark.available, dtype=np.bool_)
    spot_close = np.where(spot_available, spot.close, 0.0)
    spot_quote = np.where(spot_available, spot.quote_volume, 0.0)
    spot_taker = np.where(spot_available, spot.taker_buy_quote_volume, 0.0)
    perp_close = np.where(perp_available, perp.close, 0.0)
    perp_mark = np.where(perp_available, mark.close, 0.0)
    perp_quote = np.where(perp_available, perp.quote_volume, 0.0)
    perp_taker = np.where(perp_available, perp.taker_buy_quote_volume, 0.0)

    open_interest: np.ndarray | None = None
    global_ratio: np.ndarray | None = None
    top_ratio: np.ndarray | None = None
    derivative_available: np.ndarray | None = None
    derivative_staleness: np.ndarray | None = None
    metrics_digest: str | None = None
    if metrics is not None:
        if not isinstance(metrics, AlignedFuturesMetrics):
            raise TypeError("metrics must be AlignedFuturesMetrics or null")
        if len(metrics.available) != size:
            raise ValueError("V4 metrics do not align with decisions")
        open_interest = np.asarray(metrics.open_interest_value, dtype=np.float64)
        global_ratio = np.asarray(metrics.global_long_short_ratio, dtype=np.float64)
        top_ratio = np.asarray(metrics.top_position_long_short_ratio, dtype=np.float64)
        derivative_available = np.asarray(metrics.available, dtype=np.bool_)
        derivative_staleness = np.asarray(metrics.staleness_hours, dtype=np.float64)
        metrics_digest = metrics.digest

    source_digest = content_and_arrays_digest(
        {
            "mark_digest": mark.digest,
            "metrics_digest": metrics_digest,
            "perp_digest": perp.digest,
            "schema_version": _ASSEMBLED_SOURCE_SCHEMA,
            "spot_digest": spot.digest,
        },
        (
            ("decision_indices", decisions),
            ("decision_timestamps_ns", timestamps.astype(np.int64)),
            ("funding_event_rate", funding_rate),
            ("funding_event_available", funding_available),
        ),
    )
    return V4CrossMarketInputs(
        decision_indices=decisions,
        decision_timestamps=timestamps,
        spot_close=spot_close,
        spot_quote_volume=spot_quote,
        spot_taker_buy_quote_volume=spot_taker,
        spot_row_available=spot_available,
        perp_close=perp_close,
        perp_mark_price=perp_mark,
        perp_quote_volume=perp_quote,
        perp_taker_buy_quote_volume=perp_taker,
        perp_row_available=perp_available,
        funding_event_rate=funding_rate,
        funding_event_available=funding_available,
        open_interest_value=open_interest,
        global_long_short_ratio=global_ratio,
        top_position_long_short_ratio=top_ratio,
        derivatives_available=derivative_available,
        derivatives_staleness_hours=derivative_staleness,
        source_digest=source_digest,
    )


__all__ = [
    "AlignedV4KlineSeries",
    "align_funding_events_to_decisions",
    "align_v4_kline_to_decisions",
    "assemble_v4_cross_market_inputs",
    "vision_mark_price_kline_url",
]
