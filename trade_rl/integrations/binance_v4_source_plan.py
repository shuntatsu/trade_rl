"""Immutable Binance Vision source planning for Causal Alpha V4 context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import numpy as np

from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.domain.common import require_sha256
from trade_rl.integrations.binance_v4_context import vision_futures_metrics_url

_BINANCE_VISION_ROOT: Final = "https://data.binance.vision/data/"
_REFERENCE_CLOCK_SCHEMA: Final = "binance_v4_reference_decision_clock_v1"
_SOURCE_PLAN_SCHEMA: Final = "binance_v4_symbol_source_plan_v1"
_DECISION_INTERVAL: Final = np.timedelta64(15, "m")
_DECISION_INTERVAL_NS: Final = 15 * 60 * 1_000_000_000


def _decision_timestamps(value: object, *, field: str) -> np.ndarray:
    timestamps = np.asarray(value, dtype="datetime64[ns]").reshape(-1).copy(order="C")
    if timestamps.size == 0 or np.any(np.isnat(timestamps)):
        raise ValueError(f"{field} must be non-empty and finite")
    timestamp_ns = timestamps.astype(np.int64)
    if np.any(timestamp_ns % _DECISION_INTERVAL_NS != 0):
        raise ValueError(f"{field} must lie on the regular 15-minute cadence")
    if timestamps.size > 1 and np.any(np.diff(timestamp_ns) != _DECISION_INTERVAL_NS):
        raise ValueError(f"{field} must use a regular 15-minute cadence")
    timestamps.setflags(write=False)
    return timestamps


def _symbol(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("V4 Binance symbol must be non-empty")
    symbol = value.strip().upper()
    if not symbol.isalnum():
        raise ValueError("V4 Binance symbol must be an alphanumeric path segment")
    return symbol


def _unique_period_strings(values: np.ndarray, unit: str) -> tuple[str, ...]:
    periods = np.unique(values.astype(f"datetime64[{unit}]"))
    return tuple(str(period) for period in periods)


def _daily_kline_url(*, symbol: str, day: str, path: str) -> str:
    return f"{_BINANCE_VISION_ROOT}{path}/{symbol}/15m/{symbol}-15m-{day}.zip"


def _funding_url(*, symbol: str, month: str) -> str:
    return (
        f"{_BINANCE_VISION_ROOT}futures/um/monthly/fundingRate/{symbol}/"
        f"{symbol}-fundingRate-{month}.zip"
    )


def _metrics_url(*, symbol: str, day: str) -> str:
    return vision_futures_metrics_url(
        symbol,
        datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC),
    )


@dataclass(frozen=True, slots=True)
class V4ReferenceDecisionClock:
    """Exact existing BTCUSDT 15-minute decision clock used by V4 sources."""

    decision_indices: np.ndarray
    decision_timestamps: np.ndarray
    source_digest: str

    def __post_init__(self) -> None:
        timestamps = _decision_timestamps(
            self.decision_timestamps,
            field="V4 reference decision timestamps",
        )
        indices = np.asarray(self.decision_indices, dtype=np.int64).reshape(-1).copy()
        if indices.shape != timestamps.shape or not np.array_equal(
            indices, np.arange(timestamps.size, dtype=np.int64)
        ):
            raise ValueError(
                "V4 reference decision indices must be zero-based and contiguous"
            )
        require_sha256(self.source_digest, field="V4 reference clock source_digest")
        indices.setflags(write=False)
        object.__setattr__(self, "decision_indices", indices)
        object.__setattr__(self, "decision_timestamps", timestamps)


@dataclass(frozen=True, slots=True)
class BinanceV4SymbolSourcePlan:
    """Deterministic archive URLs required for one symbol and decision clock."""

    symbol: str
    spot_kline_urls: tuple[str, ...]
    perp_kline_urls: tuple[str, ...]
    mark_price_kline_urls: tuple[str, ...]
    funding_urls: tuple[str, ...]
    metrics_urls: tuple[str, ...]
    source_digest: str

    def __post_init__(self) -> None:
        symbol = _symbol(self.symbol)
        resolved: dict[str, tuple[str, ...]] = {}
        for field_name in (
            "spot_kline_urls",
            "perp_kline_urls",
            "mark_price_kline_urls",
            "funding_urls",
            "metrics_urls",
        ):
            urls = tuple(getattr(self, field_name))
            if field_name != "metrics_urls" and not urls:
                raise ValueError(f"V4 source plan {field_name} must not be empty")
            if len(set(urls)) != len(urls):
                raise ValueError(f"V4 source plan {field_name} contains duplicates")
            if any(
                not isinstance(url, str) or not url.startswith(_BINANCE_VISION_ROOT)
                for url in urls
            ):
                raise ValueError(f"V4 source plan {field_name} contains an invalid URL")
            resolved[field_name] = urls
        require_sha256(self.source_digest, field="V4 source plan source_digest")
        object.__setattr__(self, "symbol", symbol)
        for field_name, urls in resolved.items():
            object.__setattr__(self, field_name, urls)


def build_v4_reference_decision_clock(dataset: object) -> V4ReferenceDecisionClock:
    """Bind V4 source planning to the existing immutable BTCUSDT 15m dataset."""

    dataset_id = getattr(dataset, "dataset_id", None)
    if not isinstance(dataset_id, str):
        raise ValueError("V4 reference dataset_id is unavailable")
    require_sha256(dataset_id, field="V4 reference dataset_id")
    symbols = tuple(getattr(dataset, "symbols", ()))
    if symbols != ("BTCUSDT",):
        raise ValueError("V4 reference dataset must contain only BTCUSDT")
    if getattr(dataset, "calendar_kind", None) != "continuous_24_7":
        raise ValueError("V4 reference dataset must use the continuous_24_7 calendar")
    nominal_bar_hours = getattr(dataset, "nominal_bar_hours", None)
    if (
        isinstance(nominal_bar_hours, bool)
        or not isinstance(nominal_bar_hours, (int, float))
        or float(nominal_bar_hours) != 0.25
    ):
        raise ValueError("V4 reference dataset must use a 15-minute cadence")
    timestamps = _decision_timestamps(
        getattr(dataset, "timestamps", None),
        field="V4 reference dataset timestamps",
    )
    source_digest = content_and_arrays_digest(
        {
            "calendar_kind": "continuous_24_7",
            "dataset_id": dataset_id,
            "nominal_bar_hours": 0.25,
            "schema_version": _REFERENCE_CLOCK_SCHEMA,
            "symbols": symbols,
        },
        (("decision_timestamps_ns", timestamps.astype(np.int64)),),
    )
    return V4ReferenceDecisionClock(
        decision_indices=np.arange(timestamps.size, dtype=np.int64),
        decision_timestamps=timestamps,
        source_digest=source_digest,
    )


def plan_binance_v4_symbol_sources(
    *,
    symbol: str,
    decision_timestamps: object,
    include_metrics: bool,
) -> BinanceV4SymbolSourcePlan:
    """Plan immutable archives without reading source or outcome values."""

    if not isinstance(include_metrics, bool):
        raise TypeError("V4 include_metrics must be boolean")
    resolved_symbol = _symbol(symbol)
    timestamps = _decision_timestamps(
        decision_timestamps,
        field="V4 source-plan decision timestamps",
    )

    # Decision timestamps denote closed 15-minute bars. Kline archives are
    # therefore selected by each bar's open day, while event metrics include the
    # decision day because an event exactly at the decision close is observable.
    open_timestamps = timestamps - _DECISION_INTERVAL
    kline_days = _unique_period_strings(open_timestamps, "D")
    metrics_days = _unique_period_strings(timestamps, "D")
    funding_months = _unique_period_strings(
        np.concatenate((open_timestamps, timestamps)),
        "M",
    )

    spot_urls = tuple(
        _daily_kline_url(
            symbol=resolved_symbol,
            day=day,
            path="spot/daily/klines",
        )
        for day in kline_days
    )
    perp_urls = tuple(
        _daily_kline_url(
            symbol=resolved_symbol,
            day=day,
            path="futures/um/daily/klines",
        )
        for day in kline_days
    )
    mark_urls = tuple(
        _daily_kline_url(
            symbol=resolved_symbol,
            day=day,
            path="futures/um/daily/markPriceKlines",
        )
        for day in kline_days
    )
    funding_urls = tuple(
        _funding_url(symbol=resolved_symbol, month=month) for month in funding_months
    )
    metrics_urls = (
        tuple(_metrics_url(symbol=resolved_symbol, day=day) for day in metrics_days)
        if include_metrics
        else ()
    )
    source_digest = content_and_arrays_digest(
        {
            "funding_urls": funding_urls,
            "include_metrics": include_metrics,
            "mark_price_kline_urls": mark_urls,
            "metrics_urls": metrics_urls,
            "perp_kline_urls": perp_urls,
            "schema_version": _SOURCE_PLAN_SCHEMA,
            "spot_kline_urls": spot_urls,
            "symbol": resolved_symbol,
        },
        (("decision_timestamps_ns", timestamps.astype(np.int64)),),
    )
    return BinanceV4SymbolSourcePlan(
        symbol=resolved_symbol,
        spot_kline_urls=spot_urls,
        perp_kline_urls=perp_urls,
        mark_price_kline_urls=mark_urls,
        funding_urls=funding_urls,
        metrics_urls=metrics_urls,
        source_digest=source_digest,
    )


__all__ = [
    "BinanceV4SymbolSourcePlan",
    "V4ReferenceDecisionClock",
    "build_v4_reference_decision_clock",
    "plan_binance_v4_symbol_sources",
]
