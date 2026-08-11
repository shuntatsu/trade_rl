"""Deterministic causal construction of the maintained native indicator cache."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import FeatureSpec
from trade_rl.data.features import calculate_feature_events
from trade_rl.integrations.binance_universal import binance_universal_feature_specs
from trade_rl.integrations.postgres_market_tables import (
    UNIVERSAL_202411_202607_CACHE_ID,
)
from trade_rl.integrations.postgres_universal_source import (
    RawSymbolSource,
    UniversalSourceScope,
)

NATIVE_TIMEFRAME_MINUTES: Final = {"15m": 15, "1h": 60, "4h": 240, "1d": 1_440}
VOLUME_CONVERSION_METHOD: Final = "base_volume_times_minute_close_v1"
_MINUTE_NS: Final = 60_000_000_000
_MS_PER_HOUR: Final = 3_600_000.0


@dataclass(frozen=True, slots=True)
class NativeBars:
    """Completed native bars whose event time is the exclusive bar end."""

    open_time_ms: np.ndarray
    event_time_ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quote_volume: np.ndarray
    funding_rate: np.ndarray
    funding_available: np.ndarray
    derivative_values: np.ndarray
    derivative_available: np.ndarray
    derivative_staleness_hours: np.ndarray
    orderflow_values: np.ndarray
    orderflow_available: np.ndarray
    orderflow_staleness_hours: np.ndarray
    incomplete_source_minutes: int

    def __post_init__(self) -> None:
        for value in (
            self.open_time_ms,
            self.event_time_ms,
            self.open,
            self.high,
            self.low,
            self.close,
            self.quote_volume,
            self.funding_rate,
            self.funding_available,
            self.derivative_values,
            self.derivative_available,
            self.derivative_staleness_hours,
            self.orderflow_values,
            self.orderflow_available,
            self.orderflow_staleness_hours,
        ):
            value.setflags(write=False)


@dataclass(frozen=True, slots=True)
class NativeArtifactPayload:
    """Canonical NPZ bytes and decoded arrays for one symbol/timeframe member."""

    symbol: str
    timeframe: str
    feature_names: tuple[str, ...]
    event_time_ms: np.ndarray
    values: np.ndarray
    available: np.ndarray
    payload_schema: str
    payload_sha256: str
    payload: bytes

    def __post_init__(self) -> None:
        for value in (self.event_time_ms, self.values, self.available):
            value.setflags(write=False)

    @property
    def row_count(self) -> int:
        return len(self.event_time_ms)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def available_value_count(self) -> int:
        return int(np.count_nonzero(self.available))


@dataclass(frozen=True, slots=True)
class FeatureStatistic:
    """One feature's availability and finite distribution audit."""

    name: str
    available_count: int
    nonfinite_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    standard_deviation: float | None
    extreme_count: int

    def canonical_payload(self) -> dict[str, object]:
        return {
            "available_count": self.available_count,
            "extreme_count": self.extreme_count,
            "maximum": self.maximum,
            "mean": self.mean,
            "minimum": self.minimum,
            "name": self.name,
            "nonfinite_count": self.nonfinite_count,
            "standard_deviation": self.standard_deviation,
        }


@dataclass(frozen=True, slots=True)
class IntermediateMemberReport:
    """Auditable source, bar, auxiliary, and feature statistics for one member."""

    symbol: str
    timeframe: str
    row_count: int
    first_event_time_ms: int | None
    last_event_time_ms: int | None
    missing_timestamps: int
    duplicate_timestamps: int
    incomplete_bars: int
    ohlcv_violations: int
    derivative_available_count: int
    derivative_max_staleness_hours: float | None
    orderflow_available_count: int
    orderflow_max_staleness_hours: float | None
    nonfinite_available_features: int
    feature_statistics: tuple[FeatureStatistic, ...]
    payload_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "derivative_available_count": self.derivative_available_count,
            "derivative_max_staleness_hours": self.derivative_max_staleness_hours,
            "duplicate_timestamps": self.duplicate_timestamps,
            "feature_statistics": tuple(
                item.canonical_payload() for item in self.feature_statistics
            ),
            "first_event_time_ms": self.first_event_time_ms,
            "incomplete_bars": self.incomplete_bars,
            "last_event_time_ms": self.last_event_time_ms,
            "missing_timestamps": self.missing_timestamps,
            "nonfinite_available_features": self.nonfinite_available_features,
            "ohlcv_violations": self.ohlcv_violations,
            "orderflow_available_count": self.orderflow_available_count,
            "orderflow_max_staleness_hours": self.orderflow_max_staleness_hours,
            "payload_sha256": self.payload_sha256,
            "row_count": self.row_count,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        }


@dataclass(frozen=True, slots=True)
class IntermediateDataReport:
    """Deterministic aggregate audit report for every cache member."""

    volume_conversion_method: str
    members: tuple[IntermediateMemberReport, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class NativeCacheManifest:
    """Content-bound cache manifest ready for atomic PostgreSQL publication."""

    cache_id: str
    symbols: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    feature_specs: Mapping[str, object]
    feature_config_digest: str
    feature_count: int
    artifact_count: int
    volume_conversion_method: str
    digest: str


@dataclass(frozen=True, slots=True)
class NativeCacheBuild:
    """Complete pure build result prior to database publication."""

    market_bars: Mapping[tuple[str, str], NativeBars]
    artifacts: tuple[NativeArtifactPayload, ...]
    manifest: NativeCacheManifest
    report: IntermediateDataReport


def _scope_ns(scope: UniversalSourceScope) -> tuple[int, int]:
    return (
        int(round(scope.start.timestamp() * 1_000_000_000.0)),
        int(round(scope.end.timestamp() * 1_000_000_000.0)),
    )


def _source_slice(
    raw: RawSymbolSource, *, scope: UniversalSourceScope, symbol: str
) -> tuple[np.ndarray, ...]:
    timestamps_ns = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
    start_ns, end_ns = _scope_ns(scope)
    selected = (timestamps_ns >= start_ns) & (timestamps_ns < end_ns)
    scoped_times = timestamps_ns[selected]
    expected = np.arange(start_ns, end_ns, _MINUTE_NS, dtype=np.int64)
    if not np.array_equal(scoped_times, expected):
        raise ValueError(f"native materializer source is not contiguous for {symbol}")
    arrays = (
        raw.open,
        raw.high,
        raw.low,
        raw.close,
        raw.base_volume,
        raw.derivative_available,
        raw.orderflow_available,
    )
    if any(len(value) != len(timestamps_ns) for value in arrays):
        raise ValueError(f"native materializer source shape mismatch for {symbol}")
    if raw.derivative_values.shape != (len(timestamps_ns), 4):
        raise ValueError(f"native derivative shape mismatch for {symbol}")
    if raw.orderflow_values.shape != (len(timestamps_ns), 5):
        raise ValueError(f"native orderflow shape mismatch for {symbol}")
    return (
        scoped_times,
        raw.open[selected],
        raw.high[selected],
        raw.low[selected],
        raw.close[selected],
        raw.base_volume[selected],
        raw.derivative_values[selected],
        raw.derivative_available[selected],
        raw.orderflow_values[selected],
        raw.orderflow_available[selected],
    )


def _align_sparse_asof(
    *,
    source_times_ns: np.ndarray,
    source_values: np.ndarray,
    source_available: np.ndarray,
    event_time_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = source_values.shape[1]
    values = np.zeros((len(event_time_ms), width), dtype=np.float64)
    available = np.zeros(len(event_time_ms), dtype=np.bool_)
    staleness = np.zeros(len(event_time_ms), dtype=np.float64)
    valid_indices = np.flatnonzero(source_available)
    if not len(valid_indices) or not len(event_time_ms):
        return values, available, staleness
    valid_times = source_times_ns[valid_indices]
    events_ns = event_time_ms * 1_000_000
    positions = np.searchsorted(valid_times, events_ns, side="right") - 1
    present = positions >= 0
    source_indices = valid_indices[np.maximum(positions, 0)]
    values[present] = source_values[source_indices[present]]
    available[present] = True
    staleness[present] = (
        events_ns[present] - source_times_ns[source_indices[present]]
    ) / (_MS_PER_HOUR * 1_000_000.0)
    return values, available, staleness


def _align_funding(
    raw: RawSymbolSource, *, event_time_ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rates = np.zeros(len(event_time_ms), dtype=np.float64)
    available = np.zeros(len(event_time_ms), dtype=np.bool_)
    funding_ms = raw.funding_timestamps.astype("datetime64[ms]").astype(np.int64)
    if len(funding_ms) != len(raw.funding_rate):
        raise ValueError("funding timestamp/value shape mismatch")
    if len(funding_ms) > 1 and np.any(np.diff(funding_ms) <= 0):
        raise ValueError("funding timestamps must be strictly increasing")
    if not np.isfinite(raw.funding_rate).all():
        raise ValueError("funding rates must be finite")
    for timestamp_ms, rate in zip(funding_ms, raw.funding_rate, strict=True):
        position = int(np.searchsorted(event_time_ms, timestamp_ms, side="left"))
        if position < len(event_time_ms):
            rates[position] += float(rate)
            available[position] = True
    return rates, available


def resample_completed_bars(
    raw: RawSymbolSource,
    *,
    scope: UniversalSourceScope,
    symbol: str,
    minutes: int,
) -> NativeBars:
    """Aggregate completed bars with causal minute-close quote notionals."""

    (
        timestamps_ns,
        open_,
        high,
        low,
        close,
        base_volume,
        derivative_values,
        derivative_available,
        orderflow_values,
        orderflow_available,
    ) = _source_slice(raw, scope=scope, symbol=symbol)
    complete = len(timestamps_ns) // minutes
    used = complete * minutes
    shape = (complete, minutes)
    open_time_ms = timestamps_ns[:used].reshape(shape)[:, 0] // 1_000_000
    event_time_ms = (
        timestamps_ns[:used].reshape(shape)[:, -1] + _MINUTE_NS
    ) // 1_000_000
    funding_rate, funding_available = _align_funding(
        raw, event_time_ms=event_time_ms
    )
    derivative = _align_sparse_asof(
        source_times_ns=timestamps_ns,
        source_values=derivative_values,
        source_available=derivative_available,
        event_time_ms=event_time_ms,
    )
    orderflow = _align_sparse_asof(
        source_times_ns=timestamps_ns,
        source_values=orderflow_values,
        source_available=orderflow_available,
        event_time_ms=event_time_ms,
    )
    return NativeBars(
        open_time_ms=np.asarray(open_time_ms, dtype=np.int64),
        event_time_ms=np.asarray(event_time_ms, dtype=np.int64),
        open=open_[:used].reshape(shape)[:, 0],
        high=high[:used].reshape(shape).max(axis=1),
        low=low[:used].reshape(shape).min(axis=1),
        close=close[:used].reshape(shape)[:, -1],
        quote_volume=(base_volume[:used] * close[:used]).reshape(shape).sum(axis=1),
        funding_rate=funding_rate,
        funding_available=funding_available,
        derivative_values=derivative[0],
        derivative_available=derivative[1],
        derivative_staleness_hours=derivative[2],
        orderflow_values=orderflow[0],
        orderflow_available=orderflow[1],
        orderflow_staleness_hours=orderflow[2],
        incomplete_source_minutes=len(timestamps_ns) - used,
    )


def _feature_payload(specs: Sequence[FeatureSpec]) -> dict[str, object]:
    return {
        "base_timeframe": "15m",
        "feature_timeframes": ["1h", "4h", "1d"],
        "features": [spec.canonical_payload() for spec in specs],
    }


def _npz_payload(
    event_time_ms: np.ndarray, values: np.ndarray, available: np.ndarray
) -> bytes:
    output = io.BytesIO()
    np.savez(
        output,
        event_time_ms=np.asarray(event_time_ms, dtype=np.int64),
        values=np.asarray(values, dtype=np.float32),
        available=np.asarray(available, dtype=np.bool_),
    )
    return output.getvalue()


def _build_artifact(
    *, symbol: str, timeframe: str, bars: NativeBars, specs: Sequence[FeatureSpec]
) -> NativeArtifactPayload:
    values = np.zeros((len(bars.event_time_ms), len(specs)), dtype=np.float32)
    available = np.zeros(values.shape, dtype=np.bool_)
    active = np.ones(len(bars.event_time_ms), dtype=np.bool_)
    for index, spec in enumerate(specs):
        if not len(bars.event_time_ms):
            continue
        feature_values, feature_available, _ = calculate_feature_events(
            spec,
            open_price=bars.open,
            high=bars.high,
            low=bars.low,
            close=bars.close,
            volume=bars.quote_volume,
            funding_rate=bars.funding_rate,
            funding_available=bars.funding_available,
            row_present=active,
            active=active,
        )
        if not np.isfinite(feature_values[feature_available]).all():
            raise ValueError(f"available feature is non-finite: {symbol} {spec.name}")
        values[:, index] = np.asarray(feature_values, dtype=np.float32)
        available[:, index] = feature_available
    names = tuple(spec.name for spec in specs)
    payload = _npz_payload(bars.event_time_ms, values, available)
    schema_digest = content_digest(
        {"feature_names": names, "schema_version": "npz_native_indicator_v1"}
    )
    return NativeArtifactPayload(
        symbol=symbol,
        timeframe=timeframe,
        feature_names=names,
        event_time_ms=bars.event_time_ms.copy(),
        values=values,
        available=available,
        payload_schema=f"npz_native_indicator_v1:{schema_digest}",
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )


def _feature_statistics(artifact: NativeArtifactPayload) -> tuple[FeatureStatistic, ...]:
    result: list[FeatureStatistic] = []
    for index, name in enumerate(artifact.feature_names):
        sample = artifact.values[artifact.available[:, index], index].astype(np.float64)
        nonfinite = int(np.count_nonzero(~np.isfinite(sample)))
        finite = sample[np.isfinite(sample)]
        if not finite.size:
            result.append(FeatureStatistic(name, len(sample), nonfinite, None, None, None, None, 0))
            continue
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        extreme = (
            0
            if std <= 0.0
            else int(np.count_nonzero(np.abs(finite - mean) > 20.0 * std))
        )
        result.append(
            FeatureStatistic(
                name=name,
                available_count=len(sample),
                nonfinite_count=nonfinite,
                minimum=float(np.min(finite)),
                maximum=float(np.max(finite)),
                mean=mean,
                standard_deviation=std,
                extreme_count=extreme,
            )
        )
    return tuple(result)


def _member_report(
    *,
    artifact: NativeArtifactPayload,
    bars: NativeBars,
    expected_rows: int,
) -> IntermediateMemberReport:
    event_times = artifact.event_time_ms
    duplicate = int(len(event_times) - len(np.unique(event_times)))
    missing = max(expected_rows - len(event_times), 0)
    ohlcv_violations = int(
        np.count_nonzero(
            (bars.high < np.maximum.reduce((bars.open, bars.close, bars.low)))
            | (bars.low > np.minimum.reduce((bars.open, bars.close, bars.high)))
            | ~np.isfinite(
                np.column_stack(
                    (bars.open, bars.high, bars.low, bars.close, bars.quote_volume)
                )
            ).all(axis=1)
            | (bars.quote_volume < 0.0)
        )
    )
    statistics = _feature_statistics(artifact)
    nonfinite = sum(item.nonfinite_count for item in statistics)
    if nonfinite:
        raise ValueError("available feature report contains non-finite values")
    derivative_stale = bars.derivative_staleness_hours[bars.derivative_available]
    orderflow_stale = bars.orderflow_staleness_hours[bars.orderflow_available]
    return IntermediateMemberReport(
        symbol=artifact.symbol,
        timeframe=artifact.timeframe,
        row_count=artifact.row_count,
        first_event_time_ms=None if not len(event_times) else int(event_times[0]),
        last_event_time_ms=None if not len(event_times) else int(event_times[-1]),
        missing_timestamps=missing,
        duplicate_timestamps=duplicate,
        incomplete_bars=bars.incomplete_source_minutes,
        ohlcv_violations=ohlcv_violations,
        derivative_available_count=int(np.count_nonzero(bars.derivative_available)),
        derivative_max_staleness_hours=(
            None if not len(derivative_stale) else float(np.max(derivative_stale))
        ),
        orderflow_available_count=int(np.count_nonzero(bars.orderflow_available)),
        orderflow_max_staleness_hours=(
            None if not len(orderflow_stale) else float(np.max(orderflow_stale))
        ),
        nonfinite_available_features=nonfinite,
        feature_statistics=statistics,
        payload_sha256=artifact.payload_sha256,
    )


def build_native_indicator_cache(
    source: Mapping[str, RawSymbolSource],
    *,
    scope: UniversalSourceScope,
) -> NativeCacheBuild:
    """Build deterministic native bars, 206 features, and full intermediate audit."""

    if tuple(source) != scope.symbols:
        raise ValueError("native materializer source order must match scope symbols")
    specs = binance_universal_feature_specs(
        base_timeframe="15m", feature_timeframes=("1h", "4h", "1d")
    )
    if len(specs) != 206:
        raise RuntimeError("Universal feature contract is not 206 channels")
    specs_by_timeframe = {
        timeframe: tuple(
            spec for spec in specs if spec.resolved_timeframe("15m") == timeframe
        )
        for timeframe in NATIVE_TIMEFRAME_MINUTES
    }
    if sum(len(items) for items in specs_by_timeframe.values()) != 206:
        raise RuntimeError("Universal features do not map to maintained native clocks")

    market_bars: dict[tuple[str, str], NativeBars] = {}
    artifacts: list[NativeArtifactPayload] = []
    members: list[IntermediateMemberReport] = []
    total_minutes = int((scope.end - scope.start).total_seconds() // 60)
    for symbol in scope.symbols:
        raw = source[symbol]
        for timeframe, minutes in NATIVE_TIMEFRAME_MINUTES.items():
            bars = resample_completed_bars(
                raw, scope=scope, symbol=symbol, minutes=minutes
            )
            market_bars[(symbol, timeframe)] = bars
            artifact = _build_artifact(
                symbol=symbol,
                timeframe=timeframe,
                bars=bars,
                specs=specs_by_timeframe[timeframe],
            )
            artifacts.append(artifact)
            members.append(
                _member_report(
                    artifact=artifact,
                    bars=bars,
                    expected_rows=total_minutes // minutes,
                )
            )

    feature_payload = _feature_payload(specs)
    feature_digest = content_digest(feature_payload)
    manifest_core = {
        "artifact_digests": tuple(item.payload_sha256 for item in artifacts),
        "cache_id": UNIVERSAL_202411_202607_CACHE_ID,
        "end_time": scope.end.isoformat(),
        "feature_config_digest": feature_digest,
        "feature_count": len(specs),
        "schema_version": "native_indicator_cache_v1",
        "start_time": scope.start.isoformat(),
        "symbols": scope.symbols,
        "volume_conversion_method": VOLUME_CONVERSION_METHOD,
    }
    manifest_digest = content_digest(manifest_core)
    report_core = {
        "manifest_digest": manifest_digest,
        "members": tuple(item.canonical_payload() for item in members),
        "schema_version": "native_indicator_intermediate_report_v1",
        "volume_conversion_method": VOLUME_CONVERSION_METHOD,
    }
    report = IntermediateDataReport(
        volume_conversion_method=VOLUME_CONVERSION_METHOD,
        members=tuple(members),
        digest=content_digest(report_core),
    )
    manifest = NativeCacheManifest(
        cache_id=UNIVERSAL_202411_202607_CACHE_ID,
        symbols=scope.symbols,
        start_time=scope.start,
        end_time=scope.end,
        feature_specs=feature_payload,
        feature_config_digest=feature_digest,
        feature_count=len(specs),
        artifact_count=len(artifacts),
        volume_conversion_method=VOLUME_CONVERSION_METHOD,
        digest=manifest_digest,
    )
    return NativeCacheBuild(
        market_bars=market_bars,
        artifacts=tuple(artifacts),
        manifest=manifest,
        report=report,
    )


__all__ = [
    "NATIVE_TIMEFRAME_MINUTES",
    "VOLUME_CONVERSION_METHOD",
    "FeatureStatistic",
    "IntermediateDataReport",
    "IntermediateMemberReport",
    "NativeArtifactPayload",
    "NativeBars",
    "NativeCacheBuild",
    "NativeCacheManifest",
    "build_native_indicator_cache",
    "resample_completed_bars",
]
