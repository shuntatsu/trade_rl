"""Causal native-timeframe feature calculation and base-clock alignment."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    NormalizationMode,
    timeframe_hours,
)
from trade_rl.data.source import RawMarketSeries

_NS_PER_HOUR = 3_600_000_000_000


def _utc_datetime64(value: datetime) -> np.datetime64:
    resolved = value.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(resolved, "ns")


def _validate_regular_native_series(raw: RawMarketSeries, timeframe: str) -> None:
    expected_step = int(round(timeframe_hours(timeframe) * _NS_PER_HOUR))
    timestamps = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
    if timestamps.size >= 2 and np.any(np.diff(timestamps) != expected_step):
        raise ValueError(
            f"native {timeframe} market series must be complete and exactly regular"
        )


def _active_mask(raw: RawMarketSeries, contract: InstrumentContract) -> np.ndarray:
    active = raw.timestamps >= _utc_datetime64(contract.listed_at)
    if contract.delisted_at is not None:
        active &= raw.timestamps < _utc_datetime64(contract.delisted_at)
    return active


def _window_available_at(raw: RawMarketSeries, start: int, stop: int) -> np.datetime64:
    assert raw.available_at is not None
    return np.max(raw.available_at[start:stop])


def _native_events(
    spec: FeatureSpec,
    raw: RawMarketSeries,
    contract: InstrumentContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_bars = len(raw.timestamps)
    values = np.zeros(n_bars, dtype=np.float64)
    valid = np.zeros(n_bars, dtype=np.bool_)
    available_at = np.full(n_bars, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    active = _active_mask(raw, contract)

    if spec.kind is FeatureKind.FUNDING_BPS:
        assert raw.funding_available is not None
        assert raw.available_at is not None
        valid = raw.funding_available & active
        values[valid] = raw.funding_rate[valid] * 10_000.0
        available_at[valid] = raw.available_at[valid]
    elif spec.kind is FeatureKind.LOG_RETURN:
        for index in range(spec.lookback, n_bars):
            start = index - spec.lookback
            if not np.all(active[start : index + 1]):
                continue
            values[index] = math.log(raw.close[index] / raw.close[start])
            valid[index] = True
            available_at[index] = _window_available_at(raw, start, index + 1)
    elif spec.kind is FeatureKind.REALIZED_VOLATILITY:
        for index in range(spec.lookback, n_bars):
            start = index - spec.lookback
            if not np.all(active[start : index + 1]):
                continue
            returns = np.diff(np.log(raw.close[start : index + 1]))
            values[index] = float(np.sqrt(np.mean(np.square(returns))))
            valid[index] = True
            available_at[index] = _window_available_at(raw, start, index + 1)
    elif spec.kind is FeatureKind.VOLUME_ZSCORE:
        for index in range(n_bars):
            start = max(0, index - spec.lookback + 1)
            window_active = active[start : index + 1]
            sample = raw.volume[start : index + 1][window_active]
            if not active[index] or sample.size < spec.min_periods:
                continue
            std = float(np.std(sample))
            values[index] = (
                0.0
                if std <= 1e-12
                else (raw.volume[index] - float(np.mean(sample))) / std
            )
            valid[index] = True
            available_at[index] = _window_available_at(raw, start, index + 1)
    else:
        raise ValueError(f"unsupported feature kind: {spec.kind}")

    if spec.normalization is not NormalizationMode.ROLLING_ZSCORE:
        return values, valid, available_at

    normalized = np.zeros_like(values)
    normalized_valid = np.zeros_like(valid)
    normalized_available_at = np.full_like(available_at, np.datetime64("NaT", "ns"))
    for index in range(n_bars):
        if not valid[index]:
            continue
        start = max(0, index - spec.normalization_window + 1)
        sample_mask = valid[start : index + 1]
        sample = values[start : index + 1][sample_mask]
        if sample.size < spec.min_periods:
            continue
        std = float(np.std(sample))
        normalized[index] = (
            0.0 if std <= 1e-12 else (values[index] - float(np.mean(sample))) / std
        )
        normalized_valid[index] = True
        normalized_available_at[index] = np.max(
            available_at[start : index + 1][sample_mask]
        )
    return normalized, normalized_valid, normalized_available_at


def align_native_feature(
    spec: FeatureSpec,
    raw: RawMarketSeries,
    contract: InstrumentContract,
    base_timestamps: np.ndarray,
    base_active: np.ndarray,
    *,
    timeframe: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate on the native clock and causally align to the base clock."""

    _validate_regular_native_series(raw, timeframe)
    event_values, event_valid, event_available_at = _native_events(spec, raw, contract)
    values = np.zeros(len(base_timestamps), dtype=np.float64)
    available = np.zeros(len(base_timestamps), dtype=np.bool_)
    age_hours = np.full(
        len(base_timestamps),
        spec.max_staleness_hours,
        dtype=np.float64,
    )
    staleness = np.ones(len(base_timestamps), dtype=np.float64)

    valid_indices = np.flatnonzero(event_valid)
    if valid_indices.size == 0:
        return values, available, age_hours, staleness

    event_time_ns = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
    availability_ns = event_available_at.astype("datetime64[ns]").astype(np.int64)
    order = valid_indices[np.argsort(availability_ns[valid_indices], kind="stable")]
    base_ns = base_timestamps.astype("datetime64[ns]").astype(np.int64)

    cursor = 0
    latest_index: int | None = None
    for base_index, timestamp_ns in enumerate(base_ns):
        while cursor < len(order) and availability_ns[order[cursor]] <= timestamp_ns:
            candidate = int(order[cursor])
            if (
                latest_index is None
                or event_time_ns[candidate] > event_time_ns[latest_index]
            ):
                latest_index = candidate
            cursor += 1
        if latest_index is None or not base_active[base_index]:
            continue
        age = float(timestamp_ns - event_time_ns[latest_index]) / _NS_PER_HOUR
        if age < -1e-12:
            raise ValueError("native feature event appears before its event timestamp")
        normalized_staleness = min(age / spec.max_staleness_hours, 1.0)
        age_hours[base_index] = age
        staleness[base_index] = normalized_staleness
        if age <= spec.max_staleness_hours + 1e-12:
            values[base_index] = event_values[latest_index]
            available[base_index] = True

    return values, available, age_hours, staleness


__all__ = ["align_native_feature"]
