"""Causal native-timeframe feature calculation and base-clock alignment."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import numpy as np

from trade_rl.data.contracts import FeatureSpec, InstrumentContract, timeframe_hours
from trade_rl.data.features import calculate_feature_events
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


def _native_events(
    spec: FeatureSpec,
    raw: RawMarketSeries,
    contract: InstrumentContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = _active_mask(raw, contract)
    assert raw.funding_available is not None
    values, valid, source_start = calculate_feature_events(
        spec,
        open_price=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
        funding_rate=raw.funding_rate,
        funding_available=raw.funding_available,
        row_present=np.ones(raw.timestamps.shape, dtype=np.bool_),
        active=active,
    )
    assert raw.available_at is not None
    return (
        values,
        valid,
        _dependency_available_at(raw.available_at, valid, source_start),
    )


def _dependency_available_at(
    raw_available_at: np.ndarray,
    valid: np.ndarray,
    source_start: np.ndarray,
) -> np.ndarray:
    """Exact range maxima; linear for the maintained nondecreasing starts.

    RawMarketSeries has already excluded NaT and normalized timestamps to ns.
    Compare integer timestamps to avoid repeated datetime conversion. A decreasing
    dependency start requires rebuilding the queue, not forgetting older bars.
    """

    available_at = np.full(
        len(raw_available_at), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    raw_ns = raw_available_at.view(np.int64)
    result_ns = available_at.view(np.int64)
    candidates: deque[int] = deque()
    cursor = 0
    previous_start = 0
    for event in np.flatnonzero(valid):
        index = int(event)
        start = int(source_start[index])
        if start < 0 or start > index:
            raise ValueError("feature source range is invalid")
        if start < previous_start:
            candidates.clear()
            cursor = start
        while candidates and candidates[0] < start:
            candidates.popleft()
        cursor = max(cursor, start)
        while cursor <= index:
            while candidates and raw_ns[candidates[-1]] <= raw_ns[cursor]:
                candidates.pop()
            candidates.append(cursor)
            cursor += 1
        result_ns[index] = raw_ns[candidates[0]]
        previous_start = start
    return available_at


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
        len(base_timestamps), spec.max_staleness_hours, dtype=np.float64
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
