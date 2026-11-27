"""Causal native-timeframe feature calculation and base-clock alignment."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import cast

import numpy as np

from trade_rl.data._timestamps import _datetime64_values_to_ns_exact
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    timeframe_hours,
)
from trade_rl.data.features import calculate_feature_events
from trade_rl.data.native_clock import NativeBarClock, _native_unmasked_array
from trade_rl.data.source import RawMarketSeries

_NS_PER_HOUR = 3_600_000_000_000
# Bound temporary base-row arrays independently of the history length.
_ALIGNMENT_BATCH_ROWS = 8192


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
    *,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if active is None:
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
    return _align_events(
        spec, raw, _native_events(spec, raw, contract), base_timestamps, base_active
    )


def align_clocked_native_feature(
    spec: FeatureSpec,
    raw: RawMarketSeries,
    contract: InstrumentContract,
    base_timestamps: np.ndarray,
    base_active: np.ndarray,
    *,
    clock: NativeBarClock,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align price/volume features from an explicit complete native close grid.

    The clock is supplied separately: no legacy FeatureSpec timeframe or
    Dataset identity is changed. Bar-index windows keep their existing meaning;
    age and staleness are elapsed hours. A bar crossing the listing boundary
    cannot contribute to warm-up, and a decision at or after delisting cannot
    use stale features. Source identity, complete publication evidence and
    the clock must be bound externally before persisting or using a feature.
    This is not provider finality, cross-asset or funding-event admission.
    """
    if not isinstance(clock, NativeBarClock):
        raise ValueError("clock must be a NativeBarClock")
    if spec.timeframe is not None:
        raise ValueError("explicit clock requires no legacy feature timeframe")
    if spec.kind in {
        FeatureKind.FUNDING_BPS,
        FeatureKind.FUNDING_CHANGE,
        FeatureKind.FUNDING_ZSCORE,
        FeatureKind.RELATIVE_RETURN_TO_BTC,
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
        FeatureKind.ROLLING_BETA_TO_BTC,
        FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK,
        FeatureKind.CROSS_ASSET_DISPERSION,
    }:
        raise ValueError(
            "explicit native clock supports only single-instrument price/volume features"
        )
    starts, closes = clock.interval_bounds(raw.timestamps)
    base, active = _decision_grid(base_timestamps, base_active)
    listed, delisted = _clocked_lifetime(contract)
    native_active = _complete_lifetime_mask(starts, closes, listed, delisted)
    decision_active = active & (base >= listed)
    if delisted is not None:
        decision_active &= base < delisted
    return _align_events(
        spec,
        raw,
        _native_events(spec, raw, contract, active=native_active),
        base,
        decision_active,
    )


def _decision_grid(
    timestamps: np.ndarray, active_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    base = _datetime64_values_to_ns_exact(
        _native_unmasked_array(timestamps, field_name="base timestamps")
    )
    active = _native_unmasked_array(active_mask, field_name="base active")
    if base.ndim != 1 or not base.size or np.any(base[1:] <= base[:-1]):
        raise ValueError("base timestamps must be nonempty and strictly increasing")
    if active.dtype != np.dtype(np.bool_) or active.shape != base.shape:
        raise ValueError("base active must be a matching boolean array")
    return base, active


def _clocked_lifetime(
    contract: InstrumentContract,
) -> tuple[np.datetime64, np.datetime64 | None]:
    # Convert through microseconds to detect out-of-nanosecond-range lifetimes.
    def lifecycle_time(value: datetime) -> np.datetime64:
        utc = value.astimezone(timezone.utc).replace(tzinfo=None)
        return cast(
            np.datetime64,
            _datetime64_values_to_ns_exact(np.asarray(np.datetime64(utc, "us")))[()],
        )

    return (
        lifecycle_time(contract.listed_at),
        None if contract.delisted_at is None else lifecycle_time(contract.delisted_at),
    )


def _complete_lifetime_mask(
    starts: np.ndarray,
    closes: np.ndarray,
    listed: np.datetime64,
    delisted: np.datetime64 | None,
) -> np.ndarray:
    active = starts >= listed
    if delisted is not None:
        active &= closes < delisted
    return active


def _align_events(
    spec: FeatureSpec,
    raw: RawMarketSeries,
    events: tuple[np.ndarray, np.ndarray, np.ndarray],
    base_timestamps: np.ndarray,
    base_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _align_event_values(
        spec.max_staleness_hours, raw, events, base_timestamps, base_active
    )


def _align_event_values(
    max_staleness_hours: float,
    raw: RawMarketSeries,
    events: tuple[np.ndarray, np.ndarray, np.ndarray],
    base_timestamps: np.ndarray,
    base_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Multiple columns from a single native event share one publication frontier.
    event_values, event_valid, event_available_at = events
    values = np.zeros((len(base_timestamps), *event_values.shape[1:]), dtype=np.float64)
    available = np.zeros(len(base_timestamps), dtype=np.bool_)
    age_hours = np.full(len(base_timestamps), max_staleness_hours, dtype=np.float64)
    staleness = np.ones(len(base_timestamps), dtype=np.float64)

    valid_indices = np.flatnonzero(event_valid)
    if valid_indices.size == 0:
        return values, available, age_hours, staleness

    event_time_ns = raw.timestamps.astype("datetime64[ns]", copy=False).view(np.int64)
    availability_ns = event_available_at.astype("datetime64[ns]", copy=False).view(
        np.int64
    )
    order = valid_indices[np.argsort(availability_ns[valid_indices], kind="stable")]
    publication_ns = availability_ns[order]
    # RawMarketSeries close times are strictly increasing, so the largest
    # event index among the published prefix is the latest *close*, not
    # necessarily the most recently received event.
    latest_by_publication = np.maximum.accumulate(order)
    base_ns = base_timestamps.astype("datetime64[ns]", copy=False).view(np.int64)

    previous_frontier = np.iinfo(np.int64).min
    for begin in range(0, len(base_ns), _ALIGNMENT_BATCH_ROWS):
        end = min(begin + _ALIGNMENT_BATCH_ROWS, len(base_ns))
        times = base_ns[begin:end]
        # The legacy cursor never retreats, even for an inactive row.
        # Explicit-clock callers already validate strictly increasing times.
        frontier = np.maximum.accumulate(times)
        np.maximum(frontier, previous_frontier, out=frontier)
        previous_frontier = frontier[-1]
        ranks = np.searchsorted(publication_ns, frontier, side="right")
        active_rows = np.flatnonzero(
            (ranks > 0) & np.asarray(base_active[begin:end], dtype=np.bool_)
        )
        if not active_rows.size:
            continue
        selected = latest_by_publication[ranks[active_rows] - 1]
        observed = times[active_rows]
        event_close = event_time_ns[selected]
        # Subtract ordered signed timestamps in uint64: the mathematical
        # magnitude may exceed int64, including across the Unix epoch.
        # Convert that exact magnitude to float only after the subtraction.
        age = (
            np.maximum(observed, event_close).view(np.uint64)
            - np.minimum(observed, event_close).view(np.uint64)
        ).astype(np.float64)
        age[observed < event_close] *= -1.0
        age /= _NS_PER_HOUR
        if np.any(age < -1e-12):
            raise ValueError("native feature event appears before its event timestamp")
        rows = begin + active_rows
        age_hours[rows] = age
        with np.errstate(over="ignore"):
            staleness[rows] = np.minimum(age / max_staleness_hours, 1.0)
        fresh = age <= max_staleness_hours + 1e-12
        values[rows[fresh]] = event_values[selected[fresh]]
        available[rows[fresh]] = True

    return values, available, age_hours, staleness


__all__ = ["align_clocked_native_feature", "align_native_feature"]
