"""As-of alignment of completed native feature events onto a decision clock."""

from __future__ import annotations

import math

import numpy as np

_NS_PER_HOUR = 3_600_000_000_000


def align_native_events_asof(
    event_times: np.ndarray,
    event_values: np.ndarray,
    event_valid: np.ndarray,
    event_available_at: np.ndarray,
    base_timestamps: np.ndarray,
    base_active: np.ndarray,
    *,
    max_staleness_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prefer the newest *arrived* native event, never a later completed event.

    event_times and event_available_at are independent clock fields. Delayed
    arrivals never travel back to an earlier base decision. A missing event
    returns zero with an unavailable mask and unit normalized staleness.
    """
    event_times = np.asarray(event_times, dtype="datetime64[ns]")
    available_at = np.asarray(event_available_at, dtype="datetime64[ns]")
    values = np.asarray(event_values, dtype=np.float64)
    valid = np.asarray(event_valid, dtype=np.bool_)
    base = np.asarray(base_timestamps, dtype="datetime64[ns]")
    active = np.asarray(base_active, dtype=np.bool_)
    if (
        values.ndim != 2
        or event_times.shape != (len(values),)
        or available_at.shape != event_times.shape
        or valid.shape != event_times.shape
        or base.ndim != 1
        or active.shape != base.shape
    ):
        raise ValueError("native as-of event arrays have incompatible shapes")
    if not math.isfinite(max_staleness_hours) or max_staleness_hours <= 0.0:
        raise ValueError("native as-of maximum staleness must be positive")
    event_ns = event_times.astype(np.int64)
    arrival_ns = available_at.astype(np.int64)
    base_ns = base.astype(np.int64)
    if (
        np.any(np.diff(event_ns) <= 0)
        or np.any(np.diff(base_ns) <= 0)
        or np.any(valid & (arrival_ns < event_ns))
        or np.any(valid & (arrival_ns == np.iinfo(np.int64).min))
    ):
        raise ValueError("native events violate timestamp or publication order")

    result = np.zeros((len(base), values.shape[1]), dtype=np.float64)
    available = np.zeros(result.shape, dtype=np.bool_)
    age = np.full(result.shape, max_staleness_hours, dtype=np.float64)
    staleness = np.ones(result.shape, dtype=np.float32)
    indices = np.flatnonzero(valid)
    ordered = indices[np.argsort(arrival_ns[indices], kind="stable")]
    cursor = 0
    latest: int | None = None
    for base_index, now in enumerate(base_ns):
        while cursor < len(ordered) and arrival_ns[ordered[cursor]] <= now:
            candidate = int(ordered[cursor])
            if latest is None or event_ns[candidate] > event_ns[latest]:
                latest = candidate
            cursor += 1
        if latest is None or not active[base_index]:
            continue
        elapsed = float(now - event_ns[latest]) / _NS_PER_HOUR
        if elapsed < -1e-12:
            raise ValueError("feature source is later than its decision timestamp")
        age[base_index] = elapsed
        staleness[base_index] = min(elapsed / max_staleness_hours, 1.0)
        if elapsed <= max_staleness_hours + 1e-12:
            result[base_index] = values[latest]
            available[base_index] = True
    return result, available, age, staleness


__all__ = ["align_native_events_asof"]
