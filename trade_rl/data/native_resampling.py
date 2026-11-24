"""Complete native-bar aggregation with explicit source and target clocks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data._timestamps import _datetime64_values_to_ns_exact
from trade_rl.data.native_clock import NativeBarClock


@dataclass(frozen=True, slots=True)
class NativeResampledBars:
    """Derived bars, not provider-native candle finality evidence.

    OHLCV columns: open, high, low, close, base volume, quote volume.
    Each availability is the latest availability of all its source members.
    """

    closes: np.ndarray
    available_at: np.ndarray
    ohlcv: np.ndarray
    member_counts: np.ndarray


def resample_complete_native_bars(
    *,
    source_clock: NativeBarClock,
    source_closes: np.ndarray,
    source_available_at: np.ndarray,
    source_ohlcv: np.ndarray,
    target_clock: NativeBarClock,
    target_closes: np.ndarray,
    as_of: np.datetime64,
) -> NativeResampledBars:
    """Aggregate exactly the requested complete intervals without interpolation.

    Both clocks and the requested target grid are explicit. Every source bar
    must lie in exactly one requested interval, with no missing or crossing
    members. Source timestamps must already represent final published candles;
    the caller owns publication authenticity, source identity and permissions.
    This validates timing and numerical consistency, not venue finality.
    Coarse OHLCV does not imply a finer execution path or authorize replay.
    """
    source_starts, source_ends = source_clock.interval_bounds(source_closes)
    target_starts, target_ends = target_clock.interval_bounds(target_closes)
    available = _datetime64_values_to_ns_exact(np.asarray(source_available_at))
    if not isinstance(as_of, np.datetime64):
        raise ValueError("as_of must be a datetime64 scalar")
    cutoff = _datetime64_values_to_ns_exact(np.asarray(as_of))[()]
    if (
        available.shape != source_ends.shape
        or np.any(available < source_ends)
        or np.any(available > cutoff)
    ):
        raise ValueError("source availability must be between close and as_of")
    if source_starts[0] != target_starts[0] or source_ends[-1] != target_ends[-1]:
        raise ValueError("source coverage must match the requested target range")

    # Every target endpoint must coincide exactly with a source endpoint.
    # Searchsorted avoids repeated scans for long histories.
    stop = np.searchsorted(source_ends, target_ends, side="right")
    if np.any(stop == 0) or not np.array_equal(source_ends[stop - 1], target_ends):
        raise ValueError("source bars cross a requested target boundary")
    start = np.concatenate((np.array([0], dtype=stop.dtype), stop[:-1]))
    if np.any(stop <= start) or not np.array_equal(source_starts[start], target_starts):
        raise ValueError("target intervals must contain complete source members")

    raw = np.asarray(source_ohlcv)
    if raw.shape != (source_ends.size, 6) or raw.dtype.kind not in "iuf":
        raise ValueError("source OHLCV must be a real numeric N-by-6 array")
    # Reject precision loss at admission rather than silently rounding source
    # integer or extended-precision OHLCV before checking its invariants.
    if raw.dtype.kind == "f" and raw.dtype.itemsize > 8:
        raise ValueError("source OHLCV precision exceeds float64")
    if raw.dtype.kind in "iu" and (np.any(raw > 2**53) or np.any(raw < -(2**53))):
        raise ValueError("source OHLCV integer exceeds exact float64 range")
    values = raw.astype(np.float64)
    if (
        not np.all(np.isfinite(values))
        or np.any(values[:, :4] <= 0)
        or np.any(values[:, 4:] < 0)
        or np.any(values[:, 1] < np.maximum(values[:, 0], values[:, 3]))
        or np.any(values[:, 2] > np.minimum(values[:, 0], values[:, 3]))
    ):
        raise ValueError("invalid finite source OHLCV")
    result = np.empty((target_ends.size, 6), dtype=np.float64)
    publication = np.empty_like(target_ends)
    for i, (lo, hi) in enumerate(zip(start, stop, strict=True)):
        members = values[lo:hi]
        try:
            base_volume = math.fsum(members[:, 4])
            quote_volume = math.fsum(members[:, 5])
        except OverflowError as exc:
            raise ValueError("aggregate volume exceeds finite representation") from exc
        result[i] = (
            members[0, 0],
            np.max(members[:, 1]),
            np.min(members[:, 2]),
            members[-1, 3],
            base_volume,
            quote_volume,
        )
        publication[i] = np.max(available[lo:hi])
    return NativeResampledBars(target_ends.copy(), publication, result, stop - start)
