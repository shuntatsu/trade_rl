"""Opt-in native Binance kline time admission, without source acquisition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from trade_rl.data._timestamps import _datetime64_values_to_ns_exact
from trade_rl.data.native_clock import NativeBarClock, _native_unmasked_array
from trade_rl.integrations.binance.source_clock import _native_timestamp


def validate_native_kline_times(
    rows: Sequence[Sequence[object]],
    *,
    clock: NativeBarClock,
    expected_closes: np.ndarray,
    timestamp_unit: Literal["ms", "us"],
    available_at: np.ndarray,
    as_of: np.datetime64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Admit exact raw time fields against an explicitly requested native grid.

    Return half-open starts, exclusive closes and observed availability in ns.
    Require the entire requested range, rather than inferring it from returned
    rows. The timestamp unit and clock (including any fixed anchor) must come
    from the provider adapter, never from numeric magnitude or received rows.
    The raw inclusive close must be one native tick before the exclusive end.

    Availability must be supplied by the caller from independently established
    publication evidence. This checks its temporal consistency only: scheduled
    close fields do not establish candle finality, OHLC validity, source
    authenticity or data access authorization. No legacy source policy or
    Dataset admission is changed.
    """
    if type(clock) is not NativeBarClock:
        raise ValueError("clock must be a NativeBarClock")
    if timestamp_unit not in ("ms", "us"):
        raise ValueError("Binance native timestamp unit must be ms or us")
    tick_ns = 1_000_000 if timestamp_unit == "ms" else 1_000
    starts, closes = clock.interval_bounds(expected_closes)
    if len(rows) != closes.size:
        raise ValueError("Binance rows must cover the entire requested native grid")
    available = _datetime64_values_to_ns_exact(
        _native_unmasked_array(available_at, field_name="kline availability")
    )
    if available.shape != closes.shape:
        raise ValueError("publication timestamps must match the native close grid")
    if not isinstance(as_of, np.datetime64):
        raise ValueError("as_of must be a datetime64 scalar")
    cutoff = _datetime64_values_to_ns_exact(np.asarray(as_of))[()]
    if np.any(available < closes) or np.any(available > cutoff):
        raise ValueError(
            "publication must be at or after close and no later than as_of"
        )

    for row, start, end in zip(
        rows, starts.astype(np.int64), closes.astype(np.int64), strict=True
    ):
        if len(row) < 8:
            raise ValueError("Binance kline row must contain at least eight fields")
        start_tick, start_remainder = divmod(int(start), tick_ns)
        end_tick, end_remainder = divmod(int(end), tick_ns)
        if start_remainder or end_remainder:
            raise ValueError(
                "native grid cannot be represented in source timestamp unit"
            )
        raw_open, raw_close = row[0], row[6]
        if any(type(value) not in (str, int) for value in (raw_open, raw_close)):
            raise ValueError("Binance source time fields must be native integers")
        if _native_timestamp(raw_open) != start_tick:
            raise ValueError(
                "Binance source open differs from requested interval start"
            )
        if _native_timestamp(raw_close) != end_tick - 1:
            raise ValueError("Binance source close differs from expected inclusive end")
    return starts, closes, available
