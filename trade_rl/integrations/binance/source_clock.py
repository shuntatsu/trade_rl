"""Versioned raw Binance candle finality admission (opt-in new Dataset only)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

_KlineRow = Sequence[object]


class BinanceSourceClosePolicy(StrEnum):
    """Versioned admission of Binance source bar finality for new builds only."""

    RECONSTRUCTED_V1 = "reconstructed_v1"
    STRICT_SOURCE_CLOSE_V2 = "strict_source_close_v2"


def _native_timestamp(value: object) -> int:
    # Strict v2 accepts the exact integer fields supplied in Binance klines.
    # The generic legacy normalizer deliberately accepts/coerces more formats.
    raw = str(value)
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError("Binance source close_time requires integer native timestamps")
    return int(raw)


def _check_source_close_time(
    row: _KlineRow,
    *,
    open_ms: int,
    interval_ms: int,
    source_close_policy: BinanceSourceClosePolicy,
) -> None:
    if source_close_policy is BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2:
        # Source formats use millisecond time fields or (SPOT Vision from
        # 2025-01-01, and opt-in REST) microsecond fields. Converting the raw
        # close to ms first discards up to 999 microseconds of invalidity.
        # The raw open must agree exactly with the already-normalized grid;
        # native units cannot be mixed within the same kline row.
        source_open = _native_timestamp(row[0])
        source_close = _native_timestamp(row[6])
        if source_open == open_ms:
            unit = 1
        elif source_open == open_ms * 1000:
            unit = 1000
        else:
            raise ValueError("Binance source close_time has invalid native open time")
        expected_close = (open_ms + interval_ms) * unit - 1
        if source_close != expected_close:
            raise ValueError(
                "Binance source close_time differs from expected interval end"
            )
