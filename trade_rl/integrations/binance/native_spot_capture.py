"""Admission of already captured Binance Spot UTC combined kline messages.

This module performs no subscription, network access, historical lookup or
authorization. The caller must establish the Spot endpoint and a trustworthy
receipt clock; the same payload shape can occur on other venues or be forged.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from decimal import Decimal
from typing import Literal, NoReturn

import numpy as np

from trade_rl.data._timestamps import _datetime64_values_to_ns_exact
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.native_resampling import (
    NativeResampledBars,
    resample_complete_native_bars,
)
from trade_rl.integrations.binance.native_kline_clock import validate_native_kline_times
from trade_rl.integrations.binance.vision import binance_interval_milliseconds

_MAX_MESSAGE_BYTES = 64 * 1024
_DAY_NS = 86_400_000_000_000


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field in captured kline")
        result[key] = value
    return result


def _invalid_constant(value: str) -> NoReturn:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(
            "captured kline timestamps must be native nonnegative integers"
        )
    return value


def _number(value: object) -> tuple[Decimal, float]:
    # Spot encodes these decimal fields as strings. Reject coercion, nonfinite
    # values and loss of a nonzero amount to zero before native float64 admission.
    if type(value) is not str or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) is None:
        raise ValueError("captured kline OHLCV must be unsigned decimal text")
    exact = Decimal(value)
    number = float(value)
    if not math.isfinite(number) or (number == 0 and exact != 0):
        raise ValueError("captured kline OHLCV is outside finite float64 range")
    return exact, number


def _validate_interval(clock: NativeBarClock, interval: str) -> None:
    if interval in ("1w", "1M"):
        expected = (
            NativeClockKind.UTC_WEEK if interval == "1w" else NativeClockKind.UTC_MONTH
        )
        if clock.kind is not expected:
            raise ValueError("calendar stream must use its matching UTC clock")
        return
    seconds = (
        3 * 86400
        if interval == "3d"
        else binance_interval_milliseconds(interval) // 1000
    )
    if (
        clock.kind is not NativeClockKind.FIXED_ELAPSED
        or clock.fixed_seconds != seconds
    ):
        raise ValueError("stream interval differs from declared fixed clock")
    assert clock.fixed_anchor is not None
    anchor_ns = int(clock.fixed_anchor.astype(np.int64))
    # A multi-day phase must be declared by the caller; it is not inferred
    # from the first received row. All admitted Spot intervals use UTC.
    phase_ns = _DAY_NS if interval == "3d" else seconds * 1_000_000_000
    if anchor_ns % phase_ns:
        raise ValueError("fixed stream anchor is not UTC-aligned")


def resample_captured_spot_klines(
    *,
    captures: Sequence[tuple[bytes, np.datetime64]],
    symbol: str,
    source_interval: str,
    source_clock: NativeBarClock,
    source_closes: np.ndarray,
    timestamp_unit: Literal["ms", "us"],
    target_clock: NativeBarClock,
    target_closes: np.ndarray,
    as_of: np.datetime64,
) -> NativeResampledBars:
    """Admit one final UTC Spot message per requested source bar, then aggregate.

    The input order is the requested bar order (not arrival order). Combined
    stream identity, symbol, interval, explicit boolean closure, raw inclusive
    close and event/receipt ordering must agree. Availability is the observed
    receipt, never the scheduled close or the provider event time. The source
    range and output range are explicit and must be complete; the shared
    resampler owns OHLCV and member validation. For native bars use the same
    source and target clock/range.

    Only previously captured Spot combined streams with explicit timestamp
    unit are admitted. REST/Vision rows and UTC-offset streams are not finality
    evidence here. The caller must independently authenticate the connection,
    preserve the original message/receipt bytes and establish clock integrity,
    data permission and any funding/execution context. No Dataset or legacy
    source contract is changed.
    """
    if type(symbol) is not str or re.fullmatch(r"[A-Z0-9]+", symbol) is None:
        raise ValueError("expected uppercase Spot symbol")
    if timestamp_unit not in ("ms", "us"):
        raise ValueError("capture timestamp unit must be ms or us")
    if (
        type(source_clock) is not NativeBarClock
        or type(target_clock) is not NativeBarClock
    ):
        raise ValueError("source and target clocks must be NativeBarClock values")
    _validate_interval(source_clock, source_interval)
    _, closes = source_clock.interval_bounds(source_closes)
    if len(captures) != closes.size:
        raise ValueError("captures must cover the entire requested source grid")

    tick_ns = 1_000_000 if timestamp_unit == "ms" else 1_000
    expected_stream = f"{symbol.lower()}@kline_{source_interval}"
    available = np.empty(closes.shape, dtype="datetime64[ns]")
    rows: list[list[object]] = []
    ohlcv = np.empty((closes.size, 6), dtype=np.float64)
    for i, (raw, received) in enumerate(captures):
        if type(raw) is not bytes or not 0 < len(raw) <= _MAX_MESSAGE_BYTES:
            raise ValueError("captured kline must be bounded UTF-8 JSON bytes")
        if not isinstance(received, np.datetime64):
            raise ValueError("capture receipt must be a datetime64 scalar")
        receipt = _datetime64_values_to_ns_exact(np.asarray(received))[()]
        try:
            message = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_invalid_constant,
            )
        except (UnicodeError, RecursionError) as exc:
            raise ValueError("invalid captured kline JSON") from exc
        if not isinstance(message, dict) or message.get("stream") != expected_stream:
            raise ValueError("captured kline combined stream mismatch")
        data = message.get("data")
        if (
            not isinstance(data, dict)
            or data.get("e") != "kline"
            or data.get("s") != symbol
        ):
            raise ValueError("captured kline event or symbol mismatch")
        kline = data.get("k")
        if (
            not isinstance(kline, dict)
            or kline.get("s") != symbol
            or kline.get("i") != source_interval
            or kline.get("x") is not True
        ):
            raise ValueError(
                "captured kline is not the requested final symbol/interval"
            )
        start = _integer(kline.get("t"))
        end = _integer(kline.get("T"))
        event = _integer(data.get("E"))
        if event < end or event * tick_ns > int(receipt.astype(np.int64)):
            raise ValueError(
                "capture event must follow inclusive close and precede receipt"
            )
        parsed = [_number(kline.get(key)) for key in ("o", "h", "l", "c", "v", "q")]
        op, hi, lo, cl = (pair[0] for pair in parsed[:4])
        # Check the provider decimals before float64 rounding can conceal
        # an inverted high/low. The shared resampler validates float OHLCV.
        if hi < max(op, cl) or lo > min(op, cl):
            raise ValueError("invalid exact captured OHLC range")
        values = [pair[1] for pair in parsed]
        rows.append([start, *values[:5], end, values[5]])
        ohlcv[i] = values
        available[i] = receipt

    _, admitted_closes, admitted_available = validate_native_kline_times(
        rows,
        clock=source_clock,
        expected_closes=source_closes,
        timestamp_unit=timestamp_unit,
        available_at=available,
        as_of=as_of,
    )
    return resample_complete_native_bars(
        source_clock=source_clock,
        source_closes=admitted_closes,
        source_available_at=admitted_available,
        source_ohlcv=ohlcv,
        target_clock=target_clock,
        target_closes=target_closes,
        as_of=as_of,
    )
