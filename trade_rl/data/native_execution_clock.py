"""Exact fixed native execution clock encoding; legacy integer clocks unchanged."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from trade_rl.data._timestamps import _MAX_DATETIME64_NS, _MIN_FINITE_DATETIME64_NS
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind

_DAY_SECONDS = 86400


def fixed_execution_frequency(clock: NativeBarClock) -> Fraction:
    """Phase-one coarse execution admits explicit 3d and 7d/UTC-week only."""
    if type(clock) is not NativeBarClock:
        raise ValueError("expected a native execution clock")
    if clock.kind is NativeClockKind.UTC_WEEK:
        return Fraction(365, 7)
    if clock.kind is NativeClockKind.FIXED_ELAPSED and clock.fixed_seconds in (
        3 * _DAY_SECONDS,
        7 * _DAY_SECONDS,
    ):
        assert clock.fixed_seconds is not None
        return Fraction(365 * _DAY_SECONDS, clock.fixed_seconds)
    raise ValueError(
        "native execution currently supports 3d/7d fixed clocks and UTC weeks"
    )


def calendar_month_execution_frequency(
    clock: NativeBarClock, closes: np.ndarray
) -> Fraction:
    """Exact observed ACT/365F frequency of a complete calendar-month grid."""
    calendar_month_clock_payload(clock)
    _, ends = clock.interval_bounds(closes)
    return len(ends) / clock.elapsed_years(ends)


def calendar_month_clock_payload(clock: NativeBarClock) -> dict[str, object]:
    if type(clock) is not NativeBarClock or clock.kind is not NativeClockKind.UTC_MONTH:
        raise ValueError("expected an explicit UTC calendar month execution clock")
    return {
        "kind": NativeClockKind.UTC_MONTH.value,
        "fixed_seconds": None,
        "fixed_anchor_ns": None,
    }


def calendar_month_clock_from_payload(value: object) -> NativeBarClock:
    clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    if (
        type(value) is not dict
        or type(value.get("kind")) is not str
        or value != calendar_month_clock_payload(clock)
    ):
        raise ValueError("invalid calendar month execution clock payload")
    return clock


def frequency_payload(value: Fraction) -> dict[str, int]:
    if type(value) is not Fraction or value <= 0:
        raise ValueError("expected positive exact annual frequency")
    try:
        finite = math.isfinite(float(value)) and float(value) > 0
    except OverflowError as exc:
        raise ValueError("annual frequency is not representable") from exc
    if not finite:
        raise ValueError("annual frequency is not representable")
    return {"numerator": value.numerator, "denominator": value.denominator}


def frequency_from_payload(value: object) -> Fraction:
    if type(value) is not dict or set(value) != {"numerator", "denominator"}:
        raise ValueError("expected exact annual frequency payload")
    numerator, denominator = value["numerator"], value["denominator"]
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or not 0 < numerator <= 2**63 - 1
        or not 0 < denominator <= 2**63 - 1
    ):
        raise ValueError("invalid exact annual frequency components")
    result = Fraction(numerator, denominator)
    if frequency_payload(result) != value:
        raise ValueError("annual frequency payload must be reduced")
    return result


def execution_clock_payload(clock: NativeBarClock) -> dict[str, object]:
    fixed_execution_frequency(clock)
    return {
        "kind": clock.kind.value,
        "fixed_seconds": clock.fixed_seconds,
        "fixed_anchor_ns": None
        if clock.fixed_anchor is None
        else int(clock.fixed_anchor.astype(np.int64)),
    }


def execution_clock_from_payload(value: object) -> NativeBarClock:
    if type(value) is not dict or set(value) != {
        "kind",
        "fixed_seconds",
        "fixed_anchor_ns",
    }:
        raise ValueError("invalid native execution clock payload")
    if type(value["kind"]) is not str:
        raise ValueError("invalid native clock kind")
    kind = NativeClockKind(value["kind"])
    seconds, anchor = value["fixed_seconds"], value["fixed_anchor_ns"]
    if kind is NativeClockKind.FIXED_ELAPSED:
        if (
            type(seconds) is not int
            or type(anchor) is not int
            or not _MIN_FINITE_DATETIME64_NS <= anchor <= _MAX_DATETIME64_NS
        ):
            raise ValueError("invalid fixed clock duration or anchor")
        clock = NativeBarClock(kind, seconds, np.datetime64(anchor, "ns"))
    else:
        if seconds is not None or anchor is not None:
            raise ValueError("calendar clock cannot contain a fixed anchor")
        clock = NativeBarClock(kind)
    if execution_clock_payload(clock) != value:
        raise ValueError("invalid native execution clock encoding")
    return clock


__all__ = [
    "calendar_month_clock_from_payload",
    "calendar_month_clock_payload",
    "calendar_month_execution_frequency",
    "execution_clock_from_payload",
    "execution_clock_payload",
    "fixed_execution_frequency",
    "frequency_from_payload",
    "frequency_payload",
]
