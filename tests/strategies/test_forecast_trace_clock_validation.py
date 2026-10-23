from __future__ import annotations

import hashlib
import json
import sys
from datetime import date

import numpy as np
import pytest

from trade_rl.strategies.forecasts import training_trace
from trade_rl.strategies.forecasts.training_trace import ForecastTrainingTrace

_CLOCK_FIELDS = (
    "start_times",
    "end_times",
    "start_available_at",
    "end_available_at",
)
_NS = 1_000_000_000
_DAY_NS = 86_400 * _NS
_EPOCH = date(1970, 1, 1)


def _trace(
    clocks: tuple[np.ndarray, ...],
    *,
    row_symbols: tuple[str, ...] | None = None,
    start_close: list[float] | None = None,
    end_close: list[float] | None = None,
) -> ForecastTrainingTrace:
    count = len(clocks[0])
    symbols = row_symbols or tuple(f"S{index}" for index in range(count))
    starts = start_close or [100.0 + index for index in range(count)]
    ends = end_close or [110.0 + index for index in range(count)]
    return ForecastTrainingTrace(
        row_symbols=symbols,
        start_times=clocks[0],
        end_times=clocks[1],
        start_available_at=clocks[2],
        end_available_at=clocks[3],
        start_close=np.asarray(starts, dtype=np.float64),
        end_close=np.asarray(ends, dtype=np.float64),
    )


def _clock(values: list[int], unit: str) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).astype(f"datetime64[{unit}]")


def _fixed_ns(values: list[int], multiplier: int, unit_ns: int) -> list[int]:
    return [int(value) * multiplier * unit_ns for value in values]


def _month_ns(month_offset: int) -> int:
    month_index = int(month_offset)
    month_date = date(1970 + month_index // 12, month_index % 12 + 1, 1)
    return (month_date - _EPOCH).days * _DAY_NS


def _year_ns(year_offset: int) -> int:
    return (date(1970 + int(year_offset), 1, 1) - _EPOCH).days * _DAY_NS


def test_trace_payload_and_digest_match_independent_clock_oracle() -> None:
    start_raw = [-1, 0, 1, 2]
    end_raw = [2, 3, 4, 5]
    start_available_raw = [1, 2, 3, 4]
    end_available_raw = [1, 2, 3, 4]
    trace = _trace(
        (
            _clock(start_raw, "3h"),
            _clock(end_raw, "2D"),
            _clock(start_available_raw, "M"),
            _clock(end_available_raw, "2Y"),
        ),
        row_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"),
        start_close=[100.25, 101.5, 102.75, 104.0],
        end_close=[110.5, 111.75, 113.0, 114.25],
    )
    expected = {
        "schema": "forecast_training_trace_v1",
        "row_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"],
        "start_times": _fixed_ns(start_raw, 3, 3_600 * _NS),
        "end_times": _fixed_ns(end_raw, 2, _DAY_NS),
        "start_available_at": [_month_ns(value) for value in start_available_raw],
        "end_available_at": [_year_ns(value * 2) for value in end_available_raw],
        "start_close": [100.25, 101.5, 102.75, 104.0],
        "end_close": [110.5, 111.75, 113.0, 114.25],
    }
    assert trace.payload() == expected
    expected_bytes = json.dumps(
        expected,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert trace.digest == hashlib.sha256(expected_bytes).hexdigest()


@pytest.mark.parametrize(
    ("unit", "multiplier", "unit_ns"),
    (
        ("W", 1, 7 * _DAY_NS),
        ("D", 1, _DAY_NS),
        ("h", 1, 3_600 * _NS),
        ("m", 1, 60 * _NS),
        ("s", 1, _NS),
        ("ms", 1, 1_000_000),
        ("us", 1, 1_000),
        ("ns", 1, 1),
        ("2D", 2, _DAY_NS),
        ("3h", 3, 3_600 * _NS),
        ("5ms", 5, 1_000_000),
    ),
)
def test_fixed_units_use_declared_integer_tick_factors(
    unit: str, multiplier: int, unit_ns: int
) -> None:
    raw = ([-3, -2, -1], [-2, -1, 0], [-1, 0, 1], [0, 1, 2])
    trace = _trace(tuple(_clock(values, unit) for values in raw))
    payload = trace.payload()
    for field, values in zip(_CLOCK_FIELDS, raw, strict=True):
        assert payload[field] == _fixed_ns(values, multiplier, unit_ns)


@pytest.mark.parametrize(
    ("unit", "divisor"), (("ps", 10**3), ("fs", 10**6), ("as", 10**9))
)
def test_aligned_subnanosecond_clocks_are_exact(unit: str, divisor: int) -> None:
    raw_ns = ([-3, -2, -1], [-2, -1, 0], [-1, 0, 1], [0, 1, 2])
    trace = _trace(
        tuple(_clock([value * divisor for value in values], unit) for values in raw_ns)
    )
    payload = trace.payload()
    for field, values in zip(_CLOCK_FIELDS, raw_ns, strict=True):
        assert payload[field] == values


@pytest.mark.parametrize("field", _CLOCK_FIELDS)
@pytest.mark.parametrize(
    ("nat_first", "message"),
    ((False, "exactly representable"), (True, "non-NaT")),
)
def test_mixed_nat_and_lossy_clocks_keep_first_invalid_error(
    field: str, nat_first: bool, message: str
) -> None:
    values = _clock([1, 2_000, 3_000], "ps")
    values[0 if nat_first else 1] = np.datetime64("NaT", "ps")
    values[1 if nat_first else 0] = np.datetime64(1, "ps")
    clocks = [
        _clock([-3, -2, -1], "h"),
        _clock([-2, -1, 0], "h"),
        _clock([-1, 0, 1], "h"),
        _clock([0, 1, 2], "h"),
    ]
    clocks[_CLOCK_FIELDS.index(field)] = values
    with pytest.raises(ValueError, match=message):
        _trace(tuple(clocks))


def test_valid_nanosecond_int64_endpoints_are_preserved() -> None:
    minimum, maximum = -9_223_372_036_854_775_807, 9_223_372_036_854_775_807
    clocks = (
        np.asarray([minimum, maximum - 2], dtype=np.int64).astype("datetime64[ns]"),
        np.asarray([minimum + 1, maximum - 1], dtype=np.int64).astype("datetime64[ns]"),
        np.asarray([minimum, maximum], dtype=np.int64).astype("datetime64[ns]"),
        np.asarray([minimum + 1, maximum], dtype=np.int64).astype("datetime64[ns]"),
    )
    payload = _trace(clocks).payload()
    assert payload["start_times"] == [minimum, maximum - 2]
    assert payload["end_times"] == [minimum + 1, maximum - 1]
    assert payload["start_available_at"] == [minimum, maximum]
    assert payload["end_available_at"] == [minimum + 1, maximum]


@pytest.mark.parametrize("field", _CLOCK_FIELDS)
@pytest.mark.parametrize("seconds", (-9_223_372_037, 9_223_372_037))
def test_coarse_clock_values_that_wrap_outside_nanosecond_range_reject(
    field: str, seconds: int
) -> None:
    clocks = [
        _clock([-4, -2], "s"),
        _clock([-3, -1], "s"),
        _clock([-3, 0], "s"),
        _clock([-2, 0], "s"),
    ]
    clocks[_CLOCK_FIELDS.index(field)] = _clock([seconds, seconds], "s")
    with pytest.raises(ValueError, match="exactly representable"):
        _trace(tuple(clocks))


def test_strided_timestamp_input_is_not_mutated_or_retained() -> None:
    backing = _clock([-99, -3, -99, -2, -99, -1], "h")
    source = backing[1::2]
    before = backing.copy()
    assert not source.flags.c_contiguous
    trace = _trace(
        (
            source,
            _clock([0, 1, 2], "s"),
            _clock([0, 2, 4], "s"),
            _clock([0, 3, 5], "s"),
        )
    )
    np.testing.assert_array_equal(backing, before)
    assert trace.payload()["start_times"] == [
        -3 * 3_600 * _NS,
        -2 * 3_600 * _NS,
        -3_600 * _NS,
    ]
    backing[1] = np.datetime64(7, "h")
    assert trace.payload()["start_times"][0] == -3 * 3_600 * _NS


def test_mutating_reused_input_array_is_revalidated_on_next_construction() -> None:
    starts = _clock([-3, -2, -1], "h")
    clocks = (
        starts,
        _clock([-2, -1, 0], "h"),
        _clock([-1, 0, 1], "h"),
        _clock([0, 1, 2], "h"),
    )
    first = _trace(clocks)
    starts[1] = np.datetime64("NaT", "h")
    assert first.payload()["start_times"][1] == -2 * 3_600 * _NS
    with pytest.raises(ValueError, match="non-NaT"):
        _trace(clocks)


def test_constructor_uses_bulk_clock_validation_for_long_traces() -> None:
    count = 64
    start = np.datetime64("2000-01-01", "ns") + np.arange(count) * np.timedelta64(
        1, "s"
    )
    end = start + np.timedelta64(1, "s")
    clocks = (start, end, start.copy(), end.copy())
    timestamp_code = training_trace._timestamp.__code__
    scalar_calls = 0

    def profile(frame, event, _arg):
        nonlocal scalar_calls
        if event == "call" and frame.f_code is timestamp_code:
            scalar_calls += 1

    previous_profile = sys.getprofile()
    try:
        sys.setprofile(profile)
        trace = _trace(clocks)
    finally:
        sys.setprofile(previous_profile)
    assert len(trace.row_symbols) == count
    assert scalar_calls <= 4, (
        f"expected at most 4 scalar clock dispatches, saw {scalar_calls}"
    )
