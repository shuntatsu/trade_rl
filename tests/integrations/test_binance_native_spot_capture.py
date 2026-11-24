"""Literal captured-stream fixtures: no venue, cache, or economic access."""

import json

import numpy as np
import pytest

from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.integrations.binance.native_spot_capture import (
    resample_captured_spot_klines,
)


def _clock():
    return NativeBarClock(
        NativeClockKind.FIXED_ELAPSED, 86400, np.datetime64("1970-01-01")
    )


def _message(start, close, *, interval="1d", unit="ms"):
    start_tick = int(np.datetime64(start, unit).astype(np.int64))
    close_tick = int(np.datetime64(close, unit).astype(np.int64))
    return {
        "stream": f"testusdt@kline_{interval}",
        "data": {
            "e": "kline",
            "E": close_tick,
            "s": "TESTUSDT",
            "k": {
                "t": start_tick,
                "T": close_tick - 1,
                "s": "TESTUSDT",
                "i": interval,
                "x": True,
                "o": "100",
                "h": "103",
                "l": "98",
                "c": "101",
                "v": "2.5",
                "q": "250",
            },
        },
    }


def _encode(message):
    return json.dumps(message).encode("utf-8")


def _one(**kwargs):
    msg = _message("2024-02-01", "2024-03-01", interval="1M")
    close = np.array(["2024-03-01"], dtype="datetime64[D]")
    month = NativeBarClock(NativeClockKind.UTC_MONTH)
    values = dict(
        captures=[(_encode(msg), np.datetime64("2024-03-01T00:00:02"))],
        symbol="TESTUSDT",
        source_interval="1M",
        source_clock=month,
        source_closes=close,
        timestamp_unit="ms",
        target_clock=month,
        target_closes=close,
        as_of=np.datetime64("2024-03-01T00:00:03"),
    )
    values.update(kwargs)
    return resample_captured_spot_klines(**values)


def test_month_capture_uses_receipt_not_event_or_scheduled_close():
    result = _one()
    np.testing.assert_array_equal(result.ohlcv, [[100, 103, 98, 101, 2.5, 250]])
    np.testing.assert_array_equal(result.member_counts, [1])
    assert result.available_at[0] == np.datetime64("2024-03-01T00:00:02")


def test_daily_to_leap_month_reuses_complete_aggregation():
    closes = np.arange("2024-02-02", "2024-03-02", dtype="datetime64[D]")
    captures = []
    for i, close in enumerate(closes):
        msg = _message(close - np.timedelta64(1, "D"), close)
        msg["data"]["k"]["h"] = str(103 + i)
        captures.append((_encode(msg), close + np.timedelta64(2, "s")))
    # A late early member is still the last input available to the aggregate.
    captures[0] = (captures[0][0], np.datetime64("2024-03-02"))
    result = _one(
        captures=captures,
        source_interval="1d",
        source_clock=_clock(),
        source_closes=closes,
        as_of=np.datetime64("2024-03-02"),
    )
    np.testing.assert_array_equal(result.ohlcv, [[100, 131, 98, 101, 72.5, 7250]])
    np.testing.assert_array_equal(result.member_counts, [29])
    assert result.available_at[0] == np.datetime64("2024-03-02")


@pytest.mark.parametrize("unit", ["ms", "us"])
def test_week_native_timestamp_unit(unit):
    week = NativeBarClock(NativeClockKind.UTC_WEEK)
    close = np.array(["2024-01-08"], dtype="datetime64[D]")
    msg = _message("2024-01-01", "2024-01-08", interval="1w", unit=unit)
    result = _one(
        captures=[(_encode(msg), np.datetime64("2024-01-08T00:00:01"))],
        source_interval="1w",
        source_clock=week,
        source_closes=close,
        timestamp_unit=unit,
        target_clock=week,
        target_closes=close,
        as_of=np.datetime64("2024-01-08T00:00:02"),
    )
    assert result.member_counts[0] == 1


@pytest.mark.parametrize(
    "path,value",
    [
        (("stream",), "testusdt@kline_1M@+08:00"),
        (("data", "s"), "OTHERUSDT"),
        (("data", "e"), "trade"),
        (("data", "k", "s"), "OTHERUSDT"),
        (("data", "k", "i"), "1d"),
        (("data", "k", "x"), False),
        (("data", "k", "x"), 1),
        (("data", "k", "x"), "true"),
        (("data", "k", "T"), 0),
        (("data", "k", "t"), True),
        (("data", "E"), 0),
        (("data", "E"), "1709251200000"),
        (("data", "k", "o"), "NaN"),
        (("data", "k", "h"), "99"),
        (("data", "k", "v"), "-1"),
        (("data", "k", "q"), 250),
    ],
)
def test_wrong_stream_identity_finality_or_value(path, value):
    msg = _message("2024-02-01", "2024-03-01", interval="1M")
    obj = msg
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value
    with pytest.raises(ValueError):
        _one(captures=[(_encode(msg), np.datetime64("2024-03-01T00:00:02"))])


@pytest.mark.parametrize(
    "receipt", ["2024-02-29T23:59:59.999", "2024-03-01T00:00:04", "NaT"]
)
def test_invalid_receipt(receipt):
    msg = _message("2024-02-01", "2024-03-01", interval="1M")
    with pytest.raises(ValueError):
        _one(captures=[(_encode(msg), np.datetime64(receipt))])


def test_event_after_receipt_refused():
    msg = _message("2024-02-01", "2024-03-01", interval="1M")
    msg["data"]["E"] += 3000
    with pytest.raises(ValueError):
        _one(captures=[(_encode(msg), np.datetime64("2024-03-01T00:00:02"))])


def test_duplicate_json_field_rejected():
    raw = _encode(_message("2024-02-01", "2024-03-01", interval="1M"))
    raw = raw.replace(b'"x": true', b'"x": false, "x": true')
    with pytest.raises(ValueError):
        _one(captures=[(raw, np.datetime64("2024-03-01T00:00:02"))])


@pytest.mark.parametrize("raw", [b"{}", b"\xff", b'{"stream": NaN}', b" " * 65537])
def test_malformed_message_rejected(raw):
    with pytest.raises(ValueError):
        _one(captures=[(raw, np.datetime64("2024-03-01T00:00:02"))])


def test_missing_or_extra_capture_rejected():
    msg = _message("2024-02-01", "2024-03-01", interval="1M")
    capture = (_encode(msg), np.datetime64("2024-03-01T00:00:02"))
    for captures in ([], [capture, capture]):
        with pytest.raises(ValueError):
            _one(captures=captures)


def test_fixed_duration_must_not_be_labelled_calendar_month():
    clock = NativeBarClock(
        NativeClockKind.FIXED_ELAPSED, 29 * 86400, np.datetime64("2024-02-01")
    )
    with pytest.raises(ValueError):
        _one(source_clock=clock)


def test_daily_wrong_phase_rejected_even_if_message_matches_clock():
    clock = NativeBarClock(
        NativeClockKind.FIXED_ELAPSED, 86400, np.datetime64("2024-02-01T08:00")
    )
    close = np.array(["2024-02-02T08:00"], dtype="datetime64[m]")
    msg = _message("2024-02-01T08:00", "2024-02-02T08:00")
    with pytest.raises(ValueError):
        _one(
            source_interval="1d",
            source_clock=clock,
            source_closes=close,
            captures=[(_encode(msg), np.datetime64("2024-02-02T08:00:01"))],
            target_clock=clock,
            target_closes=close,
        )


def test_anchored_three_day_native_bar():
    clock = NativeBarClock(
        NativeClockKind.FIXED_ELAPSED, 3 * 86400, np.datetime64("2024-02-01")
    )
    close = np.array(["2024-02-04"], dtype="datetime64[D]")
    msg = _message("2024-02-01", "2024-02-04", interval="3d")
    result = _one(
        captures=[(_encode(msg), np.datetime64("2024-02-04T00:00:01"))],
        source_interval="3d",
        source_clock=clock,
        source_closes=close,
        target_clock=clock,
        target_closes=close,
    )
    assert result.member_counts[0] == 1
    with pytest.raises(ValueError):
        _one(
            captures=[(_encode(msg), np.datetime64("2024-02-04T00:00:01"))],
            source_interval="3d",
            source_clock=NativeBarClock(
                NativeClockKind.FIXED_ELAPSED, 3 * 86400, np.datetime64("2024-02-02")
            ),
            source_closes=close,
            target_clock=clock,
            target_closes=close,
        )


@pytest.mark.parametrize(
    "volume", ["0." + "0" * 400 + "1", "9" * 400, "1_000", " 1", "1e3"]
)
def test_numeric_coercion_underflow_and_overflow_rejected(volume):
    msg = _message("2024-02-01", "2024-03-01", interval="1M")
    msg["data"]["k"]["v"] = volume
    with pytest.raises(ValueError):
        _one(captures=[(_encode(msg), np.datetime64("2024-03-01T00:00:02"))])


def test_inclusive_final_event_and_zero_trade_volume_allowed():
    msg = _message("2024-02-01", "2024-03-01", interval="1M")
    msg["data"]["E"] = msg["data"]["k"]["T"]
    msg["data"]["k"]["v"] = msg["data"]["k"]["q"] = "0"
    result = _one(captures=[(_encode(msg), np.datetime64("2024-03-01T00:00:02"))])
    np.testing.assert_array_equal(result.ohlcv[0, 4:], [0, 0])


def test_later_final_capture_does_not_change_earlier_week():
    clock = NativeBarClock(NativeClockKind.UTC_WEEK)
    closes = np.array(["2024-01-08", "2024-01-15"], dtype="datetime64[D]")
    messages = []
    for close in closes:
        msg = _message(close - np.timedelta64(7, "D"), close, interval="1w")
        messages.append((_encode(msg), close + np.timedelta64(1, "s")))
    common = dict(source_interval="1w", source_clock=clock, target_clock=clock)
    prefix = _one(
        **common,
        captures=messages[:1],
        source_closes=closes[:1],
        target_closes=closes[:1],
        as_of=np.datetime64("2024-01-09"),
    )
    later = _message("2024-01-08", "2024-01-15", interval="1w")
    later["data"]["k"]["h"] = "100000"
    whole = _one(
        **common,
        captures=[messages[0], (_encode(later), messages[1][1])],
        source_closes=closes,
        target_closes=closes,
    )
    np.testing.assert_array_equal(prefix.ohlcv[0], whole.ohlcv[0])
    assert prefix.available_at[0] == whole.available_at[0]
    for captures in ([messages[1], messages[0]], [messages[0], messages[0]]):
        with pytest.raises(ValueError):
            _one(
                **common, captures=captures, source_closes=closes, target_closes=closes
            )
