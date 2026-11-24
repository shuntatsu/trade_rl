"""Literal synthetic native Binance time oracles; no source acquisition."""

from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.integrations.binance.native_kline_clock import validate_native_kline_times


def _dt(value: str) -> np.datetime64:
    return np.datetime64(value, "ns")


def _ticks(value: str, unit: str) -> int:
    # Independent datetime oracle, not the clock under test.
    days = datetime.fromisoformat(value).replace(tzinfo=UTC) - datetime(
        1970, 1, 1, tzinfo=UTC
    )
    return (days.days * 86400 + days.seconds) * (1000 if unit == "ms" else 1000000)


def _row(start: str, end: str, unit: str = "ms") -> list[object]:
    return [
        _ticks(start, unit),
        "1",
        "2",
        "1",
        "2",
        "3",
        str(_ticks(end, unit) - 1),
        "4",
    ]


def _validate(
    rows,
    closes,
    *,
    unit="ms",
    kind=NativeClockKind.UTC_MONTH,
    available=None,
    as_of="2024-06-01",
):
    closes = np.array(closes, dtype="datetime64[ns]")
    return validate_native_kline_times(
        rows,
        clock=NativeBarClock(kind),
        expected_closes=closes,
        timestamp_unit=unit,
        available_at=closes
        if available is None
        else np.array(available, dtype="datetime64[ns]"),
        as_of=_dt(as_of),
    )


@pytest.mark.parametrize("unit", ["ms", "us"])
def test_leap_months_native_unit(unit):
    dates = ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
    rows = [_row(a, b, unit) for a, b in zip(dates, dates[1:], strict=False)]
    starts, closes, available = _validate(rows, dates[1:], unit=unit)
    np.testing.assert_array_equal(starts, np.array(dates[:-1], dtype="datetime64[ns]"))
    np.testing.assert_array_equal(closes, np.array(dates[1:], dtype="datetime64[ns]"))
    np.testing.assert_array_equal(available, closes)
    np.testing.assert_array_equal(
        (closes - starts).astype("timedelta64[D]").astype(int), [31, 29, 31]
    )


def test_utc_week_and_explicit_three_day_anchor():
    rows = [_row("2024-01-01", "2024-01-08"), _row("2024-01-08", "2024-01-15")]
    _validate(rows, ["2024-01-08", "2024-01-15"], kind=NativeClockKind.UTC_WEEK)
    starts, closes, _ = validate_native_kline_times(
        [_row("2024-01-02", "2024-01-05")],
        clock=NativeBarClock(NativeClockKind.FIXED_ELAPSED, 259200, _dt("2024-01-02")),
        expected_closes=np.array(["2024-01-05"], dtype="datetime64[D]"),
        timestamp_unit="ms",
        available_at=np.array(["2024-01-05T00:00:01"], dtype="datetime64[s]"),
        as_of=_dt("2024-01-05T00:00:01"),
    )
    assert starts[0] == _dt("2024-01-02")
    assert closes[0] == _dt("2024-01-05")


@pytest.mark.parametrize("delta", [-1000, -1, 1, 1000])
@pytest.mark.parametrize("unit", ["ms", "us"])
def test_inclusive_native_close_must_be_exact(delta, unit):
    row = _row("2024-02-01", "2024-03-01", unit)
    row[6] = str(int(str(row[6])) + delta)
    with pytest.raises(ValueError, match="close"):
        _validate([row], ["2024-03-01"], unit=unit)


@pytest.mark.parametrize(
    "bad", [True, 1706745600000.0, "1706745600000.0", "-1", "１７０", None]
)
def test_non_native_open_rejected(bad):
    row = _row("2024-02-01", "2024-03-01")
    row[0] = bad
    with pytest.raises(ValueError):
        _validate([row], ["2024-03-01"])


def test_wrong_unit_and_mixed_unit_rejected():
    row = _row("2024-02-01", "2024-03-01", "us")
    with pytest.raises(ValueError):
        _validate([row], ["2024-03-01"], unit="ms")
    row[0] = _ticks("2024-02-01", "ms")
    with pytest.raises(ValueError):
        _validate([row], ["2024-03-01"], unit="ms")


@pytest.mark.parametrize(
    "variant", ["missing_first", "missing_last", "duplicate", "reordered", "extra"]
)
def test_requested_range_not_inferred_from_rows(variant):
    rows = [_row("2024-01-01", "2024-02-01"), _row("2024-02-01", "2024-03-01")]
    bad = {
        "missing_first": rows[1:],
        "missing_last": rows[:1],
        "duplicate": [rows[0], rows[0]],
        "reordered": rows[::-1],
        "extra": rows + [_row("2024-03-01", "2024-04-01")],
    }[variant]
    with pytest.raises(ValueError):
        _validate(bad, ["2024-02-01", "2024-03-01"])


@pytest.mark.parametrize(
    "available,as_of",
    [
        (["2024-02-29T23:59:59.999999999"], "2024-03-02"),
        (["2024-03-02"], "2024-03-01"),
        (["NaT"], "2024-03-02"),
        ([], "2024-03-02"),
        (["2024-03-01"], "NaT"),
    ],
)
def test_publication_and_asof(available, as_of):
    with pytest.raises(ValueError):
        _validate(
            [_row("2024-02-01", "2024-03-01")],
            ["2024-03-01"],
            available=available,
            as_of=as_of,
        )


def test_delayed_publication_retained_without_rounding():
    value = "2024-03-01T00:00:00.000000007"
    _, _, available = _validate(
        [_row("2024-02-01", "2024-03-01")],
        ["2024-03-01"],
        available=[value],
        as_of=value,
    )
    assert available[0] == _dt(value)


def test_malformed_grid_and_precision():
    row = _row("2024-02-01", "2024-03-01")
    with pytest.raises(ValueError):
        _validate([row], ["2024-03-02"])
    with pytest.raises(ValueError):
        _validate([row[:6]], ["2024-03-01"])
    with pytest.raises(ValueError):
        _validate([row], ["2024-03-01"], unit="ns")


def test_missing_calendar_month_is_not_a_complete_grid():
    with pytest.raises(ValueError, match="consecutive"):
        _validate(
            [_row("2024-01-01", "2024-02-01"), _row("2024-03-01", "2024-04-01")],
            ["2024-02-01", "2024-04-01"],
        )


def test_sub_source_tick_grid_rejected():
    start = _dt("2024-01-01T00:00:00.000000001")
    end = _dt("2024-01-02T00:00:00.000000001")
    with pytest.raises(ValueError, match="represented"):
        validate_native_kline_times(
            [_row("2024-01-01", "2024-01-02")],
            clock=NativeBarClock(NativeClockKind.FIXED_ELAPSED, 86400, start),
            expected_closes=np.array([end]),
            timestamp_unit="ms",
            available_at=np.array([end]),
            as_of=end,
        )


@pytest.mark.parametrize("field", ["expected_closes", "available_at"])
def test_masked_native_kline_timestamp_rejected(field):
    closes = np.array(["2024-02-01"], dtype="datetime64[D]")
    inputs = {"expected_closes": closes, "available_at": closes.copy()}
    inputs[field] = np.ma.array(inputs[field], mask=[True])
    with pytest.raises(ValueError, match="masked"):
        validate_native_kline_times(
            [_row("2024-01-01", "2024-02-01")],
            clock=NativeBarClock(NativeClockKind.UTC_MONTH),
            timestamp_unit="ms",
            as_of=_dt("2024-02-02"),
            **inputs,
        )
