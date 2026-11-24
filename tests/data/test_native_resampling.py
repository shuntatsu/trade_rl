"""Independent synthetic oracles for complete native OHLCV aggregation."""

import numpy as np
import pytest

from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.native_resampling import resample_complete_native_bars


def _daily():
    return NativeBarClock(
        NativeClockKind.FIXED_ELAPSED, 86400, np.datetime64("1970-01-01")
    )


def _fixture():
    closes = np.arange("2024-01-02", "2024-03-02", dtype="datetime64[D]")
    i = np.arange(len(closes), dtype=float) + 100
    values = np.column_stack((i, i + 3, i - 1, i + 1, i * 0 + 2, i * 0 + 5))
    return closes, values


def _run(
    closes,
    values,
    *,
    availability=None,
    target=None,
    source_clock=None,
    target_clock=None,
    as_of="2024-03-05",
):
    return resample_complete_native_bars(
        source_clock=_daily() if source_clock is None else source_clock,
        source_closes=closes,
        source_available_at=closes if availability is None else availability,
        source_ohlcv=values,
        target_clock=NativeBarClock(NativeClockKind.UTC_MONTH)
        if target_clock is None
        else target_clock,
        target_closes=np.array(["2024-02-01", "2024-03-01"], dtype="datetime64[D]")
        if target is None
        else target,
        as_of=np.datetime64(as_of),
    )


def test_leap_month_partition_and_late_publication():
    closes, values = _fixture()
    availability = closes.copy()
    availability[4] = np.datetime64("2024-02-03")
    out = _run(closes, values, availability=availability)
    np.testing.assert_array_equal(out.member_counts, [31, 29])
    np.testing.assert_array_equal(
        out.ohlcv,
        [
            [100, 133, 99, 131, 62, 155],
            [131, 162, 130, 160, 58, 145],
        ],
    )
    np.testing.assert_array_equal(
        out.available_at, np.array(["2024-02-03", "2024-03-01"], dtype="datetime64[ns]")
    )
    np.testing.assert_array_equal(
        out.closes, np.array(["2024-02-01", "2024-03-01"], dtype="datetime64[ns]")
    )


def test_week_and_prefix_invariance():
    closes, values = _fixture()
    target = np.array(["2024-01-08", "2024-01-15"], dtype="datetime64[D]")
    week = NativeBarClock(NativeClockKind.UTC_WEEK)
    whole = _run(closes[:14], values[:14], target=target, target_clock=week)
    prefix = _run(closes[:7], values[:7], target=target[:1], target_clock=week)
    changed = values[:14].copy()
    changed[7:, :4] *= 10
    mutated = _run(closes[:14], changed, target=target, target_clock=week)
    np.testing.assert_array_equal(whole.ohlcv[0], prefix.ohlcv[0])
    np.testing.assert_array_equal(whole.ohlcv[0], mutated.ohlcv[0])
    np.testing.assert_array_equal(whole.member_counts, [7, 7])


@pytest.mark.parametrize("index", [0, 12, 30, 59])
def test_missing_member_fails_closed(index):
    closes, values = _fixture()
    with pytest.raises(ValueError):
        _run(np.delete(closes, index), np.delete(values, index, axis=0))


def test_extra_member_fails_closed():
    closes, values = _fixture()
    with pytest.raises(ValueError, match="coverage"):
        _run(closes, values, target=np.array(["2024-02-01"], dtype="datetime64[D]"))


def test_three_day_source_cannot_straddle_month_boundary():
    closes = np.arange("2024-01-04", "2024-04-01", 3, dtype="datetime64[D]")
    values = np.tile([1, 2, 1, 2, 3, 4], (len(closes), 1))
    clock = NativeBarClock(
        NativeClockKind.FIXED_ELAPSED, 259200, np.datetime64("2024-01-01")
    )
    with pytest.raises(ValueError):
        _run(
            closes,
            values,
            target=np.array(["2024-02-01", "2024-03-01"], dtype="datetime64[D]"),
            source_clock=clock,
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, 0.0])
def test_invalid_price(bad):
    closes, values = _fixture()
    values[0, 0] = bad
    with pytest.raises(ValueError):
        _run(closes, values)


@pytest.mark.parametrize("column,bad", [(1, 99), (2, 105), (5, -1)])
def test_invalid_ohlcv(column, bad):
    closes, values = _fixture()
    values[0, column] = bad
    with pytest.raises(ValueError):
        _run(closes, values)


def test_late_unavailable_and_premature_source():
    closes, values = _fixture()
    availability = closes.copy()
    availability[0] = np.datetime64("2024-03-06")
    with pytest.raises(ValueError, match="availability"):
        _run(closes, values, availability=availability)
    availability[0] = np.datetime64("2024-01-01")
    with pytest.raises(ValueError, match="availability"):
        _run(closes, values, availability=availability)


def test_overflow_volume_rejected():
    closes, values = _fixture()
    values[:, 4] = 1e308
    with pytest.raises(ValueError, match="volume"):
        _run(closes, values)


def test_three_day_target_explicit_phase():
    closes, values = _fixture()
    clock = NativeBarClock(
        NativeClockKind.FIXED_ELAPSED, 259200, np.datetime64("2024-01-01")
    )
    result = _run(
        closes[:6],
        values[:6],
        target=np.array(["2024-01-04", "2024-01-07"], dtype="datetime64[D]"),
        target_clock=clock,
    )
    np.testing.assert_array_equal(result.member_counts, [3, 3])
    np.testing.assert_array_equal(
        result.ohlcv,
        [
            [100, 105, 99, 103, 6, 15],
            [103, 108, 102, 106, 6, 15],
        ],
    )


@pytest.mark.parametrize("kind", ["shape", "bool", "complex", "object", "wide_int"])
def test_bad_numeric_input(kind):
    closes, values = _fixture()
    if kind == "shape":
        values = values[:, :5]
    elif kind == "wide_int":
        values = values.astype(np.int64)
        values[0, 4] = 2**53 + 1
    else:
        values = values.astype(kind)
    with pytest.raises(ValueError, match="OHLCV"):
        _run(closes, values)


@pytest.mark.parametrize("kind", ["duplicate", "reverse", "nat", "off_grid"])
def test_bad_source_grid(kind):
    closes, values = _fixture()
    closes = closes.astype("datetime64[h]")
    if kind == "duplicate":
        closes[1] = closes[0]
    elif kind == "reverse":
        closes[[0, 1]] = closes[[1, 0]]
    elif kind == "nat":
        closes[0] = np.datetime64("NaT")
    else:
        closes[0] += np.timedelta64(1, "h")
    with pytest.raises(ValueError):
        _run(closes, values)


def test_derived_output_does_not_alias_source():
    closes, values = _fixture()
    out = _run(closes, values)
    assert not np.shares_memory(out.ohlcv, values)
    assert not np.shares_memory(out.closes, closes)
    values[:] = 0
    assert out.ohlcv[0, 0] == 100


@pytest.mark.parametrize("field", ["closes", "availability", "ohlcv", "target"])
@pytest.mark.parametrize("view", [False, True])
def test_missing_mask_cannot_be_discarded_by_native_admission(field, view):
    closes, values = _fixture()
    inputs = {
        "closes": closes,
        "availability": closes.copy(),
        "ohlcv": values,
        "target": np.array(["2024-02-01", "2024-03-01"], dtype="datetime64[D]"),
    }
    masked = np.ma.array(inputs[field], mask=False, copy=True)
    masked.mask.flat[0] = True
    inputs[field] = masked.astype(masked.dtype).view(np.ndarray) if view else masked
    if view:
        assert isinstance(inputs[field].base, np.ma.MaskedArray)
    with pytest.raises(ValueError, match="masked"):
        _run(
            inputs["closes"],
            inputs["ohlcv"],
            availability=inputs["availability"],
            target=inputs["target"],
        )
