from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.native_clock import NativeBarClock, NativeClockKind


def test_fixed_three_day_grid_uses_declared_anchor() -> None:
    clock = NativeBarClock(
        NativeClockKind.FIXED_ELAPSED,
        fixed_seconds=3 * 86400,
        fixed_anchor=np.datetime64("2024-01-02"),
    )
    starts, closes = clock.interval_bounds(
        np.array(["2024-01-05", "2024-01-08", "2024-01-11"], dtype="datetime64[D]")
    )
    assert (
        starts.tolist()
        == np.array(
            ["2024-01-02", "2024-01-05", "2024-01-08"], dtype="datetime64[ns]"
        ).tolist()
    )
    assert closes.dtype == np.dtype("datetime64[ns]")
    assert clock.elapsed_years(closes) == Fraction(9, 365)
    assert len(closes) / clock.elapsed_years(closes) == Fraction(365, 3)


def test_week_is_monday_utc_not_arbitrary_seven_day_cadence() -> None:
    weekly = NativeBarClock(NativeClockKind.UTC_WEEK)
    monday = np.array(["2024-01-01", "2024-01-08", "2024-01-15"], dtype="datetime64[D]")
    starts, closes = weekly.interval_bounds(monday)
    assert starts[0] == np.datetime64("2023-12-25")
    assert closes[-1] == np.datetime64("2024-01-15")
    assert len(closes) / weekly.elapsed_years(closes) == Fraction(365, 7)
    thursday = monday + np.timedelta64(3, "D")
    with pytest.raises(ValueError, match="anchor"):
        weekly.interval_bounds(thursday)
    fixed = NativeBarClock(
        NativeClockKind.FIXED_ELAPSED,
        fixed_seconds=7 * 86400,
        fixed_anchor=thursday[0],
    )
    fixed.interval_bounds(thursday)


@pytest.mark.parametrize(
    ("closes", "expected_starts", "days"),
    [
        (["2024-02", "2024-03", "2024-04"], ["2024-01", "2024-02", "2024-03"], 91),
        (["2023-02", "2023-03", "2023-04"], ["2023-01", "2023-02", "2023-03"], 90),
        (["2024-12", "2025-01", "2025-02"], ["2024-11", "2024-12", "2025-01"], 92),
        (["2024-03"], ["2024-02"], 29),
    ],
)
def test_month_uses_real_calendar_boundaries(
    closes: list[str], expected_starts: list[str], days: int
) -> None:
    clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    starts, normalized = clock.interval_bounds(np.array(closes, dtype="datetime64[M]"))
    np.testing.assert_array_equal(
        starts, np.array(expected_starts, dtype="datetime64[M]")
    )
    assert clock.elapsed_years(normalized) == Fraction(days, 365)


@pytest.mark.parametrize(
    "closes",
    [
        ["2024-01-01", "2024-03-01"],
        ["2024-02-01", "2024-02-01"],
        ["2024-02-01", "2024-01-01"],
        ["2024-02-02"],
        ["2024-02-01T00:00:00.000000001"],
    ],
)
def test_month_rejects_incomplete_or_misaligned_grid(closes: list[str]) -> None:
    with pytest.raises(ValueError):
        NativeBarClock(NativeClockKind.UTC_MONTH).interval_bounds(
            np.array(closes, dtype="datetime64[ns]")
        )


@pytest.mark.parametrize("seconds", [0, -1, True, 1.5, 10**20])
def test_fixed_rejects_invalid_duration(seconds: int) -> None:
    with pytest.raises(ValueError):
        NativeBarClock(
            NativeClockKind.FIXED_ELAPSED,
            fixed_seconds=seconds,
            fixed_anchor=np.datetime64("2024-01-01"),
        )


@pytest.mark.parametrize(
    "closes",
    [
        np.array([], dtype="datetime64[ns]"),
        np.array(["NaT"], dtype="datetime64[ns]"),
        np.array(["2024-01-01"], dtype="U10"),
        np.array([["2024-01-01"]], dtype="datetime64[D]"),
        np.array(["2500-01-01"], dtype="datetime64[D]"),
        np.array([1], dtype="datetime64[ps]"),
    ],
)
def test_rejects_invalid_timestamp_arrays(closes: np.ndarray) -> None:
    with pytest.raises(ValueError):
        NativeBarClock(NativeClockKind.UTC_WEEK).interval_bounds(closes)


def test_fixed_requires_anchor_and_calendar_refuses_fixed_overrides() -> None:
    with pytest.raises(ValueError):
        NativeBarClock(NativeClockKind.FIXED_ELAPSED, fixed_seconds=3600)
    for kind in (NativeClockKind.UTC_WEEK, NativeClockKind.UTC_MONTH):
        with pytest.raises(ValueError):
            NativeBarClock(kind, fixed_seconds=3600)
        with pytest.raises(ValueError):
            NativeBarClock(kind, fixed_anchor=np.datetime64("2024-01-01"))


def test_fixed_grid_rejects_gap_and_int64_wrap() -> None:
    clock = NativeBarClock(
        NativeClockKind.FIXED_ELAPSED,
        fixed_seconds=1,
        fixed_anchor=np.datetime64("1970-01-01"),
    )
    with pytest.raises(ValueError, match="consecutive"):
        clock.interval_bounds(np.array([0, 2], dtype="datetime64[s]"))
    with pytest.raises(ValueError, match="consecutive"):
        clock.interval_bounds(
            np.array([-9223372036, 9223372036], dtype="datetime64[s]")
        )
    with pytest.raises(ValueError, match="range"):
        clock.interval_bounds(np.array([-9223372036], dtype="datetime64[s]"))


def test_one_ns_off_week_boundary_rejected() -> None:
    with pytest.raises(ValueError, match="anchor"):
        NativeBarClock(NativeClockKind.UTC_WEEK).interval_bounds(
            np.array(["2024-01-01T00:00:00.000000001"], dtype="datetime64[ns]")
        )


def test_month_start_outside_finite_ns_rejected() -> None:
    with pytest.raises(ValueError, match="range"):
        NativeBarClock(NativeClockKind.UTC_MONTH).interval_bounds(
            np.array(["1677-10"], dtype="datetime64[M]")
        )


@pytest.mark.parametrize(
    "seconds", [900, 1800, 3600, 7200, 14400, 21600, 28800, 43200, 86400]
)
def test_existing_fixed_intervals_have_exact_duration(seconds: int) -> None:
    anchor = np.datetime64("2024-01-01T00:00:00")
    clock = NativeBarClock(
        NativeClockKind.FIXED_ELAPSED,
        fixed_seconds=seconds,
        fixed_anchor=anchor,
    )
    closes = anchor + np.arange(1, 121) * np.timedelta64(seconds, "s")
    starts, normalized = clock.interval_bounds(closes)
    np.testing.assert_array_equal(starts, closes - np.timedelta64(seconds, "s"))
    np.testing.assert_array_equal(normalized, closes)
    assert clock.elapsed_years(closes) == Fraction(120 * seconds, 365 * 86400)


def test_leap_year_duration_uses_real_elapsed_days_not_nominal_month_count() -> None:
    clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    closes = np.arange(
        np.datetime64("2024-02"), np.datetime64("2025-02"), dtype="datetime64[M]"
    )
    assert len(closes) == 12
    assert clock.elapsed_years(closes) == Fraction(366, 365)
    assert len(closes) / clock.elapsed_years(closes) == Fraction(730, 61)


def test_native_clock_does_not_enable_unsupported_dataset_config() -> None:
    from trade_rl.data.contracts import FeatureKind, FeatureSpec, MarketBuildConfig

    for timeframe in ("3d", "1w", "1M"):
        with pytest.raises(ValueError, match="unsupported timeframe"):
            MarketBuildConfig(
                base_timeframe=timeframe,
                features=(FeatureSpec(name="return", kind=FeatureKind.LOG_RETURN),),
            )


def test_unrepresentable_or_invalid_fixed_anchor_rejected() -> None:
    for anchor in (
        np.datetime64("NaT"),
        np.datetime64("2500-01-01", "D"),
        np.datetime64(1, "ps"),
    ):
        with pytest.raises(ValueError):
            NativeBarClock(
                NativeClockKind.FIXED_ELAPSED,
                fixed_seconds=1,
                fixed_anchor=anchor,
            )


def test_fixed_grid_anchor_preserves_nanosecond_phase() -> None:
    anchor = np.datetime64("2024-01-01T00:00:00.000000001", "ns")
    clock = NativeBarClock(
        NativeClockKind.FIXED_ELAPSED, fixed_seconds=1, fixed_anchor=anchor
    )
    starts, closes = clock.interval_bounds(
        anchor + np.arange(1, 4) * np.timedelta64(1, "s")
    )
    assert starts[0] == anchor
    with pytest.raises(ValueError, match="anchor"):
        clock.interval_bounds(closes - np.timedelta64(1, "ns"))


def test_native_clock_requires_an_explicit_unmasked_array() -> None:
    clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    closes = np.array(["2024-02-01"], dtype="datetime64[D]")
    with pytest.raises(ValueError, match="NumPy ndarray"):
        clock.interval_bounds([closes[0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="masked"):
        clock.interval_bounds(np.ma.array(closes, mask=[False]))
    # A genuine ordinary array is still admitted, independently of mask tests.
    starts, ends = clock.interval_bounds(closes)
    assert starts[0] == np.datetime64("2024-01-01")
    assert ends[0] == closes[0]


@pytest.mark.parametrize(("seconds", "count"), [(900, 350640), (3600, 87660)])
def test_long_fixed_grid_exact_against_integer_oracle(seconds: int, count: int) -> None:
    # Ten years of continuous synthetic observations crossing the Unix epoch.
    anchor = np.datetime64("1965-01-01T00:00:00.000000003", "ns")
    clock = NativeBarClock(NativeClockKind.FIXED_ELAPSED, seconds, anchor)
    step = seconds * 10**9
    first = int(anchor.astype(np.int64))
    expected = np.fromiter((first + i * step for i in range(count + 1)), dtype=np.int64)
    closes = expected[1:].view("datetime64[ns]")
    starts, ends = clock.interval_bounds(closes)
    np.testing.assert_array_equal(starts.view(np.int64), expected[:-1])
    np.testing.assert_array_equal(ends.view(np.int64), expected[1:])
    assert clock.elapsed_years(closes) == Fraction(count * seconds, 365 * 86400)
    for index in (1, count // 2, count - 1):
        broken = closes.copy()
        broken[index] += np.timedelta64(1, "ns")
        with pytest.raises(ValueError):
            clock.interval_bounds(broken)


def test_fixed_grid_rejects_reverse_wrap_alias() -> None:
    # A naive signed/unsigned subtraction alone observes a one-second delta.
    high = 2**63 - 500_000_000
    low = high + 10**9 - 2**64
    clock = NativeBarClock(NativeClockKind.FIXED_ELAPSED, 1, np.datetime64(high, "ns"))
    with pytest.raises(ValueError):
        clock.interval_bounds(
            np.array([high, low], dtype=np.int64).view("datetime64[ns]")
        )


def test_fixed_grid_accepts_large_exact_positive_delta_across_epoch() -> None:
    seconds = 8_000_000_000
    first = -1_000_000_000_000_000_000
    step = seconds * 10**9
    clock = NativeBarClock(
        NativeClockKind.FIXED_ELAPSED, seconds, np.datetime64(first, "ns")
    )
    closes = np.array([first, first + step], dtype=np.int64).view("datetime64[ns]")
    starts, ends = clock.interval_bounds(closes)
    np.testing.assert_array_equal(starts.view(np.int64), [first - step, first])
    np.testing.assert_array_equal(ends, closes)
