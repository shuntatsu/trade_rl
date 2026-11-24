import math
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.native_annualization import evaluate_native_bar_annualization
from trade_rl.evaluation.series import ReturnKind, ReturnSeries


@pytest.mark.parametrize("days", [1, 3, 7])
def test_fixed_intervals_match_existing_ratio_convention(days: int) -> None:
    anchor = np.datetime64("2024-01-01")
    clock = NativeBarClock(NativeClockKind.FIXED_ELAPSED, days * 86400, anchor)
    closes = anchor + np.arange(1, 5) * np.timedelta64(days, "D")
    values = (0.01, -0.02, 0.03, -0.005)
    actual = evaluate_native_bar_annualization(values, clock=clock, closes=closes)
    legacy = evaluate_performance(
        ReturnSeries(values, ReturnKind.BASE_BAR, 365, elapsed_years=4 * days / 365)
    )
    assert actual.elapsed_years == Fraction(4 * days, 365)
    assert actual.periods_per_year == Fraction(365, days)
    assert actual.sharpe == pytest.approx(legacy.sharpe)
    assert actual.sortino == pytest.approx(legacy.sortino)


def test_calendar_month_literal_duration_and_risk_oracle() -> None:
    clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    closes = np.array(["2024-02-01", "2024-03-01", "2024-04-01"], dtype="datetime64[D]")
    values = (0.031, -0.029, 0.062)
    actual = evaluate_native_bar_annualization(values, clock=clock, closes=closes)
    # The leap-year February interval has 29 days, not 30.
    days = (31, 29, 31)
    drift = sum(values) / (91 / 365)
    variance = sum(
        (r - drift * d / 365) ** 2 for r, d in zip(values, days, strict=True)
    ) / (91 / 365)
    downside = 0.029 / math.sqrt(91 / 365)
    assert actual.elapsed_years == Fraction(91, 365)
    assert actual.periods_per_year == Fraction(1095, 91)
    assert actual.annualized_arithmetic_return == pytest.approx(drift)
    assert actual.annualized_volatility == pytest.approx(math.sqrt(variance))
    assert actual.annualized_downside_deviation == pytest.approx(downside)
    assert actual.sharpe == pytest.approx(drift / math.sqrt(variance))
    assert actual.sortino == pytest.approx(drift / downside)


def test_week_anchor_and_constant_returns() -> None:
    clock = NativeBarClock(NativeClockKind.UTC_WEEK)
    closes = np.array(["2024-01-08", "2024-01-15", "2024-01-22"], dtype="datetime64[D]")
    actual = evaluate_native_bar_annualization((0.01,) * 3, clock=clock, closes=closes)
    assert actual.periods_per_year == Fraction(365, 7)
    assert actual.annualized_volatility == 0.0
    assert actual.sharpe == actual.sortino == 0.0


@pytest.mark.parametrize(
    "values",
    [
        (),
        (0.01,),
        (True, 0.01),
        ("0.01", 0.01),
        (math.nan, 0.01),
        (math.inf, 0.01),
        (-1.0, 0.01),
        (1e308, 1e308),
    ],
)
def test_reject_invalid_or_unrepresentable_returns(values: tuple) -> None:
    clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    closes = np.array(["2024-02-01", "2024-03-01"], dtype="datetime64[D]")
    with pytest.raises(ValueError):
        evaluate_native_bar_annualization(values, clock=clock, closes=closes)


@pytest.mark.parametrize(
    "dates",
    [
        ["2024-02-01", "2024-04-01"],
        ["2024-02-02", "2024-03-01"],
        ["2024-03-01", "2024-02-01"],
        ["2024-02-01", "2024-02-01"],
    ],
)
def test_reject_incomplete_or_invalid_month_grid(dates: list[str]) -> None:
    with pytest.raises(ValueError):
        evaluate_native_bar_annualization(
            (0.01, -0.02),
            clock=NativeBarClock(NativeClockKind.UTC_MONTH),
            closes=np.array(dates, dtype="datetime64[D]"),
        )


def test_single_month_includes_first_interval_and_no_variance() -> None:
    actual = evaluate_native_bar_annualization(
        (0.01,),
        clock=NativeBarClock(NativeClockKind.UTC_MONTH),
        closes=np.array(["2023-03-01"], dtype="datetime64[D]"),
    )
    assert actual.elapsed_years == Fraction(28, 365)
    assert actual.periods_per_year == Fraction(365, 28)
    assert actual.sharpe == actual.sortino == 0.0


def test_constant_fixed_return_does_not_create_rounding_variance() -> None:
    actual = evaluate_native_bar_annualization(
        (0.1,) * 3,
        clock=NativeBarClock(NativeClockKind.UTC_WEEK),
        closes=np.array(
            ["2024-01-08", "2024-01-15", "2024-01-22"], dtype="datetime64[D]"
        ),
    )
    assert actual.annualized_volatility == 0.0
    assert actual.sharpe == 0.0


def test_exact_duration_proportional_month_returns_have_zero_variance() -> None:
    actual = evaluate_native_bar_annualization(
        (31 / 1024, 29 / 1024, 31 / 1024),
        clock=NativeBarClock(NativeClockKind.UTC_MONTH),
        closes=np.array(
            ["2024-02-01", "2024-03-01", "2024-04-01"], dtype="datetime64[D]"
        ),
    )
    assert actual.annualized_volatility == 0.0
    assert actual.sharpe == 0.0
