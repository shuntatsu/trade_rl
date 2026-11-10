"""SD1 (#1146): unit checks of the measurement tools against independent references."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from trade_rl.evaluation.comparison.bootstrap import moving_block_mean_test
from trade_rl.evaluation.signal_measurement import (
    SignalMeasurementError,
    average_ranks,
    bh_adjusted,
    bh_rejections,
    block_mean_interval,
    causal_beta_residual,
    cross_sectional_demean,
    daily_ic,
    daily_rank_ic_increment,
    forward_log_returns,
    holm_adjusted,
    holm_rejections,
    linear_ic,
    market_mean,
    nonstationarity_profile,
    rank_ic,
    utc_day_ids,
)

NS_PER_HOUR = 3_600_000_000_000

# ------------------------------------------------------------------------------ ranks / IC


def test_average_ranks_match_scipy_including_ties() -> None:
    values = np.array([3.0, 1.0, 3.0, 2.0, 3.0, 1.0, -5.0, 2.0])

    assert np.array_equal(average_ranks(values), stats.rankdata(values))
    with pytest.raises(SignalMeasurementError):
        average_ranks(np.array([1.0, np.nan]))


def test_rank_and_linear_ic_match_independent_references_on_random_data() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        left = np.round(rng.standard_normal(60), 1)  # rounding creates ties
        right = np.round(0.4 * left + rng.standard_normal(60), 1)

        assert rank_ic(left, right) == pytest.approx(
            stats.spearmanr(left, right)[0], abs=1e-12
        )
        assert linear_ic(left, right) == pytest.approx(
            np.corrcoef(left, right)[0, 1], abs=1e-12
        )


def test_hand_worked_ic_values() -> None:
    # Ranks (1,2,3,4) vs (1,3,2,4): sum d^2 = 2, rho = 1 - 6*2/(4*15) = 0.8.
    assert rank_ic([1, 2, 3, 4], [10, 30, 20, 40]) == pytest.approx(0.8)
    assert linear_ic([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert linear_ic([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)


def test_rank_ic_is_invariant_to_a_monotone_transform_and_linear_ic_is_not() -> None:
    rng = np.random.default_rng(2)
    left = rng.standard_normal(500)
    right = left + rng.standard_normal(500)

    assert rank_ic(left, np.exp(3.0 * right)) == pytest.approx(rank_ic(left, right))
    assert linear_ic(left, np.exp(3.0 * right)) != pytest.approx(
        linear_ic(left, right), abs=0.05
    )


def test_undefined_cross_sections_return_none_never_zero() -> None:
    assert rank_ic([1.0], [2.0]) is None
    assert rank_ic([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # constant score
    assert linear_ic([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]) is None  # constant target
    assert rank_ic([np.nan, 1.0], [1.0, 2.0]) is None  # one eligible row left
    with pytest.raises(SignalMeasurementError):
        rank_ic([1.0, 2.0], [1.0])
    with pytest.raises(SignalMeasurementError):
        rank_ic(np.zeros((2, 2)), np.zeros((2, 2)))


def test_non_finite_rows_are_dropped_not_imputed() -> None:
    with_gaps = rank_ic([1, 2, np.nan, 3, 4, np.inf], [1, 3, 5, 2, 4, 9])
    clean = rank_ic([1, 2, 3, 4], [1, 3, 2, 4])

    assert with_gaps == pytest.approx(clean)


def test_daily_ic_groups_by_day_and_counts_skipped_days() -> None:
    days = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3])
    scores = np.array([1, 2, 3, 1, 2, 3, 5, 5, 7.0])  # day 2 constant; day 3 single row
    target = np.array([1, 2, 3, 3, 2, 1, 1, 2, 4.0])

    result = daily_ic(days, scores, target, method="rank")

    assert result.day_ids == (0, 1)
    assert result.values == (pytest.approx(1.0), pytest.approx(-1.0))
    assert result.skipped_days == 2
    assert result.mean == pytest.approx(0.0)
    with pytest.raises(SignalMeasurementError):
        daily_ic(days, scores, target, method="spearman")  # type: ignore[arg-type]
    with pytest.raises(SignalMeasurementError):
        daily_ic(days[:-1], scores, target, method="rank")
    with pytest.raises(SignalMeasurementError):
        daily_ic(days.astype(float), scores, target, method="rank")


def test_daily_ic_is_independent_of_row_order() -> None:
    rng = np.random.default_rng(3)
    days = np.repeat(np.arange(10), 30)
    scores = rng.standard_normal(300)
    target = scores + rng.standard_normal(300)
    order = rng.permutation(300)

    base = daily_ic(days, scores, target, method="linear")
    shuffled = daily_ic(days[order], scores[order], target[order], method="linear")

    assert base.day_ids == shuffled.day_ids
    assert base.values == pytest.approx(shuffled.values)


def test_ic_increment_uses_identical_rows_for_forecast_and_baseline() -> None:
    days = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    target = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    forecast = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
    baseline = np.array([1.0, 3.0, 2.0, 4.0, np.nan, 3.0, 2.0, 1.0])

    result = daily_rank_ic_increment(days, forecast, baseline, target)

    # Day 0: 1.0 - 0.8. Day 1 drops row 4 for all three series: forecast ranks of
    # (3,2,1) vs target (2,3,4) is -1, baseline (3,2,1) is -1, increment 0.
    assert result.day_ids == (0, 1)
    assert result.values == (pytest.approx(0.2), pytest.approx(0.0))
    with pytest.raises(SignalMeasurementError):
        daily_rank_ic_increment(days, forecast, baseline[:-1], target)


def test_utc_day_ids_use_whole_utc_days() -> None:
    day = 24 * NS_PER_HOUR
    stamps = np.array([0, day - 1, day, 2 * day + 5, -1], dtype=np.int64)

    assert utc_day_ids(stamps).tolist() == [0, 0, 1, 2, -1]
    with pytest.raises(SignalMeasurementError):
        utc_day_ids(stamps.astype(float))


# ------------------------------------------------------------------------ multiple testing


def _slow_holm(p: list[float]) -> list[float]:
    m = len(p)
    adjusted = []
    for index, value in enumerate(p):
        rank = sorted(p).index(value)  # first position among equal values
        best = 0.0
        for j in range(rank + 1):
            best = max(best, (m - j) * sorted(p)[j])
        adjusted.append(min(1.0, best))
    return adjusted


def test_holm_matches_the_definition_and_a_hand_worked_example() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04
    # running max: 0.02, 0.03, 0.06, 0.06
    assert holm_adjusted(p).tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    rng = np.random.default_rng(4)
    for _ in range(20):
        random_p = np.round(rng.uniform(size=12), 3).tolist()
        assert holm_adjusted(random_p).tolist() == pytest.approx(_slow_holm(random_p))


def test_bh_matches_scipy_and_a_hand_worked_example() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005, 0.01, 0.03, 0.04 -> m*p/rank = 0.02, 0.02, 0.04, 0.04
    assert bh_adjusted(p).tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])
    rng = np.random.default_rng(5)
    for _ in range(20):
        random_p = np.round(rng.uniform(size=25) ** 2, 4)
        assert bh_adjusted(random_p) == pytest.approx(
            stats.false_discovery_control(random_p, method="bh"), abs=1e-12
        )


def test_adjusted_p_values_are_capped_monotone_and_order_independent() -> None:
    rng = np.random.default_rng(6)
    p = rng.uniform(size=40)
    order = rng.permutation(40)

    for function in (holm_adjusted, bh_adjusted):
        adjusted = function(p)
        assert ((adjusted >= p - 1e-15) & (adjusted <= 1.0)).all()
        assert function(p[order]) == pytest.approx(adjusted[order])
        ranked = adjusted[np.argsort(p, kind="stable")]
        assert (np.diff(ranked) >= -1e-15).all()
    assert holm_adjusted([0.9, 0.95]).tolist() == [1.0, 1.0]
    # Holm is never less conservative than BH.
    assert (holm_adjusted(p) >= bh_adjusted(p) - 1e-15).all()


def test_equality_with_the_level_counts_as_a_rejection() -> None:
    assert bh_rejections([0.10], 0.10).tolist() == [True]
    assert bh_rejections([math.nextafter(0.10, 1.0)], 0.10).tolist() == [False]
    assert holm_rejections([0.05], 0.05).tolist() == [True]
    assert holm_rejections([0.01, 0.04, 0.03, 0.005], 0.05).tolist() == [
        True,
        False,
        False,
        True,
    ]


@pytest.mark.parametrize(
    "bad", [[], [0.5, 1.2], [-0.1], [np.nan], [[0.1]], "0.1", [0.5, np.inf]]
)
def test_p_values_are_validated(bad: object) -> None:
    with pytest.raises(SignalMeasurementError):
        holm_adjusted(bad)
    with pytest.raises(SignalMeasurementError):
        bh_adjusted(bad)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, True, "0.1"])
def test_levels_are_validated(level: object) -> None:
    with pytest.raises(SignalMeasurementError):
        bh_rejections([0.01], level)  # type: ignore[arg-type]
    with pytest.raises(SignalMeasurementError):
        holm_rejections([0.01], level)  # type: ignore[arg-type]


def test_block_mean_interval_reuses_the_repository_bootstrap_unchanged() -> None:
    rng = np.random.default_rng(7)
    series = 0.05 + rng.standard_normal(80) * 0.1

    wrapped = block_mean_interval(series, block_days=20, resamples=300, seed=0)
    direct = moving_block_mean_test(
        tuple(float(v) for v in series), n_bootstrap=300, seed=0, block_size=20
    )

    assert wrapped.lower == direct.lower_ci and wrapped.upper == direct.upper_ci
    assert wrapped.p_value_one_sided == direct.p_value
    assert wrapped.mean == pytest.approx(float(series.mean()))
    assert wrapped.n_days == 80 and wrapped.block_days == 20
    assert wrapped.lower < wrapped.mean < wrapped.upper
    for bad in ([], [1.0, np.nan], [[1.0]]):
        with pytest.raises(SignalMeasurementError):
            block_mean_interval(bad, block_days=20, resamples=10, seed=0)


# ------------------------------------------------------------------------------- targets


def test_forward_log_returns_hand_check_and_unmatured_tail() -> None:
    close = np.array([[100.0, 10.0], [110.0, 10.0], [121.0, 20.0], [121.0, 20.0]])

    out = forward_log_returns(close, 2)

    assert out[0, 0] == pytest.approx(math.log(121.0 / 100.0))
    assert out[0, 1] == pytest.approx(math.log(2.0))
    assert out[1, 0] == pytest.approx(math.log(121.0 / 110.0))
    assert np.isnan(out[2:]).all()
    for bad in (0, -1, 2.0, True):
        with pytest.raises(SignalMeasurementError):
            forward_log_returns(close, bad)  # type: ignore[arg-type]
    with pytest.raises(SignalMeasurementError):
        forward_log_returns(np.array([[1.0], [0.0]]), 1)


def test_cross_sectional_demean_removes_the_same_time_mean_only() -> None:
    returns = np.array(
        [[0.02, 0.04, 0.06], [0.01, np.nan, 0.03], [0.05, np.nan, np.nan], [np.nan] * 3]
    )

    out = cross_sectional_demean(returns)

    assert out[0].tolist() == pytest.approx([-0.02, 0.0, 0.02])
    assert out[1].tolist()[0] == pytest.approx(-0.01) and out[1].tolist()[
        2
    ] == pytest.approx(0.01)
    assert np.isnan(out[1, 1])
    assert np.isnan(out[2]).all()  # a lone symbol has no cross-section
    assert np.isnan(out[3]).all()
    assert np.nansum(out[:2], axis=1) == pytest.approx([0.0, 0.0])
    assert market_mean(returns)[[0, 1, 2]].tolist() == pytest.approx([0.04, 0.02, 0.05])
    # Adding any common level to every symbol changes nothing.
    assert cross_sectional_demean(returns + 0.5)[0].tolist() == pytest.approx(
        out[0].tolist()
    )


def _slow_beta_residual(returns: np.ndarray, horizon: int, min_obs: int) -> np.ndarray:
    rows, symbols = returns.shape
    market = np.array(
        [np.nanmean(row) if np.isfinite(row).any() else np.nan for row in returns]
    )
    out = np.full(returns.shape, np.nan)
    for t in range(rows):
        for i in range(symbols):
            history = [
                s
                for s in range(0, t - horizon + 1)
                if np.isfinite(returns[s, i]) and np.isfinite(market[s])
            ]
            if len(history) < min_obs or not (
                np.isfinite(returns[t, i]) and np.isfinite(market[t])
            ):
                continue
            x = market[history]
            y = returns[history, i]
            if np.var(x) <= 1e-18:
                continue
            slope = np.polyfit(x, y, 1)[0]
            out[t, i] = returns[t, i] - slope * market[t]
    return out


def test_causal_beta_residual_matches_a_slow_reference_with_gaps() -> None:
    rng = np.random.default_rng(8)
    returns = rng.standard_normal((70, 3)) * 0.02
    returns += 0.5 * rng.standard_normal((70, 1)) * np.array([[0.5, 1.0, 1.5]])
    returns[rng.uniform(size=returns.shape) < 0.1] = np.nan

    for horizon in (1, 4):
        got = causal_beta_residual(returns, horizon, min_observations=15)
        want = _slow_beta_residual(returns, horizon, 15)
        assert np.array_equal(np.isnan(got), np.isnan(want))
        assert got[np.isfinite(got)] == pytest.approx(want[np.isfinite(want)], abs=1e-9)


def test_causal_beta_residual_never_uses_rows_that_have_not_matured() -> None:
    rng = np.random.default_rng(10)
    returns = rng.standard_normal((80, 3)) * 0.02
    horizon, decision = 5, 60
    base = causal_beta_residual(returns, horizon, min_observations=20)

    changed = returns.copy()
    changed[decision - horizon + 1 : decision] += rng.standard_normal((horizon - 1, 3))
    changed[decision + 1 :] += rng.standard_normal((80 - decision - 1, 3))
    after = causal_beta_residual(changed, horizon, min_observations=20)

    # Rows after t - h are unmatured at t, so t's coefficient cannot move; t's own row
    # is unchanged, so the residual at t is identical.
    assert after[decision] == pytest.approx(base[decision])
    shifted = returns.copy()
    shifted[decision - horizon, 0] += 1.0  # a matured row does change the coefficient
    assert not np.allclose(
        causal_beta_residual(shifted, horizon, min_observations=20)[decision],
        base[decision],
    )


def test_causal_beta_residual_reports_missing_not_filled() -> None:
    rng = np.random.default_rng(12)
    returns = rng.standard_normal((30, 2)) * 0.01

    out = causal_beta_residual(returns, 1, min_observations=25)

    assert np.isnan(out[:25]).all()  # fewer than 25 matured rows before row 25
    assert np.isfinite(out[25:]).all()
    flat = np.zeros((30, 2))
    assert np.isnan(causal_beta_residual(flat, 1, min_observations=5)).all()
    with pytest.raises(SignalMeasurementError):
        causal_beta_residual(returns, 1, min_observations=1)
    with pytest.raises(SignalMeasurementError):
        causal_beta_residual(returns, 0, min_observations=5)


# ----------------------------------------------------------------------- non-stationarity


def _drifting_panel() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13)
    periods = np.repeat(np.arange(2021, 2026), 400)
    stable = rng.standard_normal(periods.shape[0])
    drifting = rng.standard_normal(periods.shape[0])
    drifting[periods >= 2024] += 5.0  # a level shift, like a funding rate moving
    return np.column_stack([stable, drifting]), periods


def test_profile_recovers_a_planted_level_shift_and_a_stationary_control() -> None:
    values, periods = _drifting_panel()

    result = nonstationarity_profile(
        values, periods, ("stable", "drifting"), min_train_rows=100
    )
    by_key = {(p.feature, p.period): p for p in result.profiles}

    assert result.skipped == ()
    assert {p.period for p in result.profiles} == {
        2022,
        2023,
        2024,
        2025,
    }  # first not profiled
    for period in (2022, 2023, 2024, 2025):
        assert abs(by_key[("stable", period)].z_mean) < 0.25
        assert 0.8 < by_key[("stable", period)].z_std < 1.2
    assert abs(by_key[("drifting", 2023)].z_mean) < 0.25
    assert by_key[("drifting", 2024)].z_mean > 4.0  # train window is clean 2021-2023
    assert by_key[("drifting", 2024)].share_abs_z_above_clip > 0.9
    assert by_key[("drifting", 2024)].z_p05 > 1.0
    # The shifted years enter the train window, so 2025 is judged against a shifted past.
    assert 0.5 < by_key[("drifting", 2025)].z_mean < 3.0


def test_profile_uses_only_strictly_earlier_periods() -> None:
    values, periods = _drifting_panel()
    names = ("stable", "drifting")
    base = nonstationarity_profile(values, periods, names, min_train_rows=100)

    later = values.copy()
    later[periods == 2025] += 100.0  # a future period must not move an earlier record
    after_later = nonstationarity_profile(later, periods, names, min_train_rows=100)
    same = values.copy()
    same[periods == 2023] *= (
        3.0  # the evaluated period must not move its own train stats
    )
    after_same = nonstationarity_profile(same, periods, names, min_train_rows=100)

    def record(result, feature, period):  # noqa: ANN001, ANN202
        return next(
            p for p in result.profiles if p.feature == feature and p.period == period
        )

    assert record(after_later, "stable", 2024) == record(base, "stable", 2024)
    assert (
        record(after_same, "stable", 2023).train_mean
        == record(base, "stable", 2023).train_mean
    )
    assert (
        record(after_same, "stable", 2023).train_std
        == record(base, "stable", 2023).train_std
    )
    assert (
        record(after_same, "stable", 2023).z_mean != record(base, "stable", 2023).z_mean
    )


def test_profile_lists_what_it_cannot_profile_instead_of_guessing() -> None:
    periods = np.repeat(np.array([1, 2, 3]), 50)
    values = np.column_stack(
        [
            np.arange(150, dtype=float),
            np.ones(150),  # zero train deviation
            np.where(
                periods == 3, np.nan, np.arange(150, dtype=float)
            ),  # no values in 3
        ]
    )

    result = nonstationarity_profile(
        values, periods, ("ramp", "flat", "late_gap"), min_train_rows=60
    )
    reasons = {(s.feature, s.period): s.reason for s in result.skipped}

    assert ("ramp", 2) in reasons and reasons[("ramp", 2)] == "too few train rows"
    assert reasons[("flat", 3)] == "zero train deviation"
    assert reasons[("late_gap", 3)] == "no finite values"
    assert {p.feature for p in result.profiles} == {
        "ramp"
    }  # only ramp period 3 qualifies


def test_profile_inputs_are_validated() -> None:
    values = np.ones((10, 2))
    periods = np.zeros(10, dtype=np.int64)
    for kwargs in (
        {"values": values, "period_ids": periods[:-1], "feature_names": ("a", "b")},
        {"values": values, "period_ids": periods, "feature_names": ("a",)},
        {"values": values, "period_ids": periods, "feature_names": ("a", "a")},
        {
            "values": values,
            "period_ids": periods.astype(float),
            "feature_names": ("a", "b"),
        },
        {"values": values[0], "period_ids": periods, "feature_names": ("a", "b")},
    ):
        with pytest.raises(SignalMeasurementError):
            nonstationarity_profile(min_train_rows=2, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(SignalMeasurementError):
        nonstationarity_profile(values, periods, ("a", "b"), min_train_rows=1)
    with pytest.raises(SignalMeasurementError):
        nonstationarity_profile(
            values, periods, ("a", "b"), min_train_rows=2, clip_abs=0.0
        )
