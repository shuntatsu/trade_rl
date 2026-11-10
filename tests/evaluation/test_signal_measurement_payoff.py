"""SD1 (#1146): the fast after-cost payoff is checked against a slow independent loop.

Synthetic data only. The slow reference in ``slow_payoff_reference`` shares no code with
the implementation under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.evaluation.signal_measurement_synthetic import HOURS_PER_DAY, day_ids_for
from tests.evaluation.slow_payoff_reference import (
    slow_after_cost_payoff,
    slow_daily_mean,
    slow_drift_excess_payoff,
    slow_drift_sign_payoff,
)
from trade_rl.evaluation.signal_measurement import (
    SignalMeasurementError,
    after_cost_payoff,
    block_mean_interval,
    daily_mean,
    drift_excess_payoff,
    drift_sign_payoff,
)

ROUND_TRIP = 0.0014  # 14 bp, the cost fraction the protocol uses (an argument here)


def _as_list(array: np.ndarray) -> list[float]:
    return [float(v) for v in array]


def _compare(fast: np.ndarray, slow: list[float | None]) -> None:
    assert fast.shape[0] == len(slow)
    for got, want in zip(fast, slow, strict=True):
        if want is None:
            assert math.isnan(got)
        else:
            assert got == pytest.approx(want, abs=1e-15, rel=1e-12)


def _random_case(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = 600
    realised = 0.01 * rng.standard_normal(rows)
    forecast = 0.004 * rng.standard_normal(rows)
    drift = 0.001 * rng.standard_normal(rows)
    # Boundary and degenerate rows the vectorised path must treat like the loop does.
    forecast[:5] = [0.0, ROUND_TRIP, -ROUND_TRIP, 2 * ROUND_TRIP, -2 * ROUND_TRIP]
    drift[:3] = [0.0, 0.0, 1e-9]
    forecast[10:13] = np.nan
    drift[13:15] = np.nan
    realised[15:18] = np.nan
    return forecast, drift, realised


# ----------------------------------------------------------------------- hand-worked


def test_hand_worked_after_cost_payoff_at_cost_x1_and_x2() -> None:
    forecast = np.array([0.003, -0.003, 0.001, 0.003])
    realised = np.array([0.010, 0.010, 0.010, -0.020])

    x1 = after_cost_payoff(
        forecast, realised, k=1, round_trip_cost=ROUND_TRIP, cost_multiplier=1
    )
    x2 = after_cost_payoff(
        forecast, realised, k=1, round_trip_cost=ROUND_TRIP, cost_multiplier=2
    )

    # x1 threshold 0.0014: long earns 0.01-0.0014; short loses 0.01 and pays 0.0014;
    # 0.001 is below the threshold so it stays flat and pays nothing; long loses -0.02.
    assert x1.tolist() == pytest.approx([0.0086, -0.0114, 0.0, -0.0214])
    # x2 cost 0.0028 and threshold 0.0028 (k = 1): 0.003 still trades, 0.001 is flat.
    assert x2.tolist() == pytest.approx([0.0072, -0.0128, 0.0, -0.0228])


def test_equality_with_the_entry_threshold_stays_flat() -> None:
    out = after_cost_payoff(
        np.array([ROUND_TRIP, -ROUND_TRIP, math.nextafter(ROUND_TRIP, 1.0)]),
        np.array([0.01, 0.01, 0.01]),
        k=1,
        round_trip_cost=ROUND_TRIP,
        cost_multiplier=1,
    )

    assert out[0] == 0.0 and out[1] == 0.0
    assert out[2] == pytest.approx(0.01 - ROUND_TRIP)


def test_hand_worked_drift_excess_payoff() -> None:
    forecast = np.array([0.003, 0.003, 0.0])
    drift = np.array([0.0005, -0.0005, 0.0])
    realised = np.array([0.010, 0.010, 0.010])

    x1 = drift_excess_payoff(
        forecast, drift, realised, k=1, round_trip_cost=ROUND_TRIP, cost_multiplier=1
    )

    # Long after cost 0.0086 vs a free long 0.01 -> -0.0014; vs a free short -0.01 -> 0.0186;
    # flat signal vs a flat comparator (zero drift) -> 0.
    assert x1.tolist() == pytest.approx([-0.0014, 0.0186, 0.0])
    assert drift_sign_payoff(drift, realised).tolist() == pytest.approx(
        [0.01, -0.01, 0.0]
    )


# ----------------------------------------------------- fast versus the slow reference


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("multiplier", [1, 2])
def test_fast_payoffs_equal_the_slow_loop_for_every_k_and_cost_multiplier(
    seed: int, k: int, multiplier: int
) -> None:
    forecast, drift, realised = _random_case(seed)
    settings = {"k": k, "round_trip_cost": ROUND_TRIP, "cost_multiplier": multiplier}

    _compare(
        after_cost_payoff(forecast, realised, **settings),
        slow_after_cost_payoff(_as_list(forecast), _as_list(realised), **settings),
    )
    _compare(
        drift_excess_payoff(forecast, drift, realised, **settings),
        slow_drift_excess_payoff(
            _as_list(forecast), _as_list(drift), _as_list(realised), **settings
        ),
    )
    _compare(
        drift_sign_payoff(drift, realised),
        slow_drift_sign_payoff(_as_list(drift), _as_list(realised)),
    )


def test_missing_inputs_make_a_row_ineligible_not_flat() -> None:
    forecast, drift, realised = _random_case(5)
    settings = {"k": 1, "round_trip_cost": ROUND_TRIP, "cost_multiplier": 1}

    payoff = after_cost_payoff(forecast, realised, **settings)
    excess = drift_excess_payoff(forecast, drift, realised, **settings)

    assert np.isnan(payoff[10:13]).all() and np.isnan(payoff[15:18]).all()
    assert np.isfinite(
        payoff[13:15]
    ).all()  # a missing drift does not touch the signal payoff
    assert np.isnan(excess[10:18]).all()  # forecast, drift or label missing: no excess


def test_doubling_the_cost_never_raises_a_payoff_on_a_row_both_trade() -> None:
    forecast, _, realised = _random_case(6)
    x1 = after_cost_payoff(
        forecast, realised, k=1, round_trip_cost=ROUND_TRIP, cost_multiplier=1
    )
    x2 = after_cost_payoff(
        forecast, realised, k=1, round_trip_cost=ROUND_TRIP, cost_multiplier=2
    )

    # Above the x2 entry threshold (k * 2 * cost) both settings hold the same position, so
    # the only difference is the extra round-trip charge.
    traded_at_x2 = np.isfinite(x2) & (np.abs(forecast) > 2 * ROUND_TRIP)
    assert traded_at_x2.sum() > 50
    assert (x2[traded_at_x2] < x1[traded_at_x2]).all()
    assert np.allclose(
        x1[traded_at_x2] - x2[traded_at_x2], ROUND_TRIP, atol=1e-15, rtol=0.0
    )


def test_daily_mean_matches_the_slow_grouping_and_skips_empty_days() -> None:
    forecast, drift, realised = _random_case(7)
    days = np.repeat(np.arange(30), 20)
    values = drift_excess_payoff(
        forecast, drift, realised, k=1, round_trip_cost=ROUND_TRIP, cost_multiplier=1
    )
    values[days == 4] = np.nan  # a day with no eligible row has no mean

    got = daily_mean(days, values)
    want = slow_daily_mean(
        [int(d) for d in days], [None if math.isnan(v) else float(v) for v in values]
    )

    assert got.day_ids == tuple(want)
    assert got.values == pytest.approx(tuple(want.values()), abs=1e-15)
    assert 4 not in got.day_ids and got.skipped_days == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"k": 0},
        {"k": 1.5},
        {"k": True},
        {"round_trip_cost": -0.001},
        {"round_trip_cost": 0.0},
        {"round_trip_cost": float("nan")},
        {"cost_multiplier": 0},
        {"cost_multiplier": 1.0},
    ],
)
def test_cost_parameters_are_validated(bad: dict[str, object]) -> None:
    settings: dict[str, object] = {
        "k": 1,
        "round_trip_cost": ROUND_TRIP,
        "cost_multiplier": 1,
    }
    settings.update(bad)
    with pytest.raises(SignalMeasurementError):
        after_cost_payoff(np.array([0.01]), np.array([0.01]), **settings)  # type: ignore[arg-type]


def test_payoff_inputs_must_be_aligned_numeric_vectors() -> None:
    settings = {"k": 1, "round_trip_cost": ROUND_TRIP, "cost_multiplier": 1}
    with pytest.raises(SignalMeasurementError):
        after_cost_payoff(np.zeros(3), np.zeros(4), **settings)
    with pytest.raises(SignalMeasurementError):
        after_cost_payoff(np.zeros((2, 2)), np.zeros((2, 2)), **settings)
    with pytest.raises(SignalMeasurementError):
        drift_excess_payoff(np.zeros(3), np.zeros(2), np.zeros(3), **settings)
    with pytest.raises(SignalMeasurementError):
        drift_sign_payoff(np.zeros(3), np.zeros(2))
    with pytest.raises(SignalMeasurementError):
        daily_mean(np.zeros(3, dtype=np.int64), np.zeros(2))


# ------------------------ oracles: the drift excess is not fooled by a drifting market


def _drift_market(
    seed: int, days: int, mu: float, noise: float
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    rows = days * HOURS_PER_DAY
    realised = mu + noise * rng.standard_normal(rows)
    return realised, day_ids_for(days, 1)


def test_a_drift_only_market_pays_long_only_but_not_the_drift_excess() -> None:
    """Profit without signal: the comparator earns it, so the excess interval hugs zero or below."""

    days = 120
    realised, day_ids = _drift_market(21, days, mu=0.0004, noise=0.004)
    rng = np.random.default_rng(22)
    useless_forecast = 0.004 * rng.standard_normal(realised.shape[0])  # no information
    drift = np.full(realised.shape[0], 0.0004)  # the causal drift estimate is positive

    comparator = daily_mean(day_ids, drift_sign_payoff(drift, realised))
    excess = daily_mean(
        day_ids,
        drift_excess_payoff(
            useless_forecast,
            drift,
            realised,
            k=1,
            round_trip_cost=ROUND_TRIP,
            cost_multiplier=1,
        ),
    )
    interval = block_mean_interval(excess.values, block_days=5, resamples=400, seed=0)

    assert float(np.mean(comparator.values)) > 0.0  # long-only "profits"
    assert not interval.lower > 0.0  # never a SIGNAL_CANDIDATE-style positive interval
    assert interval.mean < 0.0  # paying cost for a coin flip loses to holding


def test_a_planted_forecast_has_a_positive_drift_excess_at_cost_x1_and_x2() -> None:
    days = 120
    rng = np.random.default_rng(23)
    rows = days * HOURS_PER_DAY
    day_ids = day_ids_for(days, 1)
    conditional_mean = 0.006 * rng.standard_normal(rows)  # information well above 14 bp
    realised = conditional_mean + 0.004 * rng.standard_normal(rows)
    drift = np.full(rows, 1e-5)

    results = {}
    for multiplier in (1, 2):
        excess = daily_mean(
            day_ids,
            drift_excess_payoff(
                conditional_mean,
                drift,
                realised,
                k=1,
                round_trip_cost=ROUND_TRIP,
                cost_multiplier=multiplier,
            ),
        )
        results[multiplier] = block_mean_interval(
            excess.values, block_days=5, resamples=400, seed=0
        )

    assert results[1].lower > 0.0
    assert results[2].mean > 0.0
    assert results[2].mean < results[1].mean  # more cost, less excess
