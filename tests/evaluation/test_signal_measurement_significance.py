"""SD1 (#1146): the one-sided EWC mean test, checked against independent references."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.evaluation.signal_measurement import (
    EwcMeanTest,
    SignalMeasurementError,
    ewc_mean_test,
    ewc_terms,
    student_t_sf,
)


def _t_density(x: np.ndarray, dof: int) -> np.ndarray:
    log_norm = (
        math.lgamma(0.5 * (dof + 1))
        - math.lgamma(0.5 * dof)
        - 0.5 * math.log(dof * math.pi)
    )
    return np.exp(log_norm - 0.5 * (dof + 1) * np.log1p(x * x / dof))


def _integrated_sf(t: float, dof: int) -> float:
    """Upper tail by Simpson's rule on ``[0, t]``: 0.5 minus the central mass."""

    grid = np.linspace(0.0, abs(t), 20001)
    density = _t_density(grid, dof)
    step = grid[1] - grid[0]
    weights = np.ones_like(grid)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    central = float(step / 3.0 * np.dot(weights, density))
    return 0.5 - central if t >= 0.0 else 0.5 + central


def test_the_number_of_cosine_terms_is_the_exact_integer_rule() -> None:
    for n in range(0, 3000):
        expected = max(k for k in range(0, 200) if 125 * k**3 <= 8 * n * n)
        assert ewc_terms(n) == expected, n
    assert [ewc_terms(n) for n in (0, 1, 3, 4, 60, 124, 125, 1460)] == [
        0,
        0,
        0,
        1,
        6,
        9,
        10,  # 0.4 * 125 ** (2 / 3) is exactly 10: floating point must not make it 9
        51,
    ]
    for bad in (-1, 1.0, True, "4"):
        with pytest.raises(SignalMeasurementError):
            ewc_terms(bad)  # type: ignore[arg-type]


def test_the_student_t_tail_matches_closed_forms_and_numerical_integration() -> None:
    for t in (-4.0, -0.3, 0.0, 0.7, 2.5, 30.0):
        assert student_t_sf(t, 1) == pytest.approx(0.5 - math.atan(t) / math.pi)
        assert student_t_sf(t, 2) == pytest.approx(
            0.5 - t / (2.0 * math.sqrt(2.0 + t * t))
        )
    for dof in (3, 6, 51, 400):
        for t in (-2.0, 0.4, 1.7, 2.6, 4.5):
            assert student_t_sf(t, dof) == pytest.approx(
                _integrated_sf(t, dof), abs=1e-9
            ), (dof, t)
        assert student_t_sf(1.3, dof) + student_t_sf(-1.3, dof) == pytest.approx(1.0)
    # Heavier tails than the normal law, converging to it.
    normal = 0.5 * math.erfc(2.5 / math.sqrt(2.0))
    assert student_t_sf(2.5, 6) > student_t_sf(2.5, 51) > normal
    assert student_t_sf(2.5, 100_000) == pytest.approx(normal, rel=1e-3)
    assert student_t_sf(math.inf, 5) == 0.0 and student_t_sf(-math.inf, 5) == 1.0
    for bad in (0, -1, 2.5, True):
        with pytest.raises(SignalMeasurementError):
            student_t_sf(1.0, bad)  # type: ignore[arg-type]
    with pytest.raises(SignalMeasurementError):
        student_t_sf(math.nan, 3)


def test_the_test_equals_a_slow_reference_written_from_the_definition() -> None:
    rng = np.random.default_rng(4)
    series = 0.02 + 0.1 * rng.standard_normal(60)

    result = ewc_mean_test(series)

    n = 60
    terms = 6
    mean = sum(float(v) for v in series) / n
    squares = 0.0
    for j in range(1, terms + 1):
        projection = 0.0
        for t in range(1, n + 1):
            projection += float(series[t - 1]) * math.cos(math.pi * j * (t - 0.5) / n)
        squares += (math.sqrt(2.0 / n) * projection) ** 2
    standard_error = math.sqrt(squares / terms / n)
    assert result.n_days == n and result.cosine_terms == terms
    assert result.mean == pytest.approx(mean)
    assert result.standard_error == pytest.approx(standard_error)
    assert result.statistic == pytest.approx(mean / standard_error)
    assert result.p_value_one_sided == pytest.approx(
        _integrated_sf(mean / standard_error, terms), abs=1e-9
    )
    assert 0.0 < result.p_value_one_sided < 0.5


def test_no_input_that_cannot_support_a_test_can_reject() -> None:
    empty = ewc_mean_test([])
    assert empty == EwcMeanTest(0, 0, None, None, None, 1.0)
    assert ewc_mean_test(np.array([], dtype=np.float64)).p_value_one_sided == 1.0
    # One to three days give no cosine term, however large the mean.
    for days in (1, 2, 3):
        short = ewc_mean_test([5.0 + day for day in range(days)])
        assert short.p_value_one_sided == 1.0 and short.cosine_terms == 0
        assert short.n_days == days and short.statistic is None
    # A constant series has no variance estimate.
    constant = ewc_mean_test([0.3] * 50)
    assert constant.p_value_one_sided == 1.0 and constant.standard_error is None
    assert ewc_mean_test([0.0] * 50).p_value_one_sided == 1.0
    # A non-positive mean is never significant, and its statistic is still reported.
    rng = np.random.default_rng(2)
    negative = ewc_mean_test(-0.5 + 0.01 * rng.standard_normal(80))
    assert negative.p_value_one_sided == 1.0
    assert negative.statistic is not None and negative.statistic < 0.0
    for bad in ([[1.0, 2.0]], [1.0, math.nan], [1.0, math.inf], ["a", "b"], 3.0):
        with pytest.raises(SignalMeasurementError):
            ewc_mean_test(bad)


def test_the_test_is_deterministic_scale_free_and_monotone_in_the_mean() -> None:
    rng = np.random.default_rng(6)
    series = 0.01 + 0.1 * rng.standard_normal(400)

    base = ewc_mean_test(series)

    assert ewc_mean_test(series) == base
    assert ewc_mean_test(tuple(float(v) for v in series)) == base
    scaled = ewc_mean_test(37.0 * series)
    assert scaled.p_value_one_sided == pytest.approx(base.p_value_one_sided)
    shifted = ewc_mean_test(series + 0.02)
    assert shifted.standard_error == pytest.approx(base.standard_error)
    assert shifted.p_value_one_sided < base.p_value_one_sided
    strong = ewc_mean_test(0.5 + 0.01 * rng.standard_normal(400))
    assert strong.p_value_one_sided < 1e-12
