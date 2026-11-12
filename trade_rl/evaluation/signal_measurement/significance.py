"""One-sided test of a positive mean of a dependent day series (Issue #1146, SD1).

The equal-weighted cosine (EWC) test of Lazarus, Lewis, Stock and Watson (2018,
"HAR Inference: Recommendations for Practice"): the long-run variance of the series
is the average of its squared projections on the first ``nu`` cosine functions, and
the studentised mean is referred to a Student-t law with ``nu`` degrees of freedom.
``nu = floor(0.4 * n ** (2 / 3))`` is that paper's rule; it is computed in exact
integer arithmetic so that no rounding decides a degree of freedom.

Unlike a block bootstrap whose variance estimate is treated as known, the Student-t
reference carries the sampling error of the variance estimate, which is what keeps
the test's size at the small levels a multiple-testing correction works at. On
seeded synthetic series (``tools/issue1146_significance_calibration.py``) it holds
its size at 60 and at 1460 days where the centred moving-block bootstrap does not.

The test is deterministic: no resampling, no seed. It fails closed. The p-value is
exactly 1.0 when the mean is not positive, when the series is too short to give one
cosine term (fewer than four days, including an empty series) and when the long-run
variance estimate is not positive (a constant series).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.evaluation.signal_measurement.ic import SignalMeasurementError


def ewc_terms(n_days: int) -> int:
    """``floor(0.4 * n_days ** (2 / 3))``: the largest ``nu`` with ``125 nu^3 <= 8 n^2``."""

    if isinstance(n_days, bool) or not isinstance(n_days, int) or n_days < 0:
        raise SignalMeasurementError("n_days must be a non-negative integer")
    # nu**3 is an integer, so 125 nu**3 <= 8 n**2 exactly when nu**3 <= floor(8 n**2 / 125);
    # no floating point decides a degree of freedom and any size of n is cheap.
    return _integer_cube_root(8 * n_days * n_days // 125)


def _integer_cube_root(value: int) -> int:
    """The largest integer ``r`` with ``r**3 <= value``, by integer Newton iteration."""

    if value < 2:
        return value
    root = 1 << ((value.bit_length() + 2) // 3)
    while True:
        better = (2 * root + value // (root * root)) // 3
        if better >= root:
            return root
        root = better


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz evaluation of the continued fraction of the incomplete beta function."""

    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        even = m * (b - m) * x / ((a + 2 * m - 1.0) * (a + 2 * m))
        d = 1.0 + even * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + even / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        odd = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1.0))
        d = 1.0 + odd * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + odd / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_sf(statistic: float, degrees_of_freedom: int) -> float:
    """``P(T > statistic)`` for a Student-t variable (upper tail probability)."""

    if (
        isinstance(degrees_of_freedom, bool)
        or not isinstance(degrees_of_freedom, int)
        or degrees_of_freedom < 1
    ):
        raise SignalMeasurementError("degrees_of_freedom must be a positive integer")
    value = float(statistic)
    if math.isnan(value):
        raise SignalMeasurementError("statistic must not be NaN")
    if math.isinf(value):
        return 0.0 if value > 0.0 else 1.0
    dof = float(degrees_of_freedom)
    tail = 0.5 * _regularized_incomplete_beta(
        0.5 * dof, 0.5, dof / (dof + value * value)
    )
    return tail if value >= 0.0 else 1.0 - tail


@dataclass(frozen=True, slots=True)
class EwcMeanTest:
    """Mean of a day series and its one-sided EWC test against a mean of zero.

    ``mean`` is ``None`` for an empty series; ``standard_error`` and ``statistic``
    are ``None`` when no positive long-run variance estimate exists.
    """

    n_days: int
    cosine_terms: int
    mean: float | None
    standard_error: float | None
    statistic: float | None
    p_value_one_sided: float


def ewc_mean_test(daily_values: object) -> EwcMeanTest:
    """One-sided EWC test that the mean of ``daily_values`` is positive.

    The series is taken in the given (time) order. An empty series is accepted and
    cannot reject: its p-value is exactly 1.0, like every other case in which the
    test has nothing to stand on (see the module docstring).
    """

    array = np.asarray(daily_values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "fiu"):
        raise SignalMeasurementError("daily_values must be a 1-D numeric array")
    values = array.astype(np.float64)
    if not np.isfinite(values).all():
        raise SignalMeasurementError("daily_values must be finite")
    n = int(values.shape[0])
    terms = ewc_terms(n)
    if n == 0:
        return EwcMeanTest(0, 0, None, None, None, 1.0)
    mean = float(values.mean())
    if terms < 1 or float(values.max()) == float(values.min()):
        # Too short for one cosine term, or constant: no variance estimate exists.
        return EwcMeanTest(n, terms, mean, None, None, 1.0)
    days = np.arange(1, n + 1, dtype=np.float64)[:, None]
    orders = np.arange(1, terms + 1, dtype=np.float64)[None, :]
    basis = math.sqrt(2.0 / n) * np.cos(math.pi * orders * (days - 0.5) / n)
    # The cosines sum to zero over the days, so centring changes nothing in exact
    # arithmetic; it keeps a large common level from leaking into the projections.
    projections = (values - mean) @ basis
    long_run_variance = float(np.dot(projections, projections)) / terms
    if not long_run_variance > 0.0 or not math.isfinite(long_run_variance):
        return EwcMeanTest(n, terms, mean, None, None, 1.0)
    standard_error = math.sqrt(long_run_variance / n)
    statistic = mean / standard_error
    p_value = student_t_sf(statistic, terms) if mean > 0.0 else 1.0
    return EwcMeanTest(n, terms, mean, standard_error, statistic, p_value)


__all__ = ["EwcMeanTest", "ewc_mean_test", "ewc_terms", "student_t_sf"]
