"""Regression boundaries for synthetic SD1 measurements (#1146, PR #1202)."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from trade_rl.evaluation.signal_measurement.ic import SignalMeasurementError
from trade_rl.evaluation.signal_measurement.payoff import after_cost_payoff
from trade_rl.evaluation.signal_measurement.targets import forward_log_returns


def test_extreme_finite_closes_have_finite_log_differences() -> None:
    closes = np.array([[1e-300], [1e300], [1e-300]], dtype=np.float64)
    expected = math.log(1e300) - math.log(1e-300)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = forward_log_returns(closes, 1)

    assert result.shape == closes.shape
    assert np.isfinite(result[:2]).all()
    np.testing.assert_allclose(result[:2, 0], [expected, -expected], rtol=1e-14)
    assert np.isnan(result[-1, 0])  # unmatured final label stays missing


@pytest.mark.parametrize(
    ("cost", "multiplier", "k"),
    [
        (1e308, 2, 1),  # multiplication of two finite values overflows
        (1e308, 1, 2),  # the entry threshold overflows
        (0.0014, 1, 10**400),  # huge integer must raise a typed error
    ],
)
def test_nonrepresentable_derived_cost_or_threshold_is_rejected(
    cost: float, multiplier: int, k: int
) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(SignalMeasurementError, match="finite"):
            after_cost_payoff(
                np.array([0.1]),
                np.array([0.02]),
                k=k,
                round_trip_cost=cost,
                cost_multiplier=multiplier,
            )


def test_standard_14bps_cost_is_unchanged() -> None:
    score = np.array([0.0014, 0.001401, -0.003, 0.0])
    actual = np.array([0.02, 0.02, -0.01, 0.01])
    out = after_cost_payoff(
        score,
        actual,
        k=1,
        round_trip_cost=0.0014,
        cost_multiplier=1,
    )
    np.testing.assert_allclose(out, [0.0, 0.0186, 0.0086, 0.0])


@pytest.mark.parametrize(
    ("initial", "final"),
    [(1e20, 1e-300), (1e200, 1e-120)],
)
def test_subnormal_intermediate_ratio_preserves_log_return_precision(
    initial: float, final: float
) -> None:
    """A finite subnormal ratio must not round away significant log information."""
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        calculated = forward_log_returns(np.array([[initial], [final]]), 1)
    expected = math.log(final) - math.log(initial)
    assert calculated[0, 0] == pytest.approx(expected, abs=5e-13, rel=0)
    assert np.isnan(calculated[1, 0])


def test_nonrepresentable_round_trip_input_identifies_its_field() -> None:
    with pytest.raises(
        SignalMeasurementError, match="round_trip_cost must be a positive finite"
    ):
        after_cost_payoff(
            np.array([0.1]),
            np.array([0.02]),
            k=1,
            round_trip_cost=10**400,  # type: ignore[arg-type]
            cost_multiplier=1,
        )


def test_near_unity_return_preserves_direct_ratio_precision() -> None:
    # Subtracting two ~230-sized logs loses this one-ULP movement entirely.
    initial = 1e100
    final = float(np.nextafter(initial, np.inf))
    expected = math.log(final / initial)
    assert expected > 0.0
    assert math.log(final) - math.log(initial) == 0.0
    actual = forward_log_returns(np.array([[initial], [final]]), 1)
    assert actual[0, 0] == expected
    assert np.isnan(actual[1, 0])


@pytest.mark.parametrize(
    ("initial", "final", "ratio_is_normal"),
    [
        (1e10, 2.0e-298, False),  # ratio just below the normal boundary
        (1e10, 2.5e-298, True),  # ratio just above the normal boundary
        (1e10, 1.0e-310, False),  # deep subnormal: wrong direct path loses digits
    ],
)
def test_subnormal_boundary_selects_the_intended_log_path(
    initial: float, final: float, ratio_is_normal: bool
) -> None:
    ratio = final / initial
    assert bool(ratio >= np.finfo(np.float64).tiny) is ratio_is_normal
    if not ratio_is_normal:
        # A wrong-branch mutant must not pass because two oracles accidentally agree.
        assert math.log(ratio) != math.log(final) - math.log(initial)
    result = forward_log_returns(np.array([[initial], [final]]), 1)
    # The reference is the log of the *exact* finite binary64 operands,
    # evaluated at higher precision instead of requiring bitwise equality
    # between platform math.log and vectorized NumPy ufunc implementations.
    with localcontext() as context:
        context.prec = 100
        expected = float(
            Decimal.from_float(final).ln() - Decimal.from_float(initial).ln()
        )
    # A two-ULP error budget remains strict for binary64 transcendentals;
    # the deep-subnormal control above differs by ~1e-5 if the wrong path
    # logs the rounded intermediate ratio instead of the stable endpoints.
    assert abs(float(result[0, 0]) - expected) <= 2 * math.ulp(expected)
    assert np.isnan(result[1, 0])


@pytest.mark.parametrize(
    ("k", "multiplier"),
    [(10**400, 1), (1, 10**400)],
)
def test_oversized_derived_parameters_are_typed_errors(k: int, multiplier: int) -> None:
    with pytest.raises(
        SignalMeasurementError, match="derived cost and threshold must be finite"
    ):
        after_cost_payoff(
            np.array([0.1]),
            np.array([0.02]),
            k=k,
            round_trip_cost=0.0014,
            cost_multiplier=multiplier,
        )
