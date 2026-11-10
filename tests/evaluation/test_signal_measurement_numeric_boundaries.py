"""Regression boundaries for synthetic SD1 measurements (#1146, PR #1202)."""

from __future__ import annotations

import math

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
