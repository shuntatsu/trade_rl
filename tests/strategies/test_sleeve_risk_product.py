"""Synthetic mixed-scale sleeve variance contributions."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.strategies.allocation.sleeves import (
    SleeveCombination,
    SleeveCombinerConfig,
    SleeveCombinerError,
    SleeveMethod,
    SleeveTarget,
    combine_sleeves,
)


def _combine(
    history: np.ndarray, weights: tuple[float, float], shrinkage: float
) -> SleeveCombination:
    return combine_sleeves(
        (
            SleeveTarget("large", ("A", "B"), (1.0, 0.0)),
            SleeveTarget("small", ("A", "B"), (0.0, 1.0)),
        ),
        symbols=("A", "B"),
        history=history,
        config=SleeveCombinerConfig(
            method=SleeveMethod.FIXED,
            lookback=4,
            ddof=0,
            shrinkage=shrinkage,
            fixed_weights=weights,
        ),
    )


@pytest.mark.parametrize(
    ("large", "small"), [(1e100, 1e-100), (1e90, 1e-110), (1.0, 0.01)]
)
@pytest.mark.parametrize("weights", [(0.5, 0.5), (0.75, 0.25)])
def test_mixed_scale_diagonal_risk_contributions_remain_nonzero(
    large: float, small: float, weights: tuple[float, float]
) -> None:
    history = np.array(
        [[-large, -small], [-large, small], [large, -small], [large, small]]
    )
    result = _combine(history, weights, 1.0)
    # Independent population covariance: diagonal variances large**2 and small**2.
    for actual, size, weight in zip(
        result.risk_contributions, (large, small), weights, strict=True
    ):
        expected = float((Fraction(size) * Fraction(weight)) ** 2)
        assert expected > 0.0
        assert math.isclose(actual, expected, rel_tol=2e-15, abs_tol=0.0)


def test_nonzero_normalized_risk_weight_underflow_is_refused() -> None:
    history = np.array(
        [
            [-1e150, -1e-170],
            [-1e150, -1e-170],
            [1e150, 1e-170],
            [1e150, 1e-170],
        ]
    )
    with pytest.raises(SleeveCombinerError, match="risk weights underflow"):
        _combine(history, (1.0, 1e-20), 0.0)


def test_exact_zero_risk_weight_remains_zero() -> None:
    history = np.array([[-1.0, -0.1], [-1.0, 0.1], [1.0, -0.1], [1.0, 0.1]])
    result = _combine(history, (1.0, 0.0), 1.0)
    assert result.risk_contributions == (1.0, 0.0)


def test_negative_risk_contribution_remains_signed() -> None:
    history = np.array([[-2.0, 1.0], [-2.0, 1.0], [2.0, -1.0], [2.0, -1.0]])
    result = _combine(history, (0.5, 0.5), 0.0)
    assert result.risk_contributions == (0.5, -0.25)


def _single_gross_cap(target: tuple[float, float], cap: float) -> SleeveCombination:
    return combine_sleeves(
        (SleeveTarget("only", ("A", "B"), target),),
        symbols=("A", "B"),
        history=np.array([[-1.0], [1.0]]),
        config=SleeveCombinerConfig(
            method=SleeveMethod.FIXED,
            lookback=2,
            ddof=0,
            shrinkage=0.0,
            fixed_weights=(1.0,),
            max_gross=cap,
        ),
    )


@pytest.mark.parametrize("cap", [3 * math.ulp(0.0), 1e-310, 0.1])
def test_gross_cap_never_rounds_above_exact_limit(cap: float) -> None:
    result = _single_gross_cap((1.0, -1.0), cap)
    left, right = result.combined_weights
    share = Fraction(cap) / 2
    assert left == -right and left > 0.0
    assert Fraction(left) <= share
    assert Fraction(math.nextafter(left, math.inf)) > share
    assert abs(Fraction(left)) + abs(Fraction(right)) <= Fraction(cap)
    assert result.diagnostics.gross_cap_binding


def test_gross_cap_refuses_nonzero_component_rounded_to_zero() -> None:
    with pytest.raises(SleeveCombinerError) as info:
        _single_gross_cap((1.0, 1.0), math.ulp(0.0))
    assert info.value.reason == "non_representable"


def test_gross_cap_detects_exact_excess_hidden_by_float_sum() -> None:
    target = (1.0, 2.0**-53)
    cap = 1.0
    assert sum(target) == cap
    assert sum(map(Fraction, target)) > Fraction(cap)
    result = _single_gross_cap(target, cap)
    assert result.diagnostics.gross_cap_binding
    assert sum(map(Fraction, result.combined_weights)) <= Fraction(cap)
    assert all(value > 0.0 for value in result.combined_weights)


def test_gross_cap_preserves_signed_proportional_and_nonbinding_controls() -> None:
    bound = _single_gross_cap((0.75, -0.25), 0.5)
    assert bound.combined_weights == (0.375, -0.125)
    assert bound.diagnostics.gross_cap_binding
    loose = _single_gross_cap((0.1, -0.2), 0.4)
    assert loose.combined_weights == (0.1, -0.2)
    assert not loose.diagnostics.gross_cap_binding
