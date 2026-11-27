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
