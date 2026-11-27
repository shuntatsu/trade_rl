"""Synthetic numerical regression oracles for dimensionless Sharpe moments."""

from __future__ import annotations

import math
import sys

import pytest

from trade_rl.evaluation.comparison.multiple_testing import (
    SharpeMoments,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_moments_from_returns,
    validate_sharpe_moments,
)


@pytest.mark.parametrize("skew", [1e155, -1e155, 1e200, -1e200, sys.float_info.max])
@pytest.mark.parametrize("kurtosis", [3.0, -1.0, sys.float_info.max])
def test_impossible_pearson_moments_refused(skew: float, kurtosis: float) -> None:
    with pytest.raises(ValueError, match="kurtosis"):
        SharpeMoments(0.0, 20, skew, kurtosis)
    with pytest.raises(ValueError, match="kurtosis"):
        probabilistic_sharpe_ratio(0.0, 20, skew, kurtosis)


@pytest.mark.parametrize(
    "scale", [5e-324, 1e-308, 1e-200, 1e-100, 1.0, 1e100, 1e200, 1e308]
)
def test_symmetric_extreme_scales(scale: float) -> None:
    moments = sharpe_moments_from_returns([-scale, scale])
    assert moments == SharpeMoments(0.0, 2, 0.0, 1.0)


@pytest.mark.parametrize("scale", [1e-200, 1e-100, 1.0, 1e100, 1e200, 1e307])
def test_asymmetric_scale_invariance(scale: float) -> None:
    # [0, 0, 0, 4]: mean 1, sample std 2, population skew 2/sqrt(3).
    moments = sharpe_moments_from_returns([0.0, 0.0, 0.0, 4.0 * scale])
    assert moments.n_obs == 4
    assert moments.sharpe == pytest.approx(0.5, abs=1e-12)
    assert moments.skew == pytest.approx(2 / math.sqrt(3), abs=1e-12)
    assert moments.kurtosis == pytest.approx(7 / 3, abs=1e-12)


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0 + 1e-13],
        [1e308, 1e308],
        [1e-300, 1e-300],
        [float("inf"), 1.0],
    ],
)
def test_existing_refusals_at_extreme_scales(values: list[float]) -> None:
    with pytest.raises(ValueError):
        sharpe_moments_from_returns(values)


def test_finite_pearson_control() -> None:
    assert validate_sharpe_moments(0.0, 20, 1.0, 3.0) == (0.0, 20, 1.0, 3.0)
    assert probabilistic_sharpe_ratio(0.0, 20, 1e150, 1e300) == 0.5


def test_existing_statistical_formula_oracles() -> None:
    assert probabilistic_sharpe_ratio(0.2, 100, 0.0, 3.0) == pytest.approx(
        0.9756019368039752, abs=1e-12
    )
    assert expected_maximum_sharpe(2, 1.0) == pytest.approx(
        0.5197553442805937, abs=1e-12
    )
    assert deflated_sharpe_ratio(0.1, 250, -0.5, 4.0, 10, 0.04) == pytest.approx(
        0.0004870784890680868, abs=1e-12
    )
