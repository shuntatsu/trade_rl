from __future__ import annotations

import numpy as np
import pytest

from trade_rl.risk.portfolio import PortfolioRiskConfig, PortfolioRiskModel


def test_portfolio_risk_applies_liquidity_concentration_and_net_limits() -> None:
    model = PortfolioRiskModel(
        PortfolioRiskConfig(
            max_abs_weight=0.30,
            max_net_exposure=0.20,
            max_position_to_market_notional=0.10,
        )
    )

    result = model.constrain(
        np.array([0.8, -0.1]),
        portfolio_value=100.0,
        market_notional=np.array([100.0, 1000.0]),
    )

    assert abs(result.weights[0]) <= 0.10 + 1e-12
    assert np.max(np.abs(result.weights)) <= 0.30 + 1e-12
    assert abs(result.weights.sum()) <= 0.20 + 1e-12
    assert result.was_constrained


def test_portfolio_risk_applies_volatility_beta_and_stress_limits() -> None:
    model = PortfolioRiskModel(
        PortfolioRiskConfig(
            volatility_target=0.10,
            max_abs_beta=0.05,
            max_stress_loss=0.02,
        )
    )
    covariance = np.eye(2) * 0.25

    result = model.constrain(
        np.array([0.5, 0.5]),
        portfolio_value=100.0,
        market_notional=np.array([10000.0, 10000.0]),
        covariance=covariance,
        beta=np.array([1.0, 1.0]),
        stress_losses=np.array([0.2, 0.2]),
    )

    assert result.portfolio_volatility <= 0.10 + 1e-12
    assert abs(result.portfolio_beta) <= 0.05 + 1e-12
    assert result.stress_loss <= 0.02 + 1e-12


def test_portfolio_risk_fails_closed_on_non_finite_inputs() -> None:
    model = PortfolioRiskModel()
    with pytest.raises(ValueError, match="finite"):
        model.constrain(
            np.array([np.nan]),
            portfolio_value=100.0,
            market_notional=np.array([100.0]),
        )


@pytest.mark.parametrize("diagonal", [-0.25, -1e-300])
def test_portfolio_risk_rejects_negative_projected_covariance_variance(
    diagonal: float,
) -> None:
    model = PortfolioRiskModel(PortfolioRiskConfig(volatility_target=0.10))

    with pytest.raises(ValueError, match="negative portfolio variance"):
        model.constrain(
            np.array([0.5]),
            portfolio_value=100.0,
            market_notional=np.array([1000.0]),
            covariance=np.array([[diagonal]]),
        )


def test_portfolio_risk_preserves_near_singular_psd_covariance() -> None:
    target = np.array([0.5, -1.0])
    covariance = np.array([[1.0, 0.5], [0.5, 0.25 + np.finfo(np.float64).eps / 2.0]])
    model = PortfolioRiskModel(PortfolioRiskConfig(volatility_target=0.10))

    result = model.constrain(
        target,
        portfolio_value=100.0,
        market_notional=np.array([1000.0, 1000.0]),
        covariance=covariance,
    )

    np.testing.assert_array_equal(result.weights, target)
    assert 0.0 < result.portfolio_volatility < 1e-6
    assert result.was_constrained is False


def test_portfolio_risk_clamps_only_cancellation_scale_negative_variance() -> None:
    covariance_factor = np.array([-0.11799648404111936, 0.3431629446879452])
    covariance = np.outer(covariance_factor, covariance_factor)
    target = np.array([-0.15421573575229858, -0.05302703827516377])
    assert float(target @ covariance @ target) < 0.0

    result = PortfolioRiskModel(PortfolioRiskConfig(volatility_target=0.10)).constrain(
        target,
        portfolio_value=100.0,
        market_notional=np.array([1000.0, 1000.0]),
        covariance=covariance,
    )

    np.testing.assert_array_equal(result.weights, target)
    assert result.portfolio_volatility == 0.0
    assert result.was_constrained is False
