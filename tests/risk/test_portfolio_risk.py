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
