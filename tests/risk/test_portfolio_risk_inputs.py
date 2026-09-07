from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.risk.inputs import (
    RollingPortfolioRiskInputsConfig,
    RollingPortfolioRiskInputsProvider,
)
from trade_rl.risk.portfolio import PortfolioRiskConfig, PortfolioRiskModel


def _dataset(*, future_shift: float = 0.0) -> MarketDataset:
    n_bars = 180
    phase = np.arange(n_bars, dtype=np.float64)
    returns = np.column_stack(
        (
            0.0003 + 0.0015 * np.sin(phase / 7.0),
            0.0002 + 0.0010 * np.sin(phase / 7.0 + 0.4),
            -0.0001 + 0.0012 * np.cos(phase / 11.0),
        )
    )
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    if future_shift:
        close[121:] *= 1.0 + future_shift
    open_price = np.vstack((close[0], close[:-1]))
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 3, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full((n_bars, 3), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 3)),
        tradable=np.ones((n_bars, 3), dtype=np.bool_),
        feature_available=np.ones((n_bars, 3, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _provider() -> RollingPortfolioRiskInputsProvider:
    return RollingPortfolioRiskInputsProvider(
        RollingPortfolioRiskInputsConfig(
            lookback_bars=60,
            minimum_observations=30,
            benchmark_index=0,
            stress_quantile=0.05,
        )
    )


def test_rolling_portfolio_risk_inputs_are_causal_and_finite() -> None:
    provider = _provider()
    first = provider.inputs(_dataset(), index=120)
    shifted = provider.inputs(_dataset(future_shift=0.5), index=120)

    np.testing.assert_allclose(first.covariance, shifted.covariance)
    np.testing.assert_allclose(first.beta, shifted.beta)
    np.testing.assert_allclose(first.stress_losses, shifted.stress_losses)
    assert first.as_of_index == 120
    assert first.covariance.shape == (3, 3)
    assert first.beta.shape == (3,)
    assert first.stress_losses.shape == (3,)
    assert np.isfinite(first.covariance).all()
    assert np.isfinite(first.beta).all()
    assert np.all(first.stress_losses <= 0.0)
    assert len(first.digest) == 64
    assert len(provider.identity_digest) == 64


def test_rolling_portfolio_risk_inputs_reject_insufficient_history() -> None:
    provider = RollingPortfolioRiskInputsProvider(
        RollingPortfolioRiskInputsConfig(
            lookback_bars=40,
            minimum_observations=30,
        )
    )
    with pytest.raises(ValueError, match="insufficient"):
        provider.inputs(_dataset(), index=20)


def test_causal_inputs_feed_portfolio_risk_directly() -> None:
    dataset = _dataset()
    inputs = _provider().inputs(dataset, index=120)
    risk = PortfolioRiskModel(
        PortfolioRiskConfig(
            volatility_target=0.01,
            max_abs_beta=0.15,
            max_stress_loss=0.0005,
        )
    )

    constrained = risk.constrain(
        np.array([0.6, 0.3, -0.1]),
        portfolio_value=100_000.0,
        market_notional=np.full(3, 1_000_000_000.0),
        covariance=inputs.covariance,
        beta=inputs.beta,
        stress_losses=inputs.stress_losses,
    )

    assert constrained.was_constrained is True
    assert set(constrained.reasons) & {
        "volatility_target",
        "max_abs_beta",
        "max_stress_loss",
    }
    assert np.isfinite(constrained.weights).all()
