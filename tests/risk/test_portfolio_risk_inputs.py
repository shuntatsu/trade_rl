from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.risk.portfolio import PortfolioRiskConfig, PortfolioRiskModel
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


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


def test_rolling_portfolio_risk_inputs_are_causal_and_finite() -> None:
    from trade_rl.risk.inputs import (
        RollingPortfolioRiskInputsConfig,
        RollingPortfolioRiskInputsProvider,
    )

    provider = RollingPortfolioRiskInputsProvider(
        RollingPortfolioRiskInputsConfig(
            lookback_bars=60,
            minimum_observations=30,
            benchmark_index=0,
            stress_quantile=0.05,
        )
    )
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
    import pytest

    from trade_rl.risk.inputs import (
        RollingPortfolioRiskInputsConfig,
        RollingPortfolioRiskInputsProvider,
    )

    provider = RollingPortfolioRiskInputsProvider(
        RollingPortfolioRiskInputsConfig(
            lookback_bars=40,
            minimum_observations=30,
        )
    )
    with pytest.raises(ValueError, match="insufficient"):
        provider.inputs(_dataset(), index=20)


def test_environment_wires_causal_inputs_into_advanced_portfolio_risk() -> None:
    from trade_rl.risk.inputs import RollingPortfolioRiskInputsProvider

    dataset = _dataset()
    risk = PortfolioRiskModel(
        PortfolioRiskConfig(
            volatility_target=0.01,
            max_abs_beta=0.15,
            max_stress_loss=0.0005,
        )
    )
    env = ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        action_spec=ActionSpec(
            mode="target_weight",
            alpha_enabled=False,
            risk_tilt_enabled=False,
            target_weight_count=3,
        ),
        portfolio_risk=risk,
        config=ResidualMarketEnvConfig(
            episode_bars=8,
            decision_every=1,
            initial_capital=100_000.0,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )
    assert isinstance(
        env.portfolio_risk_inputs_provider, RollingPortfolioRiskInputsProvider
    )
    env.reset(options={"start_idx": 120, "initial_state_mode": "cash"})
    constrained = env._constrain_target(np.array([0.6, 0.3, -0.1]), env.hybrid)

    assert constrained.was_constrained is True
    assert any(reason.startswith("portfolio:") for reason in constrained.reasons)
    payload = env._digest_payload()
    assert isinstance(payload["portfolio_risk_inputs_digest"], str)
    assert len(payload["portfolio_risk_inputs_digest"]) == 64
    provider = env.portfolio_risk_inputs_provider
    assert provider is not None
    baseline = provider.inputs(dataset, index=120)
    changed = provider.inputs(_dataset(future_shift=0.5), index=120)
    np.testing.assert_allclose(baseline.covariance, changed.covariance)
