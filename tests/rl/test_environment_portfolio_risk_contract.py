from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.risk.inputs import RollingPortfolioRiskInputsProvider
from trade_rl.risk.portfolio import PortfolioRiskConfig, PortfolioRiskModel
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.environment_portfolio_risk_contract import (
    EnvironmentPortfolioRiskContract,
    EnvironmentPortfolioRiskContractBuilder,
)
from trade_rl.rl.rewards import AbsoluteGrowthRewardConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


class _Dataset:
    n_bars = 10


class _Provider:
    identity_digest = "0" * 64

    def __init__(self, minimum_index: object) -> None:
        self._minimum_index = minimum_index
        self.minimum_index_accessed = False

    @property
    def minimum_index(self) -> object:
        self.minimum_index_accessed = True
        return self._minimum_index


class _InvalidDigestProvider(_Provider):
    identity_digest = "invalid"


def _advanced_model() -> PortfolioRiskModel:
    return PortfolioRiskModel(PortfolioRiskConfig(volatility_target=0.1))


def _market() -> MarketDataset:
    n_bars = 40
    close = np.column_stack(
        [
            np.linspace(100.0, 120.0, n_bars),
            np.linspace(100.0, 90.0, n_bars),
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("A", "B"),
        timestamps=np.datetime64("2026-01-01T01:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full((n_bars, 2), 1_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_builder_creates_default_model_without_an_unneeded_provider() -> None:
    contract = EnvironmentPortfolioRiskContractBuilder(
        _Dataset(),
        portfolio_risk=None,
        inputs_provider=None,
    ).build(minimum_start_index=3)

    assert isinstance(contract, EnvironmentPortfolioRiskContract)
    assert isinstance(contract.portfolio_risk, PortfolioRiskModel)
    assert contract.inputs_provider is None
    assert contract.minimum_start_index == 3


def test_builder_preserves_supplied_model_and_provider_identities() -> None:
    model = PortfolioRiskModel()
    provider = _Provider(4)

    contract = EnvironmentPortfolioRiskContractBuilder(
        _Dataset(),
        portfolio_risk=model,
        inputs_provider=provider,
    ).build(minimum_start_index=2)

    assert contract.portfolio_risk is model
    assert contract.inputs_provider is provider
    assert contract.minimum_start_index == 4


def test_builder_selects_rolling_provider_for_advanced_risk() -> None:
    model = _advanced_model()

    contract = EnvironmentPortfolioRiskContractBuilder(
        _market(),
        portfolio_risk=model,
        inputs_provider=None,
    ).build(minimum_start_index=3)

    assert contract.portfolio_risk is model
    assert isinstance(contract.inputs_provider, RollingPortfolioRiskInputsProvider)
    assert contract.minimum_start_index == contract.inputs_provider.minimum_index


def test_builder_keeps_larger_existing_minimum_index() -> None:
    provider = _Provider(4)

    contract = EnvironmentPortfolioRiskContractBuilder(
        _Dataset(),
        portfolio_risk=PortfolioRiskModel(),
        inputs_provider=provider,
    ).build(minimum_start_index=7)

    assert contract.minimum_start_index == 7


def test_builder_validates_digest_before_reading_minimum_index() -> None:
    provider = _InvalidDigestProvider(4)

    with pytest.raises(
        ValueError, match="portfolio_risk_inputs_provider.identity_digest"
    ):
        EnvironmentPortfolioRiskContractBuilder(
            _Dataset(),
            portfolio_risk=PortfolioRiskModel(),
            inputs_provider=provider,
        ).build(minimum_start_index=2)

    assert provider.minimum_index_accessed is False


@pytest.mark.parametrize("minimum_index", [True, 1.5, -1, _Dataset.n_bars])
def test_builder_rejects_invalid_provider_minimum_index(
    minimum_index: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="portfolio risk inputs minimum_index is invalid",
    ):
        EnvironmentPortfolioRiskContractBuilder(
            _Dataset(),
            portfolio_risk=PortfolioRiskModel(),
            inputs_provider=_Provider(minimum_index),
        ).build(minimum_start_index=2)


def test_environment_uses_the_portfolio_risk_contract_for_advanced_inputs() -> None:
    model = _advanced_model()
    env = ResidualMarketEnv(
        _market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        portfolio_risk=model,
        config=ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            episode_bars=8,
            decision_every=2,
            reward=AbsoluteGrowthRewardConfig(),
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )

    assert env.portfolio_risk is model
    assert isinstance(
        env.portfolio_risk_inputs_provider, RollingPortfolioRiskInputsProvider
    )
    assert env.minimum_start_index >= env.portfolio_risk_inputs_provider.minimum_index
    assert env._digest_payload()["portfolio_risk_inputs_digest"] == (
        env.portfolio_risk_inputs_provider.identity_digest
    )
