from __future__ import annotations

import inspect

from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_portfolio_risk_contract import (
    EnvironmentPortfolioRiskContract,
    EnvironmentPortfolioRiskContractBuilder,
)


def test_portfolio_risk_contract_module_owns_maintained_boundary() -> None:
    assert EnvironmentPortfolioRiskContract.__module__ == (
        "trade_rl.rl.environment_portfolio_risk_contract"
    )
    assert EnvironmentPortfolioRiskContractBuilder.__module__ == (
        "trade_rl.rl.environment_portfolio_risk_contract"
    )


def test_environment_constructor_delegates_portfolio_risk_contract() -> None:
    source = inspect.getsource(ResidualMarketEnv.__init__)

    assert source.count("EnvironmentPortfolioRiskContractBuilder(") == 1
    for forbidden in (
        "RollingPortfolioRiskInputsProvider()",
        "portfolio_risk_inputs_provider.identity_digest",
        "portfolio risk inputs minimum_index is invalid",
    ):
        assert forbidden not in source
    assert len(source.splitlines()) <= 220


def test_portfolio_risk_builder_preserves_validation_order() -> None:
    source = inspect.getsource(EnvironmentPortfolioRiskContractBuilder.build)

    digest_position = source.index("require_sha256(")
    minimum_position = source.index("provider.minimum_index")
    aggregation_position = source.index("max(minimum_start_index, provider_minimum)")

    assert digest_position < minimum_position < aggregation_position
