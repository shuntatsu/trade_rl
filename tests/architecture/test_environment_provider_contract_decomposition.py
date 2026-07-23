from __future__ import annotations

import importlib
import inspect

from trade_rl.rl.environment import ResidualMarketEnv


def test_environment_provider_contract_is_owned_by_dedicated_module() -> None:
    module = importlib.import_module("trade_rl.rl.environment_provider_contract")

    contract = module.EnvironmentProviderContract
    builder = module.EnvironmentProviderContractBuilder

    assert contract.__module__ == "trade_rl.rl.environment_provider_contract"
    assert builder.__module__ == "trade_rl.rl.environment_provider_contract"


def test_environment_constructor_delegates_provider_contract() -> None:
    source = inspect.getsource(ResidualMarketEnv.__init__)

    assert "EnvironmentProviderContractBuilder(" in source
    for forbidden in (
        "MarketInputResolver(",
        "_resolve_provider_digest(",
        "_validated_static_basis(",
        "_resolve_factor_count(",
        'getattr(provider, "minimum_index", 0)',
    ):
        assert forbidden not in source
    assert len(source.splitlines()) <= 245


def test_environment_facade_does_not_own_provider_helpers() -> None:
    for helper in (
        "_resolve_provider_digest",
        "_validated_static_basis",
        "_resolve_factor_count",
    ):
        assert not hasattr(ResidualMarketEnv, helper)
