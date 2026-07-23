from __future__ import annotations

from trade_rl.rl.environment_provider_contract import (
    EnvironmentProviderContractBuilder,
)


class BooleanFactorCountProvider:
    n_factors = True


def test_boolean_provider_factor_count_is_not_inferred() -> None:
    assert (
        EnvironmentProviderContractBuilder._resolve_factor_count(
            factor_count=None,
            provider=BooleanFactorCountProvider(),
        )
        == 0
    )
