from __future__ import annotations

import inspect

from trade_rl.rl.environment import ResidualMarketEnv

EXPECTED_PARAMETERS = (
    "self",
    "dataset",
    "trend_strategy",
    "market_input_resolver",
    "alpha_provider",
    "alpha_enabled",
    "alpha_artifact_digest",
    "alpha_contract",
    "factor_basis",
    "factor_basis_provider",
    "factor_artifact_digest",
    "factor_count",
    "action_spec",
    "composer",
    "pre_trade_risk",
    "portfolio_risk",
    "portfolio_risk_inputs_provider",
    "normalizer",
    "sequence_normalizer",
    "execution_rule_stress",
    "config",
)


def test_residual_market_environment_constructor_signature_is_stable() -> None:
    parameters = tuple(
        inspect.signature(ResidualMarketEnv.__init__).parameters.values()
    )

    assert tuple(parameter.name for parameter in parameters) == EXPECTED_PARAMETERS
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters[2:]
    )
    assert parameters[1].default is inspect.Parameter.empty
    assert all(
        parameter.default is not inspect.Parameter.empty for parameter in parameters[2:]
    )
