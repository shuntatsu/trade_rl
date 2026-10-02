from __future__ import annotations

from tests.evaluation.experiments.test_delta import _resolved
from trade_rl.evaluation.experiments.contracts import ControlledFactor
from trade_rl.evaluation.experiments.delta import FACTOR_RULES
from trade_rl.strategies.rl.ppo_training import (
    PPO_DEFAULT_GAE_LAMBDA,
    PPO_REWARD_SCHEMA,
)


def test_ppo_discount_factor_changes_only_gamma() -> None:
    rule = FACTOR_RULES[ControlledFactor.PPO_DISCOUNT]

    assert rule.allowed_paths == frozenset({("ppo_gamma",)})
    assert rule.unaffected_strategies == frozenset(
        {
            "cash",
            "constant_long",
            "constant_short",
            "trend",
            "mean_reversion",
            "ridge24",
            "lightgbm24",
        }
    )


def test_resolved_run_v6_binds_fixed_reward_contract_and_gamma() -> None:
    config = _resolved(
        schema_version="resolved_run_config_v6",
        ppo_gamma=0.9975,
    )

    payload = config.to_payload()

    assert payload["schema_version"] == "resolved_run_config_v6"
    assert payload["ppo_reward_schema"] == PPO_REWARD_SCHEMA
    assert payload["ppo_gamma"] == 0.9975
    assert payload["ppo_gae_lambda"] == PPO_DEFAULT_GAE_LAMBDA
