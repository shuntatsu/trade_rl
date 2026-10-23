"""Same-H verified inference using the distinct native settled environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trade_rl.evaluation.rl_allocation.settled_env import SettledAllocationTradingEnv
from trade_rl.strategies.rl.allocation_artifact import (
    load_allocation_policy,
    read_allocation_policy_manifest,
)
from trade_rl.strategies.rl.allocation_model import (
    AllocationPPOPolicy,
    validate_allocation_model,
)


def _fresh(env: SettledAllocationTradingEnv) -> None:
    if not isinstance(env, SettledAllocationTradingEnv) or env._episode_sequence:
        raise ValueError(
            "settled inference requires a fresh distinct settled environment"
        )
    env.validate_binding()


def _same_recipe(env: SettledAllocationTradingEnv, manifest: dict[str, Any]) -> None:
    if (
        manifest["schema"] != "allocation_ppo_inference_bundle_v6"
        or manifest["recipe_digest"] != env.recipe_digest
        or manifest["recipe"] != env.recipe
    ):
        raise ValueError("policy recipe differs from the exact settled runtime clock")


def load_settled_allocation_policy(
    root: Path, *, env: SettledAllocationTradingEnv, expected_digest: str
) -> AllocationPPOPolicy:
    """Admit exact P/R/H before optional backend deserialization or account reset."""
    _fresh(env)
    recipe_digest = env.recipe_digest
    manifest = read_allocation_policy_manifest(
        root, expected_digest=expected_digest, expected_recipe_digest=recipe_digest
    )
    _same_recipe(env, manifest)
    loaded = load_allocation_policy(
        root, expected_digest=expected_digest, expected_recipe_digest=recipe_digest
    )
    env.validate_binding()
    _same_recipe(env, loaded.manifest)
    return loaded


def run_settled_allocation_policy(
    env: SettledAllocationTradingEnv, policy: AllocationPPOPolicy
) -> dict[str, Any]:
    """Process P actions and native reserve once; retain unqualified diagnostics."""
    _fresh(env)
    if not isinstance(policy, AllocationPPOPolicy):
        raise ValueError("settled inference requires a validated allocation policy")
    _same_recipe(env, policy.manifest)
    validate_allocation_model(policy.model, policy.manifest)
    observation, _ = env.reset()
    transitions = []
    while not env._terminated:
        action = policy.action(observation, runtime_recipe_digest=env.recipe_digest)
        observation, _, done, truncated, info = env.step(action)
        if truncated:
            raise ValueError("settled inference cannot truncate its native horizon")
        transitions.append(json.loads(info["settled_transition"]))
        if done:
            break
    return {
        "schema": "allocation_settled_inference_diagnostic_v1",
        "recipe_digest": env.recipe_digest,
        "transitions": transitions,
        "settlement_witness": env._witness(),
    }
