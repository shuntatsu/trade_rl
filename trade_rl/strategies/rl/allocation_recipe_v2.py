"""Explicit allocation-state recipe; legacy serialization stays separate."""

from __future__ import annotations

from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_payload,
)


def allocation_recipe_payload_v2(
    contract: AllocationActionContract,
    feature_names: tuple[str, ...],
    *,
    allocator: AfterCostTargetAllocator,
    expected_horizon_seconds: int,
    runtime_profile: AllocationRuntimeProfile,
    observation_schema: AllocationObservationSchema,
) -> dict[str, object]:
    if (
        not isinstance(observation_schema, AllocationObservationSchema)
        or observation_schema.feature_names != feature_names
        or observation_schema.initial_capital != runtime_profile.initial_capital
        or observation_schema.episode_steps
        != runtime_profile.economic_horizon_seconds
        // runtime_profile.decision_interval_seconds
    ):
        raise ValueError(
            "v2 observation features, capital and horizon must match runtime"
        )
    legacy = allocation_recipe_payload(
        contract,
        feature_names,
        allocator=allocator,
        expected_horizon_seconds=expected_horizon_seconds,
        runtime_profile=runtime_profile,
    )
    return legacy | {
        "schema": "allocation_ppo_recipe_v2",
        "observation": observation_schema.payload(),
        "terminal_observation": "zero_sentinel_on_true_termination_v1",
        "truncation": "not_generated",
        "bootstrap": "nonterminal_rollout_only",
    }


__all__ = ["allocation_recipe_payload_v2"]
