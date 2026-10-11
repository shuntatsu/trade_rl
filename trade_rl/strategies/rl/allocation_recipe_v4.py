"""Distinct full-horizon settled recipe, independent of a Dataset window."""

from __future__ import annotations

from typing import Any

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile
from trade_rl.strategies.rl.allocation_preprocessing import _native_json
from trade_rl.strategies.rl.allocation_recipe_v2 import allocation_recipe_payload_v2
from trade_rl.strategies.rl.allocation_settled_clock import (
    SettledAllocationClockContract,
)


def allocation_recipe_payload_v4(
    contract: AllocationActionContract,
    feature_names: tuple[str, ...],
    *,
    allocator: AfterCostTargetAllocator,
    expected_horizon_seconds: int,
    runtime_profile: AllocationRuntimeProfile,
    settled_clock: SettledAllocationClockContract,
    observation_schema: AllocationObservationSchema,
) -> dict[str, object]:
    if not isinstance(settled_clock, SettledAllocationClockContract):
        raise ValueError("settled recipe requires its distinct policy/reserve clock")
    if (
        runtime_profile.decision_interval_seconds
        != settled_clock.execution_interval_seconds
        or runtime_profile.economic_horizon_seconds
        != settled_clock.physical_horizon_steps
        * settled_clock.execution_interval_seconds
    ):
        raise ValueError("settled recipe must bind the full physical horizon")
    return allocation_recipe_payload_v2(
        contract,
        feature_names,
        allocator=allocator,
        expected_horizon_seconds=expected_horizon_seconds,
        runtime_profile=runtime_profile,
        observation_schema=observation_schema,
    ) | {
        "schema": "allocation_ppo_recipe_v4",
        "terminal_valuation": "settled",
        "settled_clock": settled_clock.payload(),
    }


def validate_settled_allocation_recipe(recipe: dict[str, Any]) -> None:
    """Reconstruct the complete closed native recipe; no legacy field stripping."""
    try:
        _native_json(recipe)
        if (
            type(recipe) is not dict
            or recipe.get("schema") != "allocation_ppo_recipe_v4"
        ):
            raise ValueError("settled recipe requires the v4 schema")
        action, profile, observation = (
            recipe["action"],
            recipe["runtime_profile"],
            recipe["observation"],
        )
        rebuilt = allocation_recipe_payload_v4(
            AllocationActionContract(action["mode"], action["scale"]),
            tuple(recipe["feature_names"]),
            allocator=AfterCostTargetAllocator(**recipe["allocator"]),
            expected_horizon_seconds=recipe["expected_horizon_seconds"],
            runtime_profile=AllocationRuntimeProfile(
                **{key: value for key, value in profile.items() if key != "schema"}
            ),
            settled_clock=SettledAllocationClockContract.from_payload(
                recipe["settled_clock"]
            ),
            observation_schema=AllocationObservationSchema(
                tuple(observation["feature_names"]),
                observation["max_active_orders"],
                observation["initial_capital"],
                observation["episode_steps"],
            ),
        )
        if canonical_json_bytes(recipe) != canonical_json_bytes(rebuilt):
            raise ValueError("settled recipe differs from its closed reconstruction")
    except (KeyError, TypeError, OverflowError, RecursionError) as error:
        raise ValueError("invalid settled allocation recipe") from error


__all__ = ["allocation_recipe_payload_v4", "validate_settled_allocation_recipe"]
