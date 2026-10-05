"""Closed frozen-feature recipe; existing runtime/bundle lanes stay unchanged."""

from __future__ import annotations

from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_observation_v3 import (
    allocation_observation_payload_v3,
)
from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile
from trade_rl.strategies.rl.allocation_preprocessing import (
    AllocationFeaturePreprocessing,
)
from trade_rl.strategies.rl.allocation_recipe_v2 import allocation_recipe_payload_v2


def allocation_recipe_payload_v3(
    contract: AllocationActionContract,
    feature_names: tuple[str, ...],
    *,
    allocator: AfterCostTargetAllocator,
    expected_horizon_seconds: int,
    runtime_profile: AllocationRuntimeProfile,
    observation_schema: AllocationObservationSchema,
    feature_preprocessing: AllocationFeaturePreprocessing,
) -> dict[str, object]:
    legacy = allocation_recipe_payload_v2(
        contract,
        feature_names,
        allocator=allocator,
        expected_horizon_seconds=expected_horizon_seconds,
        runtime_profile=runtime_profile,
        observation_schema=observation_schema,
    )
    return legacy | {
        "schema": "allocation_ppo_recipe_v3",
        "observation": allocation_observation_payload_v3(
            observation_schema, feature_preprocessing
        ),
        "feature_preprocessing": "frozen_causal_prefix_v1",
    }


__all__ = ["allocation_recipe_payload_v3"]
