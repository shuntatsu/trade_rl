"""Reconstruct frozen allocation action, observation and runtime recipes."""

from __future__ import annotations

import json
from typing import Any

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_payload,
)
from trade_rl.strategies.rl.allocation_preprocessing import (
    AllocationFeaturePreprocessing,
    _native_json,
)
from trade_rl.strategies.rl.allocation_recipe_v2 import allocation_recipe_payload_v2
from trade_rl.strategies.rl.allocation_recipe_v3 import allocation_recipe_payload_v3


def _mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} must contain exactly its declared fields")
    return value


def validate_allocation_recipe(recipe: dict[str, Any]) -> None:
    if type(recipe) is not dict or type(recipe.get("schema")) is not str:
        raise ValueError("allocation recipe dispatch requires a native mapping and tag")
    v3 = recipe.get("schema") == "allocation_ppo_recipe_v3"
    if v3:
        try:
            _native_json(recipe)
        except RecursionError as error:
            raise ValueError("v3 recipe must contain native finite JSON") from error
    action = _mapping(recipe.get("action"), {"schema", "mode", "scale"}, "action")
    if action["schema"] != "allocation_action_v1":
        raise ValueError("unknown allocation action schema")
    names = recipe.get("feature_names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("recipe requires ordered unique feature names")
    profile = _mapping(
        recipe.get("runtime_profile"),
        {
            "schema",
            "economics_digest",
            "risk_digest",
            "initial_capital",
            "currency",
            "decision_interval_seconds",
            "economic_horizon_seconds",
            "insolvency_valuation",
            "calendar_kind",
            "execution_bar_hours",
        },
        "runtime profile",
    )
    constructor = {
        key: value
        for key, value in profile.items()
        if key not in {"schema", "insolvency_valuation"}
    }
    try:
        arguments = dict(
            allocator=AfterCostTargetAllocator(**recipe["allocator"]),
            expected_horizon_seconds=recipe["expected_horizon_seconds"],
            runtime_profile=AllocationRuntimeProfile(**constructor),
        )
        contract = AllocationActionContract(action["mode"], action["scale"])
        if v3 or recipe.get("schema") == "allocation_ppo_recipe_v2":
            observation = _mapping(
                recipe.get("observation"),
                {
                    "schema",
                    "feature_names",
                    "max_active_orders",
                    "initial_capital",
                    "episode_steps",
                    "fields",
                    "projection",
                    "raw_feature_projection",
                    "order_sort",
                    "order_overflow",
                    "unused_slots",
                }
                | (
                    {"feature_preprocessing", "feature_preprocessing_digest"}
                    if v3
                    else set()
                ),
                "allocation observation",
            )
            schema = AllocationObservationSchema(
                tuple(observation["feature_names"]),
                observation["max_active_orders"],
                observation["initial_capital"],
                observation["episode_steps"],
            )
            if v3:
                reconstructed = allocation_recipe_payload_v3(
                    contract,
                    tuple(names),
                    observation_schema=schema,
                    feature_preprocessing=AllocationFeaturePreprocessing.from_payload(
                        observation["feature_preprocessing"]
                    ),
                    **arguments,
                )
            else:
                reconstructed = allocation_recipe_payload_v2(
                    contract, tuple(names), observation_schema=schema, **arguments
                )
        else:
            reconstructed = allocation_recipe_payload(
                contract, tuple(names), **arguments
            )
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("allocation recipe is malformed") from error
    differs = (
        canonical_json_bytes(reconstructed) != canonical_json_bytes(recipe)
        if v3
        else json.loads(canonical_json_bytes(reconstructed)) != recipe
    )
    if differs:
        raise ValueError("allocation recipe differs from the supported contract")
