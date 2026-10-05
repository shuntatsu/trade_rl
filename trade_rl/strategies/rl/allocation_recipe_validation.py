"""Reconstruct frozen allocation action, observation and runtime recipes."""

from __future__ import annotations

import json
from typing import Any

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_payload,
)


def _mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} must contain exactly its declared fields")
    return value


def validate_allocation_recipe(recipe: dict[str, Any]) -> None:
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
        reconstructed = allocation_recipe_payload(
            AllocationActionContract(action["mode"], action["scale"]),
            tuple(names),
            allocator=AfterCostTargetAllocator(**recipe["allocator"]),
            expected_horizon_seconds=recipe["expected_horizon_seconds"],
            runtime_profile=AllocationRuntimeProfile(**constructor),
        )
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("allocation recipe is malformed") from error
    if json.loads(canonical_json_bytes(reconstructed)) != recipe:
        raise ValueError("allocation recipe differs from the supported contract")
