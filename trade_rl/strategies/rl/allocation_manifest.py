"""Canonical allocation bundle manifest; receipts grant no authorization."""

from __future__ import annotations

import json
from typing import Any

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.strategies.rl.allocation_preprocessing import _native_json
from trade_rl.strategies.rl.allocation_preprocessing_receipt import (
    validate_allocation_training_v4,
)
from trade_rl.strategies.rl.allocation_protocol_receipt import (
    validate_allocation_training_v3,
)
from trade_rl.strategies.rl.allocation_recipe_validation import (
    validate_allocation_recipe,
)
from trade_rl.strategies.rl.allocation_training_receipt import (
    validate_allocation_training,
)

ALLOCATION_BUNDLE_SCHEMA = "allocation_ppo_inference_bundle_v1"
ALLOCATION_BUNDLE_SCHEMA_V2 = "allocation_ppo_inference_bundle_v2"
ALLOCATION_BUNDLE_SCHEMA_V3 = "allocation_ppo_inference_bundle_v3"
ALLOCATION_BUNDLE_SCHEMA_V4 = "allocation_ppo_inference_bundle_v4"


def allocation_bundle_schema(recipe: dict[str, Any]) -> str:
    if recipe.get("schema") == "allocation_ppo_recipe_v1":
        return ALLOCATION_BUNDLE_SCHEMA
    if recipe.get("schema") == "allocation_ppo_recipe_v2":
        return ALLOCATION_BUNDLE_SCHEMA_V2
    if recipe.get("schema") == "allocation_ppo_recipe_v3":
        return ALLOCATION_BUNDLE_SCHEMA_V4
    raise ValueError("unsupported allocation recipe version")


def _mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} must contain exactly its declared fields")
    return value


def validate_allocation_manifest(
    value: object, *, require_policy: bool = False
) -> dict[str, Any]:
    """Return detached canonical data; receipts do not prove source authenticity."""
    if type(value) is not dict or type(value.get("schema")) is not str:
        raise ValueError("allocation manifest dispatch requires native mapping/tag")
    try:
        if value["schema"] == ALLOCATION_BUNDLE_SCHEMA_V4:
            _native_json(value)
        normalized = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError(
            "allocation manifest must contain canonical native finite JSON data"
        ) from error
    keys = {"schema", "recipe", "recipe_digest", "training"}
    if require_policy or isinstance(normalized, dict) and "policy_sha256" in normalized:
        keys.add("policy_sha256")
    manifest = _mapping(normalized, keys, "allocation manifest")
    if not isinstance(manifest["recipe"], dict):
        raise ValueError("allocation recipe must be a mapping")
    validate_allocation_recipe(manifest["recipe"])
    explicit = manifest["schema"] in (
        ALLOCATION_BUNDLE_SCHEMA_V3,
        ALLOCATION_BUNDLE_SCHEMA_V4,
    )
    expected_recipe = (
        "allocation_ppo_recipe_v3"
        if manifest["schema"] == ALLOCATION_BUNDLE_SCHEMA_V4
        else "allocation_ppo_recipe_v2"
    )
    if (
        explicit
        and manifest["recipe"]["schema"] != expected_recipe
        or not explicit
        and manifest["schema"] != allocation_bundle_schema(manifest["recipe"])
    ):
        raise ValueError("unsupported allocation inference/recipe version pairing")
    require_sha256(manifest["recipe_digest"], field="recipe_digest")
    if content_digest(manifest["recipe"]) != manifest["recipe_digest"]:
        raise ValueError("allocation recipe digest differs from its payload")
    validator = (
        validate_allocation_training_v4
        if manifest["schema"] == ALLOCATION_BUNDLE_SCHEMA_V4
        else validate_allocation_training_v3
        if explicit
        else validate_allocation_training
    )
    validator(manifest["training"], manifest["recipe"], manifest["recipe_digest"])
    if "policy_sha256" in manifest:
        require_sha256(manifest["policy_sha256"], field="policy_sha256")
    return manifest
