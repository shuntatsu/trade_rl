"""Actual collector-input digest/count consistency; not source authenticity."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from trade_rl._validation import require_sha256
from trade_rl.artifacts import content_digest


def validate_allocation_input_receipt(
    value: object,
    recipe: dict[str, Any],
    *,
    actual_steps: int,
    rollout_steps: int,
    episode_starts: int,
    horizon_terminations: int,
) -> None:
    v3 = recipe["schema"] == "allocation_ppo_recipe_v3"
    keys = {
        "schema",
        "observation_schema_digest",
        "width",
        "dtype",
        "actor_count",
        "rollout_boundary_count",
        "terminal_count",
        "actor_digest",
        "rollout_boundary_digest",
        "terminal_digest",
    }
    if v3:
        keys.add("preprocessing_digest")
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("v2 input receipt must contain exactly its declared fields")
    if (
        v3
        and value["preprocessing_digest"]
        != recipe["observation"]["feature_preprocessing_digest"]
    ):
        raise ValueError("actual input preprocessing differs from recipe")
    for key in ("width", "actor_count", "rollout_boundary_count", "terminal_count"):
        count = value[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("v2 input counts/width must be nonnegative integers")
    for key in (
        "observation_schema_digest",
        "actor_digest",
        "rollout_boundary_digest",
        "terminal_digest",
    ):
        require_sha256(value[key], field=key)
    for phase in ("actor", "rollout_boundary", "terminal"):
        if (
            value[f"{phase}_count"] == 0
            and value[f"{phase}_digest"] != sha256(b"").hexdigest()
        ):
            raise ValueError("an empty input phase must have the empty SHA256 digest")
    if (
        value["schema"]
        != (
            "allocation_ppo_observation_consumption_v3"
            if v3
            else "allocation_ppo_observation_consumption_v2"
        )
        or value["dtype"] != "little_endian_float32"
        or value["width"] != len(recipe["observation"]["fields"])
        or value["observation_schema_digest"] != content_digest(recipe["observation"])
        or value["actor_count"] != actual_steps
        or value["rollout_boundary_count"] != actual_steps // rollout_steps
        or value["terminal_count"] > actual_steps
        or not max(episode_starts - 1, horizon_terminations)
        <= value["terminal_count"]
        <= episode_starts
    ):
        raise ValueError("v2 input receipt differs from its schema or realized budget")
