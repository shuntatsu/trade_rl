"""Pure identity split between allocation candidate structure and scenario runtime."""

from __future__ import annotations

from typing import Any

from trade_rl._validation import require_sha256
from trade_rl.artifacts import content_digest


def _recipe(value: object) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(value) is not dict:
        raise ValueError("allocation recipe must be a native mapping")
    recipe: dict[str, Any] = value
    if not isinstance(recipe.get("schema"), str) or not recipe["schema"]:
        raise ValueError("allocation recipe requires a schema")
    profile = recipe.get("runtime_profile")
    if type(profile) is not dict:
        raise ValueError("allocation recipe requires a runtime_profile mapping")
    for field in ("economics_digest", "risk_digest"):
        digest = profile.get(field)
        if not isinstance(digest, str):
            raise ValueError(f"runtime_profile.{field} must be a SHA-256 digest")
        require_sha256(digest, field=f"runtime_profile.{field}")
    return recipe, profile


def allocation_candidate_recipe_digest(recipe: object) -> str:
    """Identity of decision/policy structure, excluding scenario runtime economics."""
    value, _ = _recipe(recipe)
    candidate = {key: item for key, item in value.items() if key != "runtime_profile"}
    return content_digest(
        {
            "schema": "allocation_candidate_recipe_projection_v1",
            "recipe_schema": value["schema"],
            "candidate_recipe": candidate,
        }
    )


def allocation_runtime_invariant_digest(recipe: object) -> str:
    """Runtime identity that stress scenarios are not allowed to change."""
    _, profile = _recipe(recipe)
    invariant = {
        key: item
        for key, item in profile.items()
        if key not in {"economics_digest", "risk_digest"}
    }
    return content_digest(
        {
            "schema": "allocation_runtime_invariant_projection_v1",
            "runtime_profile": invariant,
        }
    )


def _declared_runtime_digests(recipe: object) -> tuple[str, str]:
    _, profile = _recipe(recipe)
    return str(profile["economics_digest"]), str(profile["risk_digest"])


def validate_allocation_scenario_recipe(
    base_recipe: object,
    scenario_recipe: object,
    *,
    base_economics_digest: str,
    base_risk_digest: str,
    scenario_economics_digest: str,
    scenario_risk_digest: str,
) -> None:
    """Permit declared economics/risk stress only; reject candidate/runtime drift."""
    for field, value in (
        ("base_economics_digest", base_economics_digest),
        ("base_risk_digest", base_risk_digest),
        ("scenario_economics_digest", scenario_economics_digest),
        ("scenario_risk_digest", scenario_risk_digest),
    ):
        require_sha256(value, field=field)

    base_economics, base_risk = _declared_runtime_digests(base_recipe)
    scenario_economics, scenario_risk = _declared_runtime_digests(scenario_recipe)
    if base_economics != base_economics_digest:
        raise ValueError("base allocation recipe economics differ from declaration")
    if base_risk != base_risk_digest:
        raise ValueError("base allocation recipe risk differs from declaration")
    if scenario_economics != scenario_economics_digest:
        raise ValueError("scenario allocation recipe economics differ from declaration")
    if scenario_risk != scenario_risk_digest:
        raise ValueError("scenario allocation recipe risk differs from declaration")

    if allocation_candidate_recipe_digest(
        base_recipe
    ) != allocation_candidate_recipe_digest(scenario_recipe):
        raise ValueError("scenario changed allocation candidate structure")
    if allocation_runtime_invariant_digest(
        base_recipe
    ) != allocation_runtime_invariant_digest(scenario_recipe):
        raise ValueError("scenario changed allocation runtime invariant")


__all__ = [
    "allocation_candidate_recipe_digest",
    "allocation_runtime_invariant_digest",
    "validate_allocation_scenario_recipe",
]
