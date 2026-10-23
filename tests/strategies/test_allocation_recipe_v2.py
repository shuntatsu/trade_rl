"""Versioned recipe and pre-deserialization contract, never source authority."""

from copy import deepcopy
from dataclasses import replace
from importlib import import_module

import gymnasium as gym
import numpy as np
import pytest

from tests.strategies.test_allocation_artifact import Policy, manifest
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest
from trade_rl.strategies.rl.allocation_model import AllocationPPOPolicy
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_payload,
)


def recipe_v2(*, schema=None):
    try:
        module = import_module("trade_rl.strategies.rl.allocation_recipe_v2")
    except ModuleNotFoundError as error:
        if error.name == "trade_rl.strategies.rl.allocation_recipe_v2":
            pytest.fail("the distinct v2 recipe capability is missing")
        raise
    profile = manifest()["recipe"]["runtime_profile"]
    return module.allocation_recipe_payload_v2(
        AllocationActionContract("direct", 0.5),
        ("signal",),
        allocator=AfterCostTargetAllocator(),
        expected_horizon_seconds=3600,
        runtime_profile=AllocationRuntimeProfile(
            **{k: v for k, v in profile.items() if k != "schema"}
        ),
        observation_schema=schema
        or AllocationObservationSchema(("signal",), 2, 1000, 16),
    )


def manifest_v2():
    raw = manifest()
    raw["schema"] = "allocation_ppo_inference_bundle_v2"
    raw["recipe"] = recipe_v2()
    raw["training"]["observation_consumption"] = {
        "schema": "allocation_ppo_observation_consumption_v2",
        "observation_schema_digest": content_digest(raw["recipe"]["observation"]),
        "width": 71,
        "dtype": "little_endian_float32",
        "actor_count": 64,
        "rollout_boundary_count": 2,
        "terminal_count": 4,
        "actor_digest": "1" * 64,
        "rollout_boundary_digest": "2" * 64,
        "terminal_digest": "3" * 64,
    }
    repin(raw)
    return raw


def repin(raw):
    raw["recipe_digest"] = content_digest(raw["recipe"])
    raw["training"]["objective"]["deployment_recipe_digest"] = raw["recipe_digest"]


def test_distinct_recipe_binds_schema_and_keeps_legacy_recipe_bytes():
    before = canonical_json_bytes(manifest())
    recipe = recipe_v2()
    assert recipe["schema"] == "allocation_ppo_recipe_v2"
    assert (
        recipe["observation"]
        == AllocationObservationSchema(("signal",), 2, 1000, 16).payload()
    )
    assert recipe["terminal_observation"] == "zero_sentinel_on_true_termination_v1"
    assert recipe["truncation"] == "not_generated"
    assert recipe["bootstrap"] == "nonterminal_rollout_only"
    assert validate_allocation_manifest(manifest_v2())["recipe"] == recipe
    assert canonical_json_bytes(manifest()) == before
    profile = manifest()["recipe"]["runtime_profile"]
    actual_legacy = allocation_recipe_payload(
        AllocationActionContract("direct", 0.5),
        ("signal",),
        allocator=AfterCostTargetAllocator(),
        expected_horizon_seconds=3600,
        runtime_profile=AllocationRuntimeProfile(
            **{k: v for k, v in profile.items() if k != "schema"}
        ),
    )
    assert canonical_json_bytes(actual_legacy) == canonical_json_bytes(
        manifest()["recipe"]
    )


@pytest.mark.parametrize(
    "change",
    [{"feature_names": ("other",)}, {"initial_capital": 999}, {"episode_steps": 15}],
)
def test_schema_features_capital_and_horizon_must_match_runtime(change):
    schema = AllocationObservationSchema(("signal",), 2, 1000, 16)
    with pytest.raises(ValueError, match="match"):
        recipe_v2(schema=replace(schema, **change))


@pytest.mark.parametrize(
    "path,value",
    [
        (("observation", "fields"), ["feature:signal"]),
        (("observation", "feature_names"), ["other"]),
        (("observation", "initial_capital"), 999),
        (("observation", "episode_steps"), 15),
        (("observation", "projection"), "nearest"),
        (("observation", "max_active_orders"), True),
        (("observation", "extra"), 0),
        (("terminal_observation",), "live_snapshot"),
        (("truncation",), "time_limit"),
        (("bootstrap",), "terminal_value"),
    ],
)
def test_rehashed_unsupported_v2_recipe_rejected(path, value):
    raw = manifest_v2()
    target = raw["recipe"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    repin(raw)
    with pytest.raises(ValueError):
        validate_allocation_manifest(raw)


def test_versions_cannot_be_crossed_and_feature_width_is_not_counted_twice():
    raw = manifest_v2()
    for outer in ("allocation_ppo_inference_bundle_v1", "unknown"):
        with pytest.raises(ValueError):
            validate_allocation_manifest(raw | {"schema": outer})
    model = Policy()
    model.observation_space = gym.spaces.Box(-np.inf, np.inf, (71,), np.float32)
    assert AllocationPPOPolicy(model, raw).manifest == raw
    model.observation_space = gym.spaces.Box(-np.inf, np.inf, (72,), np.float32)
    with pytest.raises(ValueError, match="spaces"):
        AllocationPPOPolicy(model, raw)


@pytest.mark.parametrize(
    "corruption", ["layout", "empty_terminal_phase", "reset_count"]
)
def test_layout_tamper_rejected_before_optional_loader_even_with_new_content_pin(
    tmp_path, monkeypatch, corruption
):
    from trade_rl.strategies.rl import allocation_artifact as module

    raw = deepcopy(manifest_v2())
    if corruption == "layout":
        raw["recipe"]["observation"]["fields"].append("feature:signal")
    else:
        raw["training"]["observation_consumption"]["terminal_count"] = 0
        if corruption == "reset_count":
            import hashlib

            raw["training"]["observation_consumption"]["terminal_digest"] = (
                hashlib.sha256(b"").hexdigest()
            )
    repin(raw)
    data = b"never-deserialized"
    import hashlib

    raw["policy_sha256"] = hashlib.sha256(data).hexdigest()
    root = tmp_path / "bad"
    root.mkdir()
    (root / "policy.zip").write_bytes(data)
    (root / "manifest.json").write_bytes(canonical_json_bytes(raw))
    monkeypatch.setattr(module, "_load_policy", lambda _: pytest.fail("loader called"))
    with pytest.raises(ValueError):
        module.load_allocation_policy(
            root,
            expected_digest=content_digest(raw),
            expected_recipe_digest=raw["recipe_digest"],
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 72),
        ("actor_count", 63),
        ("rollout_boundary_count", 3),
        ("terminal_count", 65),
        ("dtype", "float64"),
        ("observation_schema_digest", "f" * 64),
        ("actor_digest", "bad"),
    ],
)
def test_input_receipt_is_bound_to_actual_budget_schema_and_width(field, value):
    raw = manifest_v2()
    raw["training"]["observation_consumption"][field] = value
    with pytest.raises(ValueError):
        validate_allocation_manifest(raw)
