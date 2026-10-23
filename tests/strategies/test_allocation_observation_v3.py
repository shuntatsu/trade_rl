"""Literal transformed features with unchanged raw decisions and economic bytes."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from tests.strategies.test_allocation_observation_encoder_v2 import (
    declared_decision,
    declared_order,
    declared_snapshot,
    schema,
)
from tests.strategies.test_allocation_preprocessing import declaration
from tests.strategies.test_allocation_recipe_v2 import manifest_v2, recipe_v2
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest
from trade_rl.strategies.rl.allocation_observation_encoder_v2 import (
    encode_allocation_observation_v2,
)
from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile
from trade_rl.strategies.rl.allocation_recipe_validation import (
    validate_allocation_recipe,
)

OBS = "trade_rl.strategies.rl.allocation_observation_v3"
RECIPE = "trade_rl.strategies.rl.allocation_recipe_v3"


def capability(module):
    try:
        return import_module(module)
    except ModuleNotFoundError as error:
        if error.name != module:
            raise
        pytest.fail("pure allocation observation/recipe v3 is missing")


def observation(preprocessing=None, contract=None):
    return capability(OBS).allocation_observation_payload_v3(
        contract or schema(feature_names=("signal",)), preprocessing or declaration()
    )


def recipe(preprocessing=None, contract=None):
    profile = recipe_v2()["runtime_profile"]
    return capability(RECIPE).allocation_recipe_payload_v3(
        AllocationActionContract("direct", 0.5),
        ("signal",),
        allocator=AfterCostTargetAllocator(),
        expected_horizon_seconds=3600,
        runtime_profile=AllocationRuntimeProfile(
            **{k: v for k, v in profile.items() if k != "schema"}
        ),
        observation_schema=contract
        or schema(feature_names=("signal",), initial_capital=1000),
        feature_preprocessing=preprocessing or declaration(),
    )


def encode(snapshot=None, decision=None, preprocessing=None, **changes):
    snapshot = declared_snapshot() if snapshot is None else snapshot
    decision = decision or declared_decision(
        snapshot, feature_names=("signal",), feature_values=(4.0,)
    )
    args = dict(
        schema=schema(feature_names=("signal",)),
        preprocessing=preprocessing or declaration(),
        feature_config_digest="0" * 64,
        source_normalization_digest="0" * 64,
        episode_steps=16,
    )
    return capability(OBS).encode_allocation_observation_v3(
        snapshot, decision, **(args | changes)
    )


def test_literal_observation_payload_and_digest_are_detached():
    frozen, contract = declaration(), schema(feature_names=("signal",))
    expected = contract.payload() | {
        "schema": "allocation_account_observation_v3",
        "raw_feature_projection": "frozen_prefix_standardization_float64_then_float32_v1",
        "feature_preprocessing": frozen.payload(),
        "feature_preprocessing_digest": frozen.digest,
    }
    payload = observation(frozen, contract)
    assert payload == expected and len(payload["fields"]) == 1 + 22 + 24 * 2
    raw = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    assert (
        hashlib.sha256(canonical_json_bytes(payload)).digest()
        == hashlib.sha256(raw).digest()
    )
    payload["feature_preprocessing"]["statistics"]["mean"][0] = 9
    payload["fields"].pop()
    assert observation(frozen, contract) == expected


@pytest.mark.parametrize(
    "orders", [(), (declared_order(),), (declared_order(-1), declared_order())]
)
def test_literal_future_four_becomes_three_and_entire_economic_tail_is_identical(
    orders,
):
    snapshot = declared_snapshot(orders)
    decision = declared_decision(
        snapshot, feature_names=("signal",), feature_values=(4.0,)
    )
    before = (
        snapshot.canonical_bytes(),
        decision.decision_digest,
        decision.baseline.decision_digest,
    )
    base = encode_allocation_observation_v2(
        snapshot, decision, schema=schema(feature_names=("signal",)), episode_steps=16
    )
    result = encode(snapshot, decision)
    assert result.dtype == np.float32 and result.shape == (71,)
    assert base[0] == 4 and result[0] == 3
    assert result[1:].tobytes() == base[1:].tobytes()
    assert decision.feature_values == (4.0,)
    assert before == (
        snapshot.canonical_bytes(),
        decision.decision_digest,
        decision.baseline.decision_digest,
    )
    result[:] = 0
    assert encode(snapshot, decision)[0] == 3


def test_feature_zero_and_tiny_residual_cast_are_not_terminal_sentinel_logic():
    assert (
        encode(
            decision=declared_decision(feature_names=("signal",), feature_values=(0.0,))
        )[0]
        == -1
    )
    tiny = replace(declaration().normalizer, mean=(-1e-50,))
    assert (
        encode(
            decision=declared_decision(
                feature_names=("signal",), feature_values=(0.0,)
            ),
            preprocessing=declaration(normalizer=tiny),
        )[0]
        == 0
    )
    with pytest.raises(ValueError, match="declared snapshot"):
        encode(snapshot=np.zeros(71), decision=declared_decision())


@pytest.mark.parametrize(
    "changes",
    [
        dict(feature_config_digest="1" * 64),
        dict(source_normalization_digest="1" * 64),
        dict(episode_steps=15),
        dict(schema=schema()),
        dict(preprocessing=declaration(fit_as_of_ns=2**62, policy_start_time_ns=2**62)),
    ],
)
def test_encoder_requires_declared_features_build_clock_and_economic_bindings(changes):
    with pytest.raises(ValueError):
        encode(**changes)


def test_future_dataset_identity_and_local_index_are_not_fit_provenance_bindings():
    book = dict(declared_snapshot().book_facts)
    book.update(as_of_dataset_id="f" * 64, as_of_index=7)
    snapshot = declared_snapshot(
        (), dataset_id="f" * 64, decision_index=7, book_facts=book
    )
    decision = declared_decision(
        snapshot, feature_names=("signal",), feature_values=(4.0,)
    )
    assert encode(snapshot, decision)[0] == 3


def test_multiple_features_transform_in_declared_order_without_double_counting_width():
    statistics = replace(
        declaration().normalizer,
        feature_indices=(0, 1),
        feature_names=("alpha", "beta"),
        mean=(1.0, 10.0),
        scale=(1.0, 2.0),
        usable_counts=((2, 2),),
    )
    frozen = declaration(normalizer=statistics)
    decision = declared_decision(feature_values=(4.0, 14.0))
    result = encode(decision=decision, preprocessing=frozen, schema=schema())
    base = encode_allocation_observation_v2(
        declared_snapshot(), decision, schema=schema(), episode_steps=16
    )
    assert result.shape == (72,) and result[:2].tolist() == [3.0, 2.0]
    assert result[2:].tobytes() == base[2:].tobytes()


def test_distinct_recipe_reconstructs_without_changing_any_old_recipe_bytes():
    old = recipe_v2()
    before = canonical_json_bytes(old), canonical_json_bytes(manifest_v2())
    current = recipe()
    expected = old | {
        "schema": "allocation_ppo_recipe_v3",
        "observation": observation(
            contract=schema(feature_names=("signal",), initial_capital=1000)
        ),
        "feature_preprocessing": "frozen_causal_prefix_v1",
    }
    assert current == expected
    validate_allocation_recipe(current)
    assert current["terminal_observation"] == "zero_sentinel_on_true_termination_v1"
    assert current["bootstrap"] == "nonterminal_rollout_only"
    assert before == (
        canonical_json_bytes(recipe_v2()),
        canonical_json_bytes(manifest_v2()),
    )


@pytest.mark.parametrize(
    "changes",
    [{"feature_names": ("other",)}, {"initial_capital": 999}, {"episode_steps": 15}],
)
def test_v3_recipe_preserves_schema_capital_horizon_and_feature_guards(changes):
    with pytest.raises(ValueError):
        recipe(
            contract=replace(
                schema(feature_names=("signal",), initial_capital=1000), **changes
            )
        )


@pytest.mark.parametrize(
    "path,value",
    [
        (("observation", "schema"), "allocation_account_observation_v2"),
        (("observation", "fields"), ["feature:signal"]),
        (("observation", "projection"), "nearest"),
        (("observation", "raw_feature_projection"), "ordinary_guarded_float32_v1"),
        (("observation", "feature_preprocessing_digest"), "f" * 64),
        (("observation", "max_active_orders"), True),
        (("observation", "initial_capital"), 1000),
        (("observation", "extra"), 0),
        (("observation", "feature_preprocessing", "scale_fallback"), True),
        (
            ("observation", "feature_preprocessing", "statistics", "feature_names"),
            [Path("signal")],
        ),
        (("feature_preprocessing",), "raw_available_values_v1"),
        (("terminal_observation",), "normalized_zero"),
        (("schema",), "allocation_ppo_recipe_v2"),
    ],
)
def test_closed_v3_recipe_rejects_aliases_and_fixed_or_variable_crosslinks(path, value):
    payload = deepcopy(recipe())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_allocation_recipe(payload)


@pytest.mark.parametrize(
    "bundle",
    [
        "allocation_ppo_inference_bundle_v1",
        "allocation_ppo_inference_bundle_v2",
        "allocation_ppo_inference_bundle_v3",
        "allocation_ppo_inference_bundle_v4",
    ],
)
def test_old_receipts_cannot_make_a_v3_recipe_bundle_valid(bundle):
    raw = manifest_v2()
    raw.update(schema=bundle, recipe=recipe())
    raw["recipe_digest"] = content_digest(raw["recipe"])
    raw["training"]["objective"]["deployment_recipe_digest"] = raw["recipe_digest"]
    error = (
        "v4 training requires" if bundle.endswith("v4") else "unsupported allocation"
    )
    with pytest.raises(ValueError, match=error):
        validate_allocation_manifest(raw)


def test_reader_rejects_non_native_json_mapping_and_cycle():
    payload = recipe()
    payload["cycle"] = payload
    with pytest.raises(ValueError):
        validate_allocation_recipe(payload)

    class MappingSubclass(dict):
        pass

    with pytest.raises(ValueError):
        validate_allocation_recipe(MappingSubclass(recipe()))


def test_frozen_coefficients_change_observation_and_recipe_identity():
    value = declaration(normalizer=replace(declaration().normalizer, mean=(2.0,)))
    assert content_digest(observation(value)) != content_digest(observation())
    assert content_digest(recipe(value)) != content_digest(recipe())


@pytest.mark.parametrize("kind", ["schema", "mapping"])
def test_non_native_dispatch_cannot_hide_a_v3_tag_in_the_legacy_lane(kind):
    class SchemaAlias(str):
        def __eq__(self, other):
            return other == "allocation_ppo_recipe_v2"

        __hash__ = str.__hash__

    class MappingAlias(dict):
        def get(self, key, default=None):
            return (
                "allocation_ppo_recipe_v2"
                if key == "schema"
                else super().get(key, default)
            )

        def __ne__(self, other):
            return False

    payload = recipe_v2()
    payload["schema"] = (
        SchemaAlias("allocation_ppo_recipe_v3")
        if kind == "schema"
        else "allocation_ppo_recipe_v3"
    )
    if kind == "mapping":
        payload = MappingAlias(payload)
    with pytest.raises(ValueError):
        validate_allocation_recipe(payload)
