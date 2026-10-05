"""Closed v4 declarations and repinned pre-loader counterexamples."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from tests.strategies.test_allocation_observation_v3 import recipe
from tests.strategies.test_allocation_preprocessing import declaration
from tests.strategies.test_allocation_protocol_receipt import manifest_v3
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest
from trade_rl.strategies.rl.allocation_receipt_time import parse_source_timestamp


def manifest_v4():
    raw = manifest_v3()
    source = raw["training"]["source"]
    frozen = replace(
        declaration(),
        normalizer=replace(
            declaration().normalizer, source_dataset_id=source["dataset_id"]
        ),
        policy_start_index=source["start_index"],
        policy_start_time_ns=int(
            parse_source_timestamp(source["decision_start"]).astype("int64")
        ),
    )
    raw.update(schema="allocation_ppo_inference_bundle_v4", recipe=recipe(frozen))
    raw["recipe_digest"] = content_digest(raw["recipe"])
    training = raw["training"]
    training["objective"]["deployment_recipe_digest"] = raw["recipe_digest"]
    training["schema"] = "allocation_ppo_training_receipt_v4"
    training["observation_consumption"].update(
        schema="allocation_ppo_observation_consumption_v3",
        observation_schema_digest=content_digest(raw["recipe"]["observation"]),
        preprocessing_digest=frozen.digest,
    )
    training["preprocessing_fit"] = {
        "schema": "allocation_ppo_preprocessing_fit_receipt_v1",
        "declaration_digest": frozen.digest,
        "fit_source_dataset_id": source["dataset_id"],
        "feature_config_digest": "0" * 64,
        "source_normalization_digest": "0" * 64,
        "fit_consumption_digest": "a" * 64,
        "policy_start_index": source["start_index"],
        "policy_start_time_ns": frozen.policy_start_time_ns,
    }
    return raw


def test_v4_closed_source_links_preserve_valid_v3_bytes():
    old = canonical_json_bytes(manifest_v3())
    raw = manifest_v4()
    assert validate_allocation_manifest(raw) == raw
    assert canonical_json_bytes(validate_allocation_manifest(manifest_v3())) == old


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema",), "allocation_ppo_inference_bundle_v3"),
        (("training", "schema"), "allocation_ppo_training_receipt_v3"),
        (("training", "preprocessing_fit", "declaration_digest"), "f" * 64),
        (("training", "preprocessing_fit", "fit_source_dataset_id"), "f" * 64),
        (("training", "preprocessing_fit", "feature_config_digest"), "f" * 64),
        (("training", "preprocessing_fit", "source_normalization_digest"), "f" * 64),
        (("training", "preprocessing_fit", "fit_consumption_digest"), "f" * 64),
        (("training", "preprocessing_fit", "policy_start_index"), 7),
        (("training", "preprocessing_fit", "policy_start_time_ns"), 1),
        (("training", "preprocessing_fit", "extra"), 0),
        (("training", "observation_consumption", "preprocessing_digest"), "f" * 64),
        (
            ("training", "observation_consumption", "schema"),
            "allocation_ppo_observation_consumption_v2",
        ),
        (("training", "observation_consumption", "width"), 72),
        (
            ("recipe", "observation", "raw_feature_projection"),
            "ordinary_guarded_float32_v1",
        ),
        (
            ("recipe", "observation", "feature_preprocessing", "statistics", "mean"),
            [2.0],
        ),
        (("recipe", "observation", "initial_capital"), 1000),
    ],
)
def test_rehashed_bad_v4_crosslinks_reject_before_loader(
    tmp_path, monkeypatch, path, value
):
    from trade_rl.strategies.rl import allocation_artifact as module

    raw = deepcopy(manifest_v4())
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path[0] == "recipe":
        observation = raw["recipe"]["observation"]
        observation["feature_preprocessing_digest"] = content_digest(
            observation["feature_preprocessing"]
        )
        raw["recipe_digest"] = content_digest(raw["recipe"])
        raw["training"]["objective"]["deployment_recipe_digest"] = raw["recipe_digest"]
        raw["training"]["observation_consumption"]["observation_schema_digest"] = (
            content_digest(observation)
        )
    policy = b"must-not-load"
    raw["policy_sha256"] = sha256(policy).hexdigest()
    (tmp_path / "policy.zip").write_bytes(policy)
    (tmp_path / "manifest.json").write_bytes(canonical_json_bytes(raw))
    monkeypatch.setattr(module, "_load_policy", lambda _: pytest.fail("loader reached"))
    with pytest.raises(ValueError):
        module.load_allocation_policy(
            tmp_path,
            expected_digest=content_digest(raw),
            expected_recipe_digest=raw["recipe_digest"],
        )


def test_v4_rejects_json_coercion_before_types_disappear():
    class Alias(str):
        pass

    raw = manifest_v4()
    raw["training"]["preprocessing_fit"]["feature_config_digest"] = Alias("0" * 64)
    with pytest.raises(ValueError, match="native"):
        validate_allocation_manifest(raw)


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_v4_rejects_every_valid_old_recipe_receipt_pair(version):
    from tests.strategies.test_allocation_artifact import manifest
    from tests.strategies.test_allocation_recipe_v2 import manifest_v2

    raw = {"v1": manifest, "v2": manifest_v2, "v3": manifest_v3}[version]()
    raw["schema"] = "allocation_ppo_inference_bundle_v4"
    with pytest.raises(ValueError, match="version pairing"):
        validate_allocation_manifest(raw)


@pytest.mark.parametrize("version", [3, 4])
def test_direct_training_readers_reject_cross_recipe_versions(version):
    from trade_rl.strategies.rl.allocation_preprocessing_receipt import (
        validate_allocation_training_v4,
    )
    from trade_rl.strategies.rl.allocation_protocol_receipt import (
        validate_allocation_training_v3,
    )

    reader = {3: validate_allocation_training_v3, 4: validate_allocation_training_v4}[
        version
    ]
    correct = {3: manifest_v3, 4: manifest_v4}[version]()
    reader(correct["training"], correct["recipe"], correct["recipe_digest"])
    wrong = {3: manifest_v4, 4: manifest_v3}[version]()
    training = wrong["training"]
    training["schema"] = f"allocation_ppo_training_receipt_v{version}"
    if version == 3:
        training.pop("preprocessing_fit")
    with pytest.raises(ValueError, match=f"requires recipe v{version - 1}"):
        reader(training, wrong["recipe"], wrong["recipe_digest"])
