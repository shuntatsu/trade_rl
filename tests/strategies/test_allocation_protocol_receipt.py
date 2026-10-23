"""Independent v3 receipt and pre-loader failures; no learner or market fit."""

from copy import deepcopy
from hashlib import sha256

import pytest

from tests.strategies.test_allocation_recipe_v2 import manifest_v2
from tests.strategies.test_allocation_training_protocol import DECLARATION
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest
from trade_rl.strategies.rl.allocation_training_protocol import (
    AllocationPPOTrainingProtocol,
)


def protocol(**changes):
    return AllocationPPOTrainingProtocol(**(DECLARATION | changes))


def manifest_v3():
    raw = manifest_v2()
    declaration = protocol(n_steps=32, batch_size=32)
    raw["schema"] = "allocation_ppo_inference_bundle_v3"
    raw["training"].update(
        schema="allocation_ppo_training_receipt_v3",
        protocol=declaration.payload(),
        protocol_digest=declaration.digest,
        optimization={
            "schema": "allocation_ppo_optimization_receipt_v1",
            "rollout_update_count": 2,
            "successful_optimizer_step_calls": 20,
            "epoch_iterations": 20,
            "optimizer_step_events_digest": "b" * 64,
            "final_capture": "after_final_train_v1",
        },
    )
    raw["training"]["ppo"]["net_arch"] = {"pi": [32, 32], "vf": [32, 32]}
    return raw


def test_v3_extends_training_only_and_preserves_v2_recipe_identity():
    old = manifest_v2()
    before = canonical_json_bytes(old)
    raw = manifest_v3()
    assert validate_allocation_manifest(raw) == raw
    assert raw["recipe"] == old["recipe"]
    assert raw["recipe_digest"] == old["recipe_digest"]
    assert canonical_json_bytes(old) == before
    with pytest.raises(ValueError):
        validate_allocation_manifest(raw | {"schema": old["schema"]})
    with pytest.raises(ValueError):
        validate_allocation_manifest(old | {"schema": raw["schema"]})


@pytest.mark.parametrize(
    "path,value",
    [
        (("protocol_digest",), "f" * 64),
        (("protocol", "ppo", "n_steps"), 16),
        (("protocol", "ppo", "gae_lambda"), 0.5),
        (("protocol", "optimizer", "eps"), 0),
        (("optimization", "successful_optimizer_step_calls"), 19),
        (("optimization", "successful_optimizer_step_calls"), True),
        (("optimization", "epoch_iterations"), 19),
        (("optimization", "rollout_update_count"), 1),
        (("optimization", "final_capture"), "before_train"),
        (("optimization", "extra"), 0),
    ],
)
def test_rehashed_protocol_or_update_tamper_fails_before_loader(
    tmp_path, monkeypatch, path, value
):
    from trade_rl.strategies.rl import allocation_artifact as module

    raw = deepcopy(manifest_v3())
    target = raw["training"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path[0] == "protocol":
        raw["training"]["protocol_digest"] = content_digest(raw["training"]["protocol"])
    data = b"must-not-deserialize"
    raw["policy_sha256"] = sha256(data).hexdigest()
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "policy.zip").write_bytes(data)
    (root / "manifest.json").write_bytes(canonical_json_bytes(raw))
    monkeypatch.setattr(module, "_load_policy", lambda _: pytest.fail("loader reached"))
    with pytest.raises(ValueError):
        module.load_allocation_policy(
            root,
            expected_digest=content_digest(raw),
            expected_recipe_digest=raw["recipe_digest"],
        )


def test_kl_can_stop_before_first_step_but_empty_digest_and_epoch_bounds_are_required():
    raw = manifest_v3()
    declaration = protocol(n_steps=32, batch_size=32, target_kl=0.1)
    raw["training"].update(
        protocol=declaration.payload(), protocol_digest=declaration.digest
    )
    update = raw["training"]["optimization"]
    update.update(
        successful_optimizer_step_calls=0,
        epoch_iterations=2,
        optimizer_step_events_digest=sha256(b"").hexdigest(),
    )
    assert validate_allocation_manifest(raw) == raw
    for key, value in (
        ("epoch_iterations", 1),
        ("optimizer_step_events_digest", "a" * 64),
    ):
        bad = deepcopy(raw)
        bad["training"]["optimization"][key] = value
        with pytest.raises(ValueError):
            validate_allocation_manifest(bad)
