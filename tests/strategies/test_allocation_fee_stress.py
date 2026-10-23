"""Closed fee-view declarations retain the ordinary full-recipe boundary."""

from copy import deepcopy
from importlib import import_module

import numpy as np
import pytest

from tests.evaluation.allocation_fee_stress_fixture import native_pair, publish_fake
from tests.evaluation.test_allocation_fee_stress import declaration
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.strategies.rl.allocation_artifact import load_allocation_policy


def view_case(tmp_path, monkeypatch):
    base, stress = native_pair()
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    import trade_rl.strategies.rl.allocation_artifact as archive

    monkeypatch.setattr(archive, "_load_policy", lambda _: model)
    policy = load_allocation_policy(
        root, expected_digest=digest, expected_recipe_digest=base.recipe_digest
    )
    module = import_module("trade_rl.strategies.rl.allocation_fee_stress")
    _, contract, _ = declaration(base, stress, root, digest)
    raw = {
        "schema": "allocation_fee_stress_binding_v1",
        "base_recipe": base.recipe,
        "stress_recipe": stress.recipe,
        "base_cost_payload": base.execution_cost.execution_policy_payload(),
        "stress_cost_payload": stress.execution_cost.execution_policy_payload(),
        "fee_factor": 2.0,
        "original_policy_digest": digest,
        "original_policy_sha256": policy.manifest["policy_sha256"],
        "model_state_digest": module.allocation_policy_state_digest(policy),
        "contract_digest": contract.digest,
        "scenario_digest": contract.scenarios[1].digest,
    }
    return module, policy, raw, stress


def test_dedicated_view_detaches_declaration_and_keeps_ordinary_guard(
    tmp_path, monkeypatch
):
    module, policy, raw, stress = view_case(tmp_path, monkeypatch)
    original = canonical_json_bytes(policy.manifest)
    binding = module.AllocationFeeStressBinding(raw)
    view = module.AllocationFeeStressPolicyView(policy, binding)
    raw["fee_factor"] = 7
    exposed = binding.payload()
    exposed["stress_recipe"]["feature_names"][0] = "changed"
    assert binding.payload()["fee_factor"] == 2
    with pytest.raises(ValueError, match="runtime recipe"):
        policy.action(
            np.zeros(policy.model.observation_space.shape),
            runtime_recipe_digest=stress.recipe_digest,
        )
    with pytest.raises(ValueError, match="recipe"):
        view.action(
            np.zeros(policy.model.observation_space.shape),
            runtime_recipe_digest=policy.manifest["recipe_digest"],
        )
    assert (
        view.action(
            np.zeros(policy.model.observation_space.shape),
            runtime_recipe_digest=stress.recipe_digest,
        )
        == 2
    )
    assert canonical_json_bytes(policy.manifest) == original


@pytest.mark.parametrize(
    "change", ["extra", "factor", "schema", "risk", "cost", "H", "model", "archive"]
)
def test_rehashed_invalid_binding_and_original_pins_are_rejected(
    tmp_path, monkeypatch, change
):
    module, policy, raw, _ = view_case(tmp_path, monkeypatch)
    invalid = deepcopy(raw)
    if change == "extra":
        invalid["allowed_digests"] = []
    elif change == "factor":
        invalid["fee_factor"] = True
    elif change == "schema":
        invalid["schema"] = "allocation_fee_stress_binding_v2"
    elif change == "risk":
        invalid["stress_recipe"]["runtime_profile"]["risk_digest"] = "f" * 64
    elif change == "cost":
        invalid["stress_cost_payload"]["spread_rate"] = 0.01
    elif change == "H":
        invalid["stress_recipe"]["observation"]["episode_steps"] = 2
    else:
        invalid[
            "model_state_digest" if change == "model" else "original_policy_sha256"
        ] = "f" * 64
    # No merely self-consistent digest can waive the domain/policy crosslinks.
    content_digest(invalid)
    with pytest.raises(ValueError):
        binding = module.AllocationFeeStressBinding(invalid)
        module.AllocationFeeStressPolicyView(policy, binding)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_native_cost_payload_must_have_complete_closed_roster(
    tmp_path, monkeypatch, change
):
    module, _, raw, _ = view_case(tmp_path, monkeypatch)
    for name in ("base_cost_payload", "stress_cost_payload"):
        if change == "missing":
            raw[name].pop("max_participation_rate")
        else:
            raw[name]["unused_permission"] = True
    for recipe_name, cost_name in (
        ("base_recipe", "base_cost_payload"),
        ("stress_recipe", "stress_cost_payload"),
    ):
        raw[recipe_name]["runtime_profile"]["economics_digest"] = (
            module.retained_debt_economics_digest(raw[cost_name])
        )
    with pytest.raises(ValueError, match="cost|payload"):
        module.AllocationFeeStressBinding(raw)
