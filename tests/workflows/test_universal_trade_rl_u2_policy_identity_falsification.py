from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from tests.workflows.test_universal_trade_rl_u2_training import (
    _assembly,
    _u1_extractor,
    _u2_routed_u1_probe,
)
from trade_rl.rl.checkpointing import checkpoint_identity_payload_for_model
from trade_rl.rl.policy_identity import (
    bind_sb3_policy_identity,
    validated_sb3_policy_identity,
)


def _model(probe: object, assembly: Any) -> SimpleNamespace:
    extractor = _u1_extractor(probe, assembly)
    return SimpleNamespace(
        policy=SimpleNamespace(
            features_extractor=extractor,
            shared_actor_head=assembly.policy_actor_head,
            shared_actor_gate_temperature=assembly.hierarchical_gate_temperature,
            action_distribution_name="masked_shared_squashed_diag_gaussian_v1",
            log_std=SimpleNamespace(shape=(1,)),
            use_sde=False,
        )
    )


def test_u2_u1_checkpoint_identity_carries_exact_adapter_binding() -> None:
    probe = _u2_routed_u1_probe()
    try:
        _observation, _info = probe.reset(seed=0)
        _config, assembly = _assembly(probe)
        model = _model(probe, assembly)
        policy_identity = bind_sb3_policy_identity(model, assembly)

        checkpoint_identity = checkpoint_identity_payload_for_model(model)

        assert checkpoint_identity is not None
        assert checkpoint_identity["schema_version"] == "sb3_checkpoint_identity_v2"
        assert checkpoint_identity["policy"] == policy_identity
        checkpoint_policy = checkpoint_identity["policy"]
        assert isinstance(checkpoint_policy, dict)
        assert checkpoint_policy["sequence_observation_adapter"] == policy_identity[
            "sequence_observation_adapter"
        ]
        assert checkpoint_policy["sequence_observation_adapter_digest"] == (
            policy_identity["sequence_observation_adapter_digest"]
        )
        assert checkpoint_policy["current_weight_observation"] == policy_identity[
            "current_weight_observation"
        ]
    finally:
        probe.close()


def test_u2_u1_serialized_identity_cannot_drop_adapter_binding() -> None:
    probe = _u2_routed_u1_probe()
    try:
        _observation, _info = probe.reset(seed=0)
        _config, assembly = _assembly(probe)
        identity = bind_sb3_policy_identity(_model(probe, assembly), assembly)
        tampered = dict(identity)
        tampered.pop("sequence_observation_adapter")
        tampered.pop("sequence_observation_adapter_digest")

        with pytest.raises(ValueError, match="current-weight observation identity mismatch"):
            validated_sb3_policy_identity(tampered)
    finally:
        probe.close()


def test_u2_u1_policy_identity_rejects_adapter_metadata_drift() -> None:
    probe = _u2_routed_u1_probe()
    try:
        _observation, _info = probe.reset(seed=0)
        _config, assembly = _assembly(probe)
        model = _model(probe, assembly)
        assert assembly.sequence_metadata is not None
        metadata = dict(assembly.sequence_metadata)
        metadata["current_weight_field"] = "previous_action"
        drifted_assembly = replace(assembly, sequence_metadata=metadata)

        with pytest.raises(ValueError, match="current-weight field mismatch"):
            bind_sb3_policy_identity(model, drifted_assembly)
    finally:
        probe.close()


def test_u2_u1_policy_identity_rejects_model_adapter_digest_drift() -> None:
    probe = _u2_routed_u1_probe()
    try:
        _observation, _info = probe.reset(seed=0)
        _config, assembly = _assembly(probe)
        model = _model(probe, assembly)
        model.policy.features_extractor.adapter_contract_digest = "f" * 64

        with pytest.raises(ValueError, match="model/assembly identity mismatch"):
            bind_sb3_policy_identity(model, assembly)
    finally:
        probe.close()
