from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from trade_rl.integrations.sb3_model_assembly import SB3PolicyAssembly, build_sb3_model
from trade_rl.rl.checkpointing import checkpoint_identity_payload_for_model
from trade_rl.rl.policy_identity import (
    bind_sb3_policy_identity,
    model_sb3_policy_identity,
    validate_model_sb3_policy_identity,
    validated_sb3_policy_identity,
)
from trade_rl.rl.sequence_policy import SequencePolicyArchitecture


def _architecture(*, timeframe_layers: int = 1) -> SequencePolicyArchitecture:
    return SequencePolicyArchitecture(
        input_channels={"15m": 6, "1h": 6, "4h": 6, "1d": 6},
        window_lengths={"15m": 4, "1h": 4, "4h": 4, "1d": 4},
        latent_dims={"15m": 8, "1h": 8, "4h": 8, "1d": 8},
        asset_state_width=4,
        snapshot_width=8,
        n_symbols=3,
        d_model=24,
        timeframe_attention_heads=4,
        timeframe_attention_layers=timeframe_layers,
        timeframe_ffn_multiplier=2,
        timeframe_gate_bias=-2.0,
        asset_attention_heads=4,
        asset_attention_layers=1,
        asset_ffn_multiplier=2,
        asset_gate_bias=-2.0,
        dropout=0.0,
        encoder_widths={
            "15m": (8, 8),
            "1h": (8, 8),
            "4h": (8, 8),
            "1d": (8, 8),
        },
    )


def _model(architecture: SequencePolicyArchitecture) -> SimpleNamespace:
    extractor = SimpleNamespace(
        asset_encoder=SimpleNamespace(architecture=architecture)
    )
    return SimpleNamespace(
        policy=SimpleNamespace(
            features_extractor=extractor,
            shared_actor_head="hierarchical_gate_target_v1",
            shared_actor_gate_temperature=1.0,
            action_distribution_name="masked_shared_squashed_diag_gaussian_v1",
            log_std=SimpleNamespace(shape=(1,)),
            use_sde=False,
        )
    )


def _assembly() -> SimpleNamespace:
    return SimpleNamespace(
        observation_encoder="hierarchical_sequence_v2",
        sequence_symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        sequence_action_names=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        policy_actor_head="hierarchical_gate_target_v1",
        hierarchical_gate_temperature=1.0,
    )


def test_identity_binds_actual_architecture_and_exploration_contract() -> None:
    model = _model(_architecture())
    payload = bind_sb3_policy_identity(model, _assembly())

    assert payload["schema_version"] == "sb3_policy_identity_v3"
    assert payload["observation_encoder"] == "hierarchical_sequence_v2"
    assert payload["sequence_architecture_digest"]
    assert payload["policy_architecture_digest"]
    assert payload["actor_head"] == "hierarchical_gate_target_v1"
    assert payload["gate_temperature"] == 1.0
    assert payload["current_weight_observation"]["key"] == "current_weights"
    assert payload["exploration_contract"] == {
        "action_distribution": "masked_shared_squashed_diag_gaussian_v1",
        "change_intensity_coupling": "post_composition_gate_independent_v1",
        "log_std_parameterization": "shared_scalar_v1",
        "state_dependent_noise": False,
        "schema_version": "hierarchical_exploration_v1",
        "squashing": "tanh",
    }
    assert model_sb3_policy_identity(model) == payload


def test_legacy_v2_identity_is_rejected_with_migration_error() -> None:
    payload = bind_sb3_policy_identity(_model(_architecture()), _assembly())
    legacy = {**payload, "schema_version": "sb3_policy_identity_v2"}

    with pytest.raises(ValueError, match="migrate sb3_policy_identity_v2"):
        validated_sb3_policy_identity(legacy)


def test_different_actual_timeframe_architecture_is_rejected() -> None:
    expected_model = _model(_architecture(timeframe_layers=1))
    expected = bind_sb3_policy_identity(expected_model, _assembly())
    loaded_model = _model(_architecture(timeframe_layers=2))
    bind_sb3_policy_identity(loaded_model, _assembly())

    with pytest.raises(ValueError, match="architecture identity mismatch"):
        validate_model_sb3_policy_identity(loaded_model, expected)


def test_unsupported_exploration_distribution_is_rejected() -> None:
    model = _model(_architecture())
    model.policy.action_distribution_name = "gate_scaled_gaussian_experiment"

    with pytest.raises(ValueError, match="action distribution"):
        bind_sb3_policy_identity(model, _assembly())


def test_non_scalar_log_std_parameterization_is_rejected() -> None:
    model = _model(_architecture())
    model.policy.log_std = SimpleNamespace(shape=(3,))

    with pytest.raises(ValueError, match="shared scalar log_std"):
        bind_sb3_policy_identity(model, _assembly())


def test_state_dependent_noise_is_rejected_for_hierarchical_actor() -> None:
    model = _model(_architecture())
    model.policy.use_sde = True

    with pytest.raises(ValueError, match="gSDE"):
        bind_sb3_policy_identity(model, _assembly())


def test_checkpoint_identity_composes_policy_and_algorithm_contracts() -> None:
    class CostModel(SimpleNamespace):
        def checkpoint_identity_payload(self) -> dict[str, object]:
            return {"schema_version": "cost_identity_v1", "cost_heads": 7}

    model = CostModel(**vars(_model(_architecture())))
    policy = bind_sb3_policy_identity(model, _assembly())

    payload = checkpoint_identity_payload_for_model(model)

    assert payload is not None
    assert payload["schema_version"] == "sb3_checkpoint_identity_v2"
    assert payload["policy"] == policy
    assert payload["algorithm"] == {
        "schema_version": "cost_identity_v1",
        "cost_heads": 7,
    }


def test_behavior_cloning_weight_changes_do_not_change_architecture_identity() -> None:
    model = _model(_architecture())
    before = bind_sb3_policy_identity(model, _assembly())
    model.policy.behavior_cloning_updates = 12
    after = bind_sb3_policy_identity(model, _assembly())
    assert after == before


def test_every_sb3_constructor_return_is_bound_to_policy_identity() -> None:
    source = inspect.getsource(build_sb3_model)
    assert "bind_sb3_policy_identity" in source
    assert source.count("_bind_identity(") >= 4


def test_policy_assembly_exposes_identity_inputs() -> None:
    fields = SB3PolicyAssembly.__dataclass_fields__
    assert "observation_encoder" in fields
    assert "sequence_symbols" in fields
    assert "sequence_action_names" in fields
    assert "policy_actor_head" in fields
    assert "hierarchical_gate_temperature" in fields


def test_actor_head_and_temperature_drift_are_rejected() -> None:
    model = _model(_architecture())
    expected = bind_sb3_policy_identity(model, _assembly())
    drifted = _model(_architecture())
    drifted.policy.shared_actor_gate_temperature = 0.5
    assembly = _assembly()
    assembly.hierarchical_gate_temperature = 0.5
    bind_sb3_policy_identity(drifted, assembly)
    with pytest.raises(ValueError, match="architecture identity mismatch"):
        validate_model_sb3_policy_identity(drifted, expected)
    wrong_head = _model(_architecture())
    wrong_head.policy.shared_actor_head = "legacy_shared_target_v1"
    with pytest.raises(ValueError, match="actor-head"):
        bind_sb3_policy_identity(wrong_head, _assembly())
