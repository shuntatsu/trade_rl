from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trade_rl.rl.policies import (
    SharedPerAssetActionHead,
    SharedPerAssetActorCriticPolicy,
    SharedPerAssetGateTargetHead,
)
from trade_rl.rl.policy_identity import (
    bind_sb3_policy_identity,
    validate_model_sb3_policy_identity,
    validated_sb3_policy_identity,
)
from trade_rl.rl.sequence_policy import SequencePolicyArchitecture
from trade_rl.rl.training import ResidualTrainingConfig


@pytest.mark.parametrize(
    "actor_head",
    ("hierarchical_gate_target_v1", "shared_target_v1"),
)
def test_sequence_training_accepts_supported_action_ablation_heads(
    actor_head: str,
) -> None:
    config = ResidualTrainingConfig(
        timesteps=8,
        gamma=0.99,
        seeds=(0,),
        n_steps=8,
        batch_size=8,
        policy="MultiInputPolicy",
        observation_encoder="hierarchical_sequence_v2",
        policy_actor_head=actor_head,
    )

    assert config.policy_actor_head == actor_head


def test_sequence_training_rejects_unknown_action_ablation_head() -> None:
    with pytest.raises(ValueError, match="policy_actor_head"):
        ResidualTrainingConfig(
            timesteps=8,
            gamma=0.99,
            seeds=(0,),
            n_steps=8,
            batch_size=8,
            policy="MultiInputPolicy",
            observation_encoder="hierarchical_sequence_v2",
            policy_actor_head="discrete_buy_sell_hold_v1",
        )


def test_direct_sequence_head_rejects_active_gate_temperature() -> None:
    with pytest.raises(ValueError, match="inactive.*shared_target_v1"):
        ResidualTrainingConfig(
            timesteps=8,
            gamma=0.99,
            seeds=(0,),
            n_steps=8,
            batch_size=8,
            policy="MultiInputPolicy",
            observation_encoder="hierarchical_sequence_v2",
            policy_actor_head="shared_target_v1",
            hierarchical_gate_temperature=0.5,
        )


class _ActorPassThrough(nn.Module):
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features


class _PolicyHarness(SharedPerAssetActorCriticPolicy):
    def __init__(self, action_net: nn.Module) -> None:
        nn.Module.__init__(self)
        self.action_net = action_net
        self.mlp_extractor = _ActorPassThrough()

    def extract_features(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return observations["features"]


def _actor_latent() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0, 0.20, 1.0, 0.0, 0.0, -0.30, 0.0]],
        dtype=torch.float32,
    )


@pytest.mark.parametrize(
    ("head", "expects_change_intensity"),
    (
        (
            SharedPerAssetGateTargetHead(
                n_symbols=2,
                token_dim=1,
                context_dim=4,
                hidden_dims=(4,),
            ),
            True,
        ),
        (
            SharedPerAssetActionHead(
                n_symbols=2,
                token_dim=1,
                context_dim=4,
                hidden_dims=(4,),
            ),
            False,
        ),
    ),
)
def test_both_sequence_heads_expose_common_action_stage_outputs(
    head: nn.Module,
    expects_change_intensity: bool,
) -> None:
    policy = _PolicyHarness(head)

    outputs = policy.action_stage_outputs({"features": _actor_latent()})

    torch.testing.assert_close(
        outputs.current_weights,
        torch.tensor([[0.20, 0.0]], dtype=torch.float32),
    )
    assert outputs.deterministic_actions.shape == (1, 2)
    assert outputs.active_mask.tolist() == [[True, False]]
    assert outputs.deterministic_actions[0, 1].item() == 0.0
    assert (outputs.change_intensity is not None) is expects_change_intensity


def _architecture() -> SequencePolicyArchitecture:
    return SequencePolicyArchitecture(
        input_channels={"15m": 6, "1h": 6, "4h": 6, "1d": 6},
        window_lengths={"15m": 4, "1h": 4, "4h": 4, "1d": 4},
        latent_dims={"15m": 8, "1h": 8, "4h": 8, "1d": 8},
        asset_state_width=4,
        snapshot_width=8,
        n_symbols=3,
        d_model=24,
        timeframe_attention_heads=4,
        timeframe_attention_layers=1,
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


def _identity_model(actor_head: str) -> SimpleNamespace:
    extractor = SimpleNamespace(
        asset_encoder=SimpleNamespace(architecture=_architecture())
    )
    return SimpleNamespace(
        policy=SimpleNamespace(
            features_extractor=extractor,
            shared_actor_head=actor_head,
            shared_actor_gate_temperature=1.0,
            action_distribution_name="masked_shared_squashed_diag_gaussian_v1",
            log_std=SimpleNamespace(shape=(1,)),
            use_sde=False,
        )
    )


def _identity_assembly(actor_head: str) -> SimpleNamespace:
    return SimpleNamespace(
        observation_encoder="hierarchical_sequence_v2",
        sequence_symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        sequence_action_names=(
            "target_weight:BTCUSDT",
            "target_weight:ETHUSDT",
            "target_weight:BNBUSDT",
        ),
        policy_actor_head=actor_head,
        hierarchical_gate_temperature=1.0,
    )


@pytest.mark.parametrize(
    (
        "actor_head",
        "coupling_field",
        "coupling_value",
        "exploration_schema",
        "gate_temperature",
    ),
    (
        (
            "hierarchical_gate_target_v1",
            "change_intensity_coupling",
            "post_composition_gate_independent_v1",
            "hierarchical_exploration_v1",
            1.0,
        ),
        (
            "shared_target_v1",
            "mean_coupling",
            "direct_target_mean_v1",
            "direct_target_weight_exploration_v1",
            None,
        ),
    ),
)
def test_policy_identity_v4_binds_each_action_head(
    actor_head: str,
    coupling_field: str,
    coupling_value: str,
    exploration_schema: str,
    gate_temperature: float | None,
) -> None:
    payload = bind_sb3_policy_identity(
        _identity_model(actor_head),
        _identity_assembly(actor_head),
    )

    assert payload["schema_version"] == "sb3_policy_identity_v4"
    assert payload["actor_head"] == actor_head
    assert payload["gate_temperature"] == gate_temperature
    assert payload["asset_binding"]["action_names"] == (
        "target_weight:BTCUSDT",
        "target_weight:ETHUSDT",
        "target_weight:BNBUSDT",
    )
    exploration = payload["exploration_contract"]
    assert exploration[coupling_field] == coupling_value
    assert exploration["schema_version"] == exploration_schema


def test_policy_identity_rejects_cross_head_loading() -> None:
    hierarchical = bind_sb3_policy_identity(
        _identity_model("hierarchical_gate_target_v1"),
        _identity_assembly("hierarchical_gate_target_v1"),
    )
    direct_model = _identity_model("shared_target_v1")
    bind_sb3_policy_identity(direct_model, _identity_assembly("shared_target_v1"))

    with pytest.raises(ValueError, match="architecture identity mismatch"):
        validate_model_sb3_policy_identity(direct_model, hierarchical)


def test_policy_identity_v4_round_trips_through_validation() -> None:
    payload = bind_sb3_policy_identity(
        _identity_model("hierarchical_gate_target_v1"),
        _identity_assembly("hierarchical_gate_target_v1"),
    )

    assert validated_sb3_policy_identity(payload) == payload
