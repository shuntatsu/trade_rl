"""Typed architecture candidates for universal single-instrument policies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from trade_rl.rl.training import ResidualTrainingConfig


class UniversalArchitectureName(str, Enum):
    U_SMALL_DIRECT = "u_small_direct"
    U_MEDIUM_DIRECT = "u_medium_direct"
    U_MEDIUM_GATE = "u_medium_gate"
    U_LARGE_DIRECT = "u_large_direct"


@dataclass(frozen=True, slots=True)
class UniversalArchitectureSpec:
    name: UniversalArchitectureName
    tcn_capacity: str
    d_model: int
    attention_heads: int
    attention_layers: int
    ffn_multiplier: int
    actor_head: str
    actor_mlp: tuple[int, ...]
    critic_mlp: tuple[int, ...]
    sequence_dropout: float = 0.0
    action_shape: tuple[int, ...] = (1,)


_SPECS: dict[UniversalArchitectureName, UniversalArchitectureSpec] = {
    UniversalArchitectureName.U_SMALL_DIRECT: UniversalArchitectureSpec(
        name=UniversalArchitectureName.U_SMALL_DIRECT,
        tcn_capacity="compact",
        d_model=192,
        attention_heads=4,
        attention_layers=1,
        ffn_multiplier=3,
        actor_head="shared_target_v1",
        actor_mlp=(256, 128),
        critic_mlp=(256, 128),
    ),
    UniversalArchitectureName.U_MEDIUM_DIRECT: UniversalArchitectureSpec(
        name=UniversalArchitectureName.U_MEDIUM_DIRECT,
        tcn_capacity="compact",
        d_model=256,
        attention_heads=4,
        attention_layers=1,
        ffn_multiplier=3,
        actor_head="shared_target_v1",
        actor_mlp=(256, 128),
        critic_mlp=(256, 128),
    ),
    UniversalArchitectureName.U_MEDIUM_GATE: UniversalArchitectureSpec(
        name=UniversalArchitectureName.U_MEDIUM_GATE,
        tcn_capacity="compact",
        d_model=256,
        attention_heads=4,
        attention_layers=1,
        ffn_multiplier=3,
        actor_head="hierarchical_gate_target_v1",
        actor_mlp=(256, 128),
        critic_mlp=(256, 128),
    ),
    UniversalArchitectureName.U_LARGE_DIRECT: UniversalArchitectureSpec(
        name=UniversalArchitectureName.U_LARGE_DIRECT,
        tcn_capacity="standard",
        d_model=336,
        attention_heads=8,
        attention_layers=2,
        ffn_multiplier=3,
        actor_head="shared_target_v1",
        actor_mlp=(256, 128),
        critic_mlp=(256, 128),
    ),
}


def architecture_spec(
    name: UniversalArchitectureName | str,
) -> UniversalArchitectureSpec:
    resolved = UniversalArchitectureName(name)
    return _SPECS[resolved]


def architecture_ablation_candidates() -> tuple[UniversalArchitectureSpec, ...]:
    return tuple(_SPECS[name] for name in UniversalArchitectureName)


def apply_architecture_to_training_config(
    config: ResidualTrainingConfig,
    name: UniversalArchitectureName | str,
) -> ResidualTrainingConfig:
    """Project one U5 candidate into the existing SB3 assembly contract."""

    spec = architecture_spec(name)
    return replace(
        config,
        policy="MultiInputPolicy",
        observation_encoder="hierarchical_sequence_v2",
        sequence_tcn_capacity=spec.tcn_capacity,
        sequence_d_model=spec.d_model,
        sequence_timeframe_attention_heads=spec.attention_heads,
        sequence_timeframe_attention_layers=spec.attention_layers,
        sequence_timeframe_ffn_multiplier=spec.ffn_multiplier,
        sequence_dropout=spec.sequence_dropout,
        policy_actor_head=spec.actor_head,
        policy_net_arch=spec.actor_mlp,
        value_net_arch=spec.critic_mlp,
    )
