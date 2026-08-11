from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UniversalArchitectureName(str, Enum):
    U_SMALL_DIRECT = "u_small_direct"
    U_MEDIUM_DIRECT = "u_medium_direct"
    U_MEDIUM_GATE = "u_medium_gate"
    U_LARGE_DIRECT = "u_large_direct"


@dataclass(frozen=True)
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
    action_shape: tuple[int, ...] = (1,)
    sequence_dropout: float = 0.0
    shared_scalar_log_std: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.action_shape != (1,):
            raise ValueError("universal single-instrument action shape must be (1,)")
        if self.sequence_dropout != 0.0:
            raise ValueError("maintained universal candidates use zero sequence dropout")


_SPECS = {
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


def architecture_spec(name: UniversalArchitectureName | str) -> UniversalArchitectureSpec:
    resolved = UniversalArchitectureName(name)
    return _SPECS[resolved]


def architecture_ablation_candidates() -> tuple[UniversalArchitectureSpec, ...]:
    return tuple(_SPECS[name] for name in UniversalArchitectureName)
