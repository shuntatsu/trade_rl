"""Typed algorithm-specific views over the shared training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trade_rl.rl.cost_learning import (
    CostLearningSchema,
    canonical_cost_learning_schema,
)
from trade_rl.rl.lagrangian import LagrangianSchema, canonical_lagrangian_schema

if TYPE_CHECKING:
    from trade_rl.rl.training import ResidualTrainingConfig


@dataclass(frozen=True, slots=True)
class CommonAlgorithmConfig:
    timesteps: int
    gamma: float
    learning_rate: float
    learning_rate_schedule: str
    learning_rate_final_ratio: float
    batch_size: int
    policy: str
    device: str
    policy_net_arch: tuple[int, ...]
    value_net_arch: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PPOConfig(CommonAlgorithmConfig):
    n_steps: int
    n_epochs: int
    gae_lambda: float
    clip_range: float
    normalize_advantage: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    log_std_init: float
    target_kl: float | None
    use_sde: bool
    sde_sample_freq: int


@dataclass(frozen=True, slots=True)
class CostCriticPPOConfig(PPOConfig):
    """Opt-in PPO view with independent Cost Critic learning settings."""

    cost_schema: CostLearningSchema
    cost_learning_rate: float
    cost_n_epochs: int
    cost_batch_size: int | None
    cost_continuous_hidden_dims: tuple[int, ...]
    cost_event_hidden_dims: tuple[int, ...]
    cost_max_grad_norm: float
    cost_architecture_variant: str


@dataclass(frozen=True, slots=True)
class LagrangianPPOConfig(CostCriticPPOConfig):
    """Cost Critic PPO plus explicit constrained-optimization semantics."""

    lagrangian_schema: LagrangianSchema
    probe_episodes: int
    probe_max_steps_per_episode: int
    actor_composition_mode: str = "raw_lagrangian_then_sb3_normalize_v1"


@dataclass(frozen=True, slots=True)
class OffPolicyConfig(CommonAlgorithmConfig):
    buffer_size: int
    learning_starts: int
    train_freq: int
    gradient_steps: int


@dataclass(frozen=True, slots=True)
class SACConfig(OffPolicyConfig):
    use_sde: bool
    sde_sample_freq: int


@dataclass(frozen=True, slots=True)
class TD3Config(OffPolicyConfig):
    pass


@dataclass(frozen=True, slots=True)
class TQCConfig(OffPolicyConfig):
    use_sde: bool
    sde_sample_freq: int


AlgorithmConfig = (
    LagrangianPPOConfig
    | CostCriticPPOConfig
    | PPOConfig
    | SACConfig
    | TD3Config
    | TQCConfig
)


def _ppo_config_payload(source: ResidualTrainingConfig) -> dict[str, object]:
    return {
        "timesteps": source.timesteps,
        "gamma": source.gamma,
        "learning_rate": source.learning_rate,
        "learning_rate_schedule": source.learning_rate_schedule,
        "learning_rate_final_ratio": source.learning_rate_final_ratio,
        "batch_size": source.batch_size,
        "policy": source.policy,
        "device": source.device,
        "policy_net_arch": source.policy_net_arch,
        "value_net_arch": source.value_net_arch,
        "n_steps": source.n_steps,
        "n_epochs": source.n_epochs,
        "gae_lambda": source.gae_lambda,
        "clip_range": source.clip_range,
        "normalize_advantage": source.normalize_advantage,
        "ent_coef": source.ent_coef,
        "vf_coef": source.vf_coef,
        "max_grad_norm": source.max_grad_norm,
        "log_std_init": source.log_std_init,
        "target_kl": source.target_kl,
        "use_sde": source.use_sde,
        "sde_sample_freq": source.sde_sample_freq,
    }


def _cost_schema(source: ResidualTrainingConfig) -> CostLearningSchema:
    return canonical_cost_learning_schema(
        continuous_gae_lambda=source.cost_continuous_gae_lambda,
        event_gae_lambda=source.cost_event_gae_lambda,
        value_loss_coefficient=source.cost_value_loss_coefficient,
        auxiliary_event_loss_coefficient=(source.cost_auxiliary_event_loss_coefficient),
    )


def build_algorithm_config(
    source: ResidualTrainingConfig,
    *,
    algorithm: str | None = None,
) -> AlgorithmConfig:
    resolved = source.algorithm if algorithm is None else algorithm.lower()
    if resolved == "ppo":
        return PPOConfig(**_ppo_config_payload(source))  # type: ignore[arg-type]
    if resolved == "cost_critic_ppo":
        return CostCriticPPOConfig(
            **_ppo_config_payload(source),  # type: ignore[arg-type]
            cost_schema=_cost_schema(source),
            cost_learning_rate=source.cost_learning_rate,
            cost_n_epochs=source.cost_n_epochs,
            cost_batch_size=source.cost_batch_size,
            cost_continuous_hidden_dims=source.cost_continuous_hidden_dims,
            cost_event_hidden_dims=source.cost_event_hidden_dims,
            cost_max_grad_norm=source.cost_max_grad_norm,
            cost_architecture_variant=source.cost_architecture_variant,
        )
    if resolved == "lagrangian_ppo":
        cost_schema = _cost_schema(source)
        return LagrangianPPOConfig(
            **_ppo_config_payload(source),  # type: ignore[arg-type]
            cost_schema=cost_schema,
            cost_learning_rate=source.cost_learning_rate,
            cost_n_epochs=source.cost_n_epochs,
            cost_batch_size=source.cost_batch_size,
            cost_continuous_hidden_dims=source.cost_continuous_hidden_dims,
            cost_event_hidden_dims=source.cost_event_hidden_dims,
            cost_max_grad_norm=source.cost_max_grad_norm,
            cost_architecture_variant=source.cost_architecture_variant,
            lagrangian_schema=canonical_lagrangian_schema(
                names=cost_schema.names,
                budgets=source.lagrangian_budgets,
                dual_learning_rates=source.lagrangian_dual_learning_rates,
                ema_betas=source.lagrangian_ema_betas,
                initial_multipliers=source.lagrangian_initial_multipliers,
                max_multipliers=source.lagrangian_max_multipliers,
                warmup_rollouts=source.lagrangian_warmup_rollouts,
                update_interval_rollouts=(source.lagrangian_update_interval_rollouts),
                minimum_completed_episodes=(
                    source.lagrangian_minimum_completed_episodes
                ),
            ),
            probe_episodes=source.lagrangian_probe_episodes,
            probe_max_steps_per_episode=(source.lagrangian_probe_max_steps_per_episode),
        )
    common = dict(
        timesteps=source.timesteps,
        gamma=source.gamma,
        learning_rate=source.learning_rate,
        learning_rate_schedule=source.learning_rate_schedule,
        learning_rate_final_ratio=source.learning_rate_final_ratio,
        batch_size=source.batch_size,
        policy=source.policy,
        device=source.device,
        policy_net_arch=source.policy_net_arch,
        value_net_arch=source.value_net_arch,
        buffer_size=source.buffer_size,
        learning_starts=source.learning_starts,
        train_freq=source.train_freq,
        gradient_steps=source.gradient_steps,
    )
    if resolved == "sac":
        return SACConfig(
            timesteps=source.timesteps,
            gamma=source.gamma,
            learning_rate=source.learning_rate,
            learning_rate_schedule=source.learning_rate_schedule,
            learning_rate_final_ratio=source.learning_rate_final_ratio,
            batch_size=source.batch_size,
            policy=source.policy,
            device=source.device,
            policy_net_arch=source.policy_net_arch,
            value_net_arch=source.value_net_arch,
            buffer_size=source.buffer_size,
            learning_starts=source.learning_starts,
            train_freq=source.train_freq,
            gradient_steps=source.gradient_steps,
            use_sde=source.use_sde,
            sde_sample_freq=source.sde_sample_freq,
        )
    if resolved == "td3":
        return TD3Config(**common)  # type: ignore[arg-type]
    if resolved == "tqc":
        return TQCConfig(
            timesteps=source.timesteps,
            gamma=source.gamma,
            learning_rate=source.learning_rate,
            learning_rate_schedule=source.learning_rate_schedule,
            learning_rate_final_ratio=source.learning_rate_final_ratio,
            batch_size=source.batch_size,
            policy=source.policy,
            device=source.device,
            policy_net_arch=source.policy_net_arch,
            value_net_arch=source.value_net_arch,
            buffer_size=source.buffer_size,
            learning_starts=source.learning_starts,
            train_freq=source.train_freq,
            gradient_steps=source.gradient_steps,
            use_sde=source.use_sde,
            sde_sample_freq=source.sde_sample_freq,
        )
    raise ValueError(f"unsupported training algorithm: {resolved}")


__all__ = [
    "AlgorithmConfig",
    "CostCriticPPOConfig",
    "LagrangianPPOConfig",
    "PPOConfig",
    "SACConfig",
    "TD3Config",
    "TQCConfig",
    "build_algorithm_config",
]
