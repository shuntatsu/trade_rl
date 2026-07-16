"""Typed algorithm-specific views over the shared training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trade_rl.rl.training import ResidualTrainingConfig


@dataclass(frozen=True, slots=True)
class CommonAlgorithmConfig:
    timesteps: int
    gamma: float
    learning_rate: float
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


AlgorithmConfig = PPOConfig | SACConfig | TD3Config | TQCConfig


def build_algorithm_config(
    source: ResidualTrainingConfig,
    *,
    algorithm: str | None = None,
) -> AlgorithmConfig:
    resolved = source.algorithm if algorithm is None else algorithm.lower()
    if resolved == "ppo":
        return PPOConfig(
            timesteps=source.timesteps,
            gamma=source.gamma,
            learning_rate=source.learning_rate,
            batch_size=source.batch_size,
            policy=source.policy,
            device=source.device,
            policy_net_arch=source.policy_net_arch,
            value_net_arch=source.value_net_arch,
            n_steps=source.n_steps,
            n_epochs=source.n_epochs,
            gae_lambda=source.gae_lambda,
            clip_range=source.clip_range,
            normalize_advantage=source.normalize_advantage,
            ent_coef=source.ent_coef,
            vf_coef=source.vf_coef,
            max_grad_norm=source.max_grad_norm,
            log_std_init=source.log_std_init,
            target_kl=source.target_kl,
            use_sde=source.use_sde,
            sde_sample_freq=source.sde_sample_freq,
        )
    common = dict(
        timesteps=source.timesteps,
        gamma=source.gamma,
        learning_rate=source.learning_rate,
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
    "PPOConfig",
    "SACConfig",
    "TD3Config",
    "TQCConfig",
    "build_algorithm_config",
]
