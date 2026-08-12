"""Short-episode Universal Lagrangian mechanics diagnostics."""

from __future__ import annotations

from dataclasses import replace

from trade_rl.rl.training_run_config import TrainingRunConfig


def build_lagrangian_mechanics_config(
    base: TrainingRunConfig,
    *,
    episode_hours: float = 8.0,
    timesteps: int = 1_024,
) -> TrainingRunConfig:
    """Shorten only diagnostic horizons while preserving the economic objective."""

    training = base.training
    if training.algorithm != "lagrangian_ppo":
        raise ValueError("mechanics smoke requires lagrangian_ppo")
    if episode_hours <= 0.0 or episode_hours % base.environment.decision_hours != 0.0:
        raise ValueError("episode_hours must be a positive whole decision count")
    if timesteps <= 0 or timesteps % training.n_steps != 0:
        raise ValueError("timesteps must be a positive multiple of n_steps")
    episode_decisions = int(round(episode_hours / base.environment.decision_hours))
    rollout_count = timesteps // training.n_steps
    completed_episode_capacity = timesteps // episode_decisions
    required_rollouts = max(training.lagrangian_warmup_rollouts) + 1
    required_episodes = max(training.lagrangian_minimum_completed_episodes)
    if rollout_count < required_rollouts:
        raise ValueError("timesteps do not extend beyond Lagrangian warmup")
    if completed_episode_capacity < required_episodes:
        raise ValueError("timesteps cannot satisfy completed-episode requirements")
    mechanics_training = replace(
        training,
        timesteps=timesteps,
        checkpoint_interval_steps=training.n_steps,
        max_checkpoints=rollout_count,
        lagrangian_probe_max_steps_per_episode=episode_decisions,
    )
    mechanics_environment = replace(base.environment, episode_hours=episode_hours)
    resolved = replace(
        base,
        training=mechanics_training,
        environment=mechanics_environment,
    )
    if resolved.reward != base.reward:
        raise RuntimeError("mechanics smoke changed the reward objective")
    return resolved


__all__ = ["build_lagrangian_mechanics_config"]
