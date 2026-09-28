"""Stable PPO layout and rollout-budget contract values."""

from __future__ import annotations

import math

PPO_TRAINING_LAYOUT_SEQUENTIAL = "sequential"
PPO_TRAINING_LAYOUT_INTERLEAVED = "interleaved"
PPO_DEFAULT_N_STEPS = 2048
PPO_MINIBATCH_SIZE = 64


def validated_training_layout(
    training_layout: str,
    rollout_steps_per_env: int | None,
) -> str:
    if not isinstance(training_layout, str) or training_layout not in {
        PPO_TRAINING_LAYOUT_SEQUENTIAL,
        PPO_TRAINING_LAYOUT_INTERLEAVED,
    }:
        raise ValueError("training_layout must be 'sequential' or 'interleaved'")
    if (
        training_layout == PPO_TRAINING_LAYOUT_SEQUENTIAL
        and rollout_steps_per_env is not None
    ):
        raise ValueError("sequential training does not accept rollout_steps_per_env")
    return training_layout


def validated_interleaved_rollout_steps(
    rollout_steps_per_env: int | None,
    *,
    n_envs: int,
) -> int:
    if (
        isinstance(rollout_steps_per_env, bool)
        or not isinstance(rollout_steps_per_env, int)
        or rollout_steps_per_env <= 0
    ):
        raise ValueError(
            "rollout_steps_per_env must be a positive integer for interleaved training"
        )
    if rollout_steps_per_env * n_envs % PPO_MINIBATCH_SIZE != 0:
        raise ValueError(
            "interleaved rollout batch must be divisible by PPO batch_size=64"
        )
    return rollout_steps_per_env


def expected_ppo_realized_timesteps(
    total_timesteps: int,
    *,
    training_layout: str,
    rollout_steps_per_env: int | None,
    n_envs: int,
) -> int:
    """Return SB3's rollout-rounded transition count for one PPO fit."""

    if isinstance(total_timesteps, bool) or not isinstance(total_timesteps, int):
        raise ValueError("total_timesteps must be a positive integer")
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be a positive integer")
    if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError("n_envs must be a positive integer")
    layout = validated_training_layout(training_layout, rollout_steps_per_env)
    if layout == PPO_TRAINING_LAYOUT_SEQUENTIAL:
        rollout_transitions = PPO_DEFAULT_N_STEPS
    else:
        rollout_steps = validated_interleaved_rollout_steps(
            rollout_steps_per_env,
            n_envs=n_envs,
        )
        rollout_transitions = rollout_steps * n_envs
    return math.ceil(total_timesteps / rollout_transitions) * rollout_transitions


__all__ = [
    "PPO_DEFAULT_N_STEPS",
    "PPO_MINIBATCH_SIZE",
    "PPO_TRAINING_LAYOUT_INTERLEAVED",
    "PPO_TRAINING_LAYOUT_SEQUENTIAL",
    "expected_ppo_realized_timesteps",
    "validated_interleaved_rollout_steps",
    "validated_training_layout",
]
