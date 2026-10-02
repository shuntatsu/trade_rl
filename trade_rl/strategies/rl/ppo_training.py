"""Stable PPO layout and rollout-budget contract values."""

from __future__ import annotations

import math

PPO_TRAINING_LAYOUT_SEQUENTIAL = "sequential"
PPO_TRAINING_LAYOUT_INTERLEAVED = "interleaved"
PPO_DEFAULT_N_STEPS = 2048
PPO_MINIBATCH_SIZE = 64
PPO_DEFAULT_GAMMA = 0.99
PPO_DEFAULT_GAE_LAMBDA = 0.95
PPO_REWARD_SCHEMA = "net_log_return_v1"


def validated_ppo_gamma(value: object) -> float:
    """Validate the PPO temporal discount without changing reward semantics."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("ppo_gamma must be a finite number within (0, 1]")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 < resolved <= 1.0:
        raise ValueError("ppo_gamma must be a finite number within (0, 1]")
    return resolved


def ppo_training_objective_contract_payload(
    *,
    gamma: float = PPO_DEFAULT_GAMMA,
) -> dict[str, object]:
    """Return the explicit PPO optimization contract for research provenance."""

    return {
        "schema": "ppo_training_objective_v1",
        "reward_schema": PPO_REWARD_SCHEMA,
        "reward_scope": "per_symbol_account_after_cost_log_return",
        "terminal_settlement": "included_in_terminal_transition",
        "gamma": validated_ppo_gamma(gamma),
        "gae_lambda": PPO_DEFAULT_GAE_LAMBDA,
        "normalize_advantage": True,
    }


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
    "PPO_DEFAULT_GAE_LAMBDA",
    "PPO_DEFAULT_GAMMA",
    "PPO_DEFAULT_N_STEPS",
    "PPO_MINIBATCH_SIZE",
    "PPO_REWARD_SCHEMA",
    "PPO_TRAINING_LAYOUT_INTERLEAVED",
    "PPO_TRAINING_LAYOUT_SEQUENTIAL",
    "expected_ppo_realized_timesteps",
    "ppo_training_objective_contract_payload",
    "validated_interleaved_rollout_steps",
    "validated_ppo_gamma",
    "validated_training_layout",
]
