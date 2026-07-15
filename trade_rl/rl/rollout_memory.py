"""Fail-closed memory accounting for Stable-Baselines3 PPO rollouts."""

from __future__ import annotations

import math

from gymnasium import spaces

_FLOAT32_BYTES = 4
_PPO_SCALAR_ARRAYS = (
    6  # rewards, returns, episode starts, values, log probs, advantages
)


def _space_elements(observation_space: spaces.Space) -> int:
    if isinstance(observation_space, spaces.Dict):
        return sum(
            _space_elements(space) for space in observation_space.spaces.values()
        )
    shape = observation_space.shape
    if shape is None:
        raise ValueError("observation space must declare a finite shape")
    return math.prod(int(width) for width in shape)


def estimate_ppo_rollout_buffer_bytes(
    observation_space: spaces.Space,
    *,
    n_steps: int,
    n_envs: int,
    action_dim: int,
) -> int:
    """Return exact NumPy payload bytes allocated by SB3 RolloutBuffer.reset()."""

    for name, value in (
        ("n_steps", n_steps),
        ("n_envs", n_envs),
        ("action_dim", action_dim),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    elements = _space_elements(observation_space) + action_dim + _PPO_SCALAR_ARRAYS
    return elements * n_steps * n_envs * _FLOAT32_BYTES


__all__ = ["estimate_ppo_rollout_buffer_bytes"]
