"""Memory-bounded SB3 Dict rollout storage for structured sequence policies."""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.buffers import DictRolloutBuffer


class CompactDictRolloutBuffer(DictRolloutBuffer):
    """Retain each observation component in its declared Gymnasium dtype.

    Stable-Baselines3 2.3.2 allocates every Dict observation component as
    float32. Structured histories deliberately declare normalized values and
    staleness as float16 and availability as uint8. This implementation mirrors
    ``DictRolloutBuffer.reset`` without first allocating the large float32
    tensors, preventing both persistent and transient duplicate storage. Policy
    computation remains float32 because the feature extractor casts inputs
    explicitly.
    """

    def reset(self) -> None:
        self.observations = {
            key: np.zeros(
                (self.buffer_size, self.n_envs, *obs_input_shape),
                dtype=np.dtype(self.observation_space.spaces[key].dtype),
            )
            for key, obs_input_shape in self.obs_shape.items()
        }
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray | dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: Any,
        log_prob: Any,
    ) -> None:
        if not isinstance(obs, dict):
            raise TypeError("compact Dict rollout buffer requires mapping observations")
        cast_observations = {
            key: np.asarray(item, dtype=self.observation_space.spaces[key].dtype)
            for key, item in obs.items()
        }
        super().add(
            cast_observations,
            action,
            reward,
            episode_start,
            value,
            log_prob,
        )


__all__ = ["CompactDictRolloutBuffer"]
