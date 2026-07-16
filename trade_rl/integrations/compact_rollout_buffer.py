"""Index-backed SB3 rollout storage for structured sequence policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from trade_rl.data.market import MarketDataset
from trade_rl.rl.sequence_observations import (
    SequenceNormalizerProtocol,
    SequenceObservationBuilder,
    sequence_policy_values,
)

_SEQUENCE_PREFIX = "sequence_"
_DECISION_INDEX_KEY = "decision_index"
_FLOAT16_MAX = float(np.finfo(np.float16).max)


@dataclass(frozen=True, slots=True)
class SequenceRolloutReconstructor:
    """Rebuild overlapping native histories only for sampled PPO minibatches."""

    dataset: MarketDataset
    builder: SequenceObservationBuilder
    normalizer: SequenceNormalizerProtocol | None
    expected_dataset_id: str
    expected_layout_digest: str

    def __post_init__(self) -> None:
        if self.dataset.dataset_id != self.expected_dataset_id:
            raise ValueError("rollout reconstruction dataset identity mismatch")
        if self.builder.layout_digest(self.dataset) != self.expected_layout_digest:
            raise ValueError("rollout reconstruction sequence layout mismatch")

    def reconstruct(self, decision_indices: np.ndarray) -> dict[str, np.ndarray]:
        indices = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("rollout reconstruction indices must not be empty")
        if np.any(indices < self.builder.minimum_index(self.dataset)) or np.any(
            indices >= self.dataset.n_bars
        ):
            raise ValueError("rollout reconstruction index is outside causal history")
        cache: dict[int, dict[str, np.ndarray]] = {}
        for raw_index in np.unique(indices):
            index = int(raw_index)
            sequence = self.builder.build(self.dataset, index=index)
            components: dict[str, np.ndarray] = {}
            for timeframe in sequence.values:
                components[f"sequence_{timeframe}_values"] = sequence_policy_values(
                    timeframe=timeframe,
                    values=sequence.values[timeframe],
                    available=sequence.available[timeframe],
                    feature_names=sequence.feature_names[timeframe],
                    sequence_normalizer=self.normalizer,
                )
                components[f"sequence_{timeframe}_available"] = np.asarray(
                    sequence.available[timeframe], dtype=np.uint8
                )
                components[f"sequence_{timeframe}_staleness"] = np.asarray(
                    np.clip(sequence.staleness[timeframe], 0.0, _FLOAT16_MAX),
                    dtype=np.float16,
                )
            cache[index] = components
        keys = tuple(cache[int(indices[0])])
        return {
            key: np.stack([cache[int(index)][key] for index in indices], axis=0)
            for key in keys
        }


class IndexBackedDictRolloutBuffer(DictRolloutBuffer):
    """Store current state and indices; reconstruct native histories on sampling."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Any = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        *,
        sequence_reconstructor: SequenceRolloutReconstructor,
    ) -> None:
        if _DECISION_INDEX_KEY not in observation_space.spaces:
            raise ValueError("index-backed rollout requires decision_index observation")
        self._sequence_keys = tuple(
            key for key in observation_space.spaces if key.startswith(_SEQUENCE_PREFIX)
        )
        if not self._sequence_keys:
            raise ValueError("index-backed rollout requires sequence components")
        self._compact_keys = tuple(
            key for key in observation_space.spaces if key not in self._sequence_keys
        )
        self.sequence_reconstructor = sequence_reconstructor
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        self.observations = {
            key: np.zeros(
                (self.buffer_size, self.n_envs, *self.obs_shape[key]),
                dtype=np.dtype(self.observation_space.spaces[key].dtype),
            )
            for key in self._compact_keys
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
            raise TypeError("index-backed Dict rollout requires mapping observations")
        missing = set(self.observation_space.spaces).difference(obs)
        if missing:
            raise ValueError(
                f"rollout observation is missing components: {sorted(missing)}"
            )
        decision_index = np.asarray(obs[_DECISION_INDEX_KEY])
        if not np.issubdtype(decision_index.dtype, np.integer):
            raise ValueError("decision_index observation must be integral")
        super().add(obs, action, reward, episode_start, value, log_prob)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> Any:
        if env is not None:
            raise ValueError(
                "index-backed sequence rollout does not support VecNormalize"
            )
        raw_indices = self.observations[_DECISION_INDEX_KEY][batch_inds]
        decision_indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
        observations = {
            key: values[batch_inds]
            for key, values in self.observations.items()
            if key != _DECISION_INDEX_KEY
        }
        observations.update(self.sequence_reconstructor.reconstruct(decision_indices))
        return DictRolloutBufferSamples(
            observations={
                key: self.to_torch(value) for key, value in observations.items()
            },
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )


CompactDictRolloutBuffer = IndexBackedDictRolloutBuffer

__all__ = [
    "CompactDictRolloutBuffer",
    "IndexBackedDictRolloutBuffer",
    "SequenceRolloutReconstructor",
]
