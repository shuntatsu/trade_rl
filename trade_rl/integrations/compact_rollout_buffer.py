"""Index-backed SB3 rollout storage for structured sequence policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from trade_rl.data.market import MarketDataset
from trade_rl.rl.sequence_observations import (
    SequenceNormalizerProtocol,
    SequenceObservationBuilder,
    SequencePolicyPlane,
    sequence_policy_values,
)
from trade_rl.rl.training_performance import (
    measure_sequence_reconstruction,
    measure_sequence_tensor_conversion,
)

_SEQUENCE_PREFIX = "sequence_"
_DECISION_INDEX_KEY = "decision_index"
_FLOAT16_MAX = float(np.finfo(np.float16).max)
_SEQUENCE_TRANSFER_MODES = frozenset({"synchronous", "pinned_non_blocking"})


def _validate_sequence_transfer_mode(value: str) -> str:
    if value not in _SEQUENCE_TRANSFER_MODES:
        raise ValueError(
            "sequence_transfer_mode must be synchronous or pinned_non_blocking"
        )
    return value


@dataclass(frozen=True, slots=True)
class SequenceRolloutReconstructor:
    """Rebuild overlapping native histories only for sampled PPO minibatches."""

    dataset: MarketDataset
    builder: SequenceObservationBuilder
    normalizer: SequenceNormalizerProtocol | None
    expected_dataset_id: str
    expected_layout_digest: str
    policy_plane: SequencePolicyPlane | None = None

    def __post_init__(self) -> None:
        if self.dataset.dataset_id != self.expected_dataset_id:
            raise ValueError("rollout reconstruction dataset identity mismatch")
        if self.builder.layout_digest(self.dataset) != self.expected_layout_digest:
            raise ValueError("rollout reconstruction sequence layout mismatch")
        if self.policy_plane is not None and (
            self.policy_plane.dataset_id != self.expected_dataset_id
            or self.policy_plane.layout_digest != self.expected_layout_digest
        ):
            raise ValueError("rollout reconstruction policy plane identity mismatch")

    def reconstruct(self, decision_indices: np.ndarray) -> dict[str, np.ndarray]:
        indices = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("rollout reconstruction indices must not be empty")
        if np.any(indices < self.builder.minimum_index(self.dataset)) or np.any(
            indices >= self.dataset.n_bars
        ):
            raise ValueError("rollout reconstruction index is outside causal history")
        if self.policy_plane is not None:
            return self.policy_plane.batch_components(indices)
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
        sequence_reconstructor: SequenceRolloutReconstructor | None = None,
        sequence_transfer_mode: str = "synchronous",
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
        self.sequence_transfer_mode = _validate_sequence_transfer_mode(
            sequence_transfer_mode
        )
        # Keep one pinned staging tensor per reconstructed component.  Reusing
        # these allocations prevents the pinned allocator from growing by a
        # complete sequence rollout on every PPO update.
        self._pinned_sequence_staging: dict[str, Any] = {}
        self._pinned_sequence_copy_events: dict[str, Any] = {}
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def bind_sequence_reconstructor(
        self,
        reconstructor: SequenceRolloutReconstructor,
        *,
        sequence_transfer_mode: str | None = None,
    ) -> None:
        if not isinstance(reconstructor, SequenceRolloutReconstructor):
            raise TypeError("sequence reconstructor has an invalid type")
        self.sequence_reconstructor = reconstructor
        if sequence_transfer_mode is not None:
            self.sequence_transfer_mode = _validate_sequence_transfer_mode(
                sequence_transfer_mode
            )

    def reset(self) -> None:
        self._materialized_sequence_observations: dict[str, Any] | None = None
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

    def _sequence_to_torch(
        self,
        value: np.ndarray,
        *,
        staging_key: str = "__sequence__",
    ) -> Any:
        # Convert one reconstructed sequence tensor using the configured path.

        mode = _validate_sequence_transfer_mode(
            getattr(self, "sequence_transfer_mode", "synchronous")
        )
        if mode == "synchronous":
            return self.to_torch(value)
        device = torch.device(self.device)
        if device.type != "cuda":
            raise RuntimeError(
                "pinned_non_blocking sequence transfer requires a CUDA device"
            )
        cpu_tensor = torch.as_tensor(value)
        if not cpu_tensor.is_contiguous():
            cpu_tensor = cpu_tensor.contiguous()

        staging = getattr(self, "_pinned_sequence_staging", None)
        if staging is None:
            staging = {}
            self._pinned_sequence_staging = staging
        copy_events = getattr(self, "_pinned_sequence_copy_events", None)
        if copy_events is None:
            copy_events = {}
            self._pinned_sequence_copy_events = copy_events

        previous_copy = copy_events.pop(staging_key, None)
        if previous_copy is not None:
            # The pinned source must not be overwritten until its preceding
            # asynchronous H2D transfer has completed.
            previous_copy.synchronize()

        pinned = staging.get(staging_key)
        if (
            pinned is None
            or pinned.shape != cpu_tensor.shape
            or pinned.dtype != cpu_tensor.dtype
        ):
            pinned = torch.empty_like(cpu_tensor, device="cpu", pin_memory=True)
            staging[staging_key] = pinned
        pinned.copy_(cpu_tensor)
        result = pinned.to(device, non_blocking=True)
        transfer_complete = torch.cuda.Event()
        transfer_complete.record(torch.cuda.current_stream(device))
        copy_events[staging_key] = transfer_complete
        return result

    def _materialize_sequence_observations(
        self, reconstructor: SequenceRolloutReconstructor
    ) -> dict[str, Any]:
        cached = self._materialized_sequence_observations
        if cached is not None:
            return cached
        raw_indices = self.observations[_DECISION_INDEX_KEY]
        decision_indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
        with measure_sequence_reconstruction():
            reconstructed = reconstructor.reconstruct(decision_indices)
        with measure_sequence_tensor_conversion():
            cached = {
                key: self._sequence_to_torch(value, staging_key=key)
                for key, value in reconstructed.items()
            }
        self._materialized_sequence_observations = cached
        return cached

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
        observations = {
            key: values[batch_inds]
            for key, values in self.observations.items()
            if key != _DECISION_INDEX_KEY
        }
        reconstructor = self.sequence_reconstructor
        if reconstructor is None:
            raise RuntimeError("sequence rollout reconstructor is not bound")
        sequence_observations = self._materialize_sequence_observations(reconstructor)
        tensor_indices = self.to_torch(
            np.asarray(batch_inds, dtype=np.int64), copy=False
        )
        torch_observations = {
            key: self.to_torch(value) for key, value in observations.items()
        }
        torch_observations.update(
            {
                key: value.index_select(0, tensor_indices)
                for key, value in sequence_observations.items()
            }
        )
        return DictRolloutBufferSamples(
            observations=torch_observations,
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
