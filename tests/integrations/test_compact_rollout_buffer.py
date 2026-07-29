from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from gymnasium import spaces

from trade_rl.integrations.compact_rollout_buffer import IndexBackedDictRolloutBuffer


class _Reconstructor:
    def reconstruct(self, decision_indices: np.ndarray) -> dict[str, np.ndarray]:
        indices = np.asarray(decision_indices, dtype=np.float32).reshape(-1)
        return {
            "sequence_15m_values": indices.reshape(-1, 1, 1, 1).astype(np.float16),
        }


def _space(*, current_weights: spaces.Space | None = None) -> spaces.Dict:
    weights = (
        spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        if current_weights is None
        else current_weights
    )
    return spaces.Dict(
        {
            "decision_index": spaces.Box(0, 100, shape=(1,), dtype=np.int64),
            "current_weights": weights,
            "current_snapshot": spaces.Box(
                -np.inf, np.inf, shape=(2, 1), dtype=np.float32
            ),
            "sequence_15m_values": spaces.Box(
                -np.inf, np.inf, shape=(1, 1, 1), dtype=np.float16
            ),
        }
    )


def _buffer(
    *,
    buffer_size: int = 1,
    observation_space: spaces.Dict | None = None,
    reconstructor: Any | None = None,
) -> IndexBackedDictRolloutBuffer:
    return IndexBackedDictRolloutBuffer(
        buffer_size,
        _space() if observation_space is None else observation_space,
        spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        sequence_reconstructor=_Reconstructor()
        if reconstructor is None
        else reconstructor,
    )


def _observation(current_weights: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "decision_index": np.asarray([[5]], dtype=np.int64),
        "current_weights": current_weights,
        "current_snapshot": np.zeros((1, 2, 1), dtype=np.float32),
        "sequence_15m_values": np.zeros((1, 1, 1, 1), dtype=np.float16),
    }


def _add(
    buffer: IndexBackedDictRolloutBuffer, observation: dict[str, np.ndarray]
) -> None:
    buffer.add(
        observation,
        action=np.zeros((1, 2), dtype=np.float32),
        reward=np.zeros(1, dtype=np.float32),
        episode_start=np.ones(1, dtype=np.float32),
        value=torch.zeros(1),
        log_prob=torch.zeros(1),
    )


def test_current_weights_round_trip_through_compact_storage_and_samples() -> None:
    buffer = _buffer()
    weights = np.asarray([[0.25, -0.5]], dtype=np.float32)

    _add(buffer, _observation(weights))
    weights[:] = 0.0

    assert buffer.observations["current_weights"].shape == (1, 1, 2)
    assert buffer.observations["current_weights"].dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        buffer.observations["current_weights"][0, 0],
        np.asarray([0.25, -0.5], dtype=np.float32),
    )

    samples = next(buffer.get(batch_size=1))
    torch.testing.assert_close(
        samples.observations["current_weights"],
        torch.tensor([[0.25, -0.5]], dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.asarray([[0.1, 0.2]], dtype=np.float64), "float32"),
        (np.asarray([[0.1, np.nan]], dtype=np.float32), "finite"),
        (np.asarray([[0.1]], dtype=np.float32), "shape"),
        (np.asarray([[0.1, 1.1]], dtype=np.float32), "within"),
    ],
)
def test_current_weight_rollout_values_fail_closed(
    weights: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _add(_buffer(), _observation(weights))


def test_compact_rollout_rejects_missing_current_weight_value() -> None:
    observation = _observation(np.zeros((1, 2), dtype=np.float32))
    del observation["current_weights"]

    with pytest.raises(ValueError, match="missing components.*current_weights"):
        _add(_buffer(), observation)


@pytest.mark.parametrize(
    ("weight_space", "message"),
    [
        (spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64), "float32"),
        (spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32), "shape"),
        (spaces.Box(-0.5, 0.5, shape=(2,), dtype=np.float32), "bounds"),
    ],
)
def test_current_weight_observation_contract_fails_closed(
    weight_space: spaces.Space,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _buffer(observation_space=_space(current_weights=weight_space))


def test_compact_rollout_requires_current_weight_observation_space() -> None:
    observation_space = _space()
    del observation_space.spaces["current_weights"]

    with pytest.raises(ValueError, match="current_weights"):
        _buffer(observation_space=observation_space)


def test_sampling_rejects_uninitialized_current_weight_storage() -> None:
    buffer = _buffer(buffer_size=2)

    with pytest.raises(ValueError, match="uninitialized"):
        buffer._get_samples(np.asarray([0], dtype=np.int64))
