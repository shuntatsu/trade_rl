from __future__ import annotations

import numpy as np
import pytest
import torch

from trade_rl.integrations.behavior_cloning import pretrain_policy
from trade_rl.learning.behavior_cloning import BehaviorCloningConfig
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset


class _Distribution:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean

    def get_actions(self, *, deterministic: bool = False) -> torch.Tensor:
        assert deterministic is True
        return self.mean


class _LinearPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(2, 1)
        torch.nn.init.zeros_(self.actor.weight)
        torch.nn.init.zeros_(self.actor.bias)
        self.device = torch.device("cpu")

    def get_distribution(self, observations: torch.Tensor) -> _Distribution:
        return _Distribution(torch.tanh(self.actor(observations)))


def _dataset() -> SupervisedPolicyDataset:
    observations = np.asarray(
        [
            [-1.0, 0.0],
            [-0.5, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.25, 0.0],
            [-0.25, 0.0],
        ],
        dtype=np.float32,
    )
    return SupervisedPolicyDataset(
        observations=observations,
        actions=observations[:, :1].copy(),
        dataset_id="a" * 64,
        train_start=0,
        train_stop=7,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
    )


def _split() -> BehaviorCloningSplit:
    return BehaviorCloningSplit(
        train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        validation_indices=np.asarray([4, 5], dtype=np.int64),
        train_episode_ids=np.asarray([0], dtype=np.int64),
        validation_episode_ids=np.asarray([1], dtype=np.int64),
    )


def _config() -> BehaviorCloningConfig:
    return BehaviorCloningConfig(
        epochs=2,
        learning_rate=0.01,
        batch_size=2,
        validation_fraction=1 / 3,
    )


def test_behavior_cloning_accepts_validated_custom_train_batches() -> None:
    calls: list[int] = []

    def provider(
        epoch: int,
        train_indices: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, ...]:
        calls.append(epoch)
        assert set(train_indices.tolist()) == {0, 1, 2, 3}
        assert batch_size == 2
        return (
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([1, 3], dtype=np.int64),
        )

    result = pretrain_policy(
        _LinearPolicy(),
        _dataset(),
        config=_config(),
        seed=7,
        split=_split(),
        training_batch_provider=provider,
    )

    assert calls == [1, 2]
    assert result.training_sample_count == 4
    assert result.validation_sample_count == 2


def test_behavior_cloning_rejects_custom_batch_outside_train_scope() -> None:
    def provider(
        epoch: int,
        train_indices: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, ...]:
        del epoch, train_indices, batch_size
        return (
            np.asarray([0, 4], dtype=np.int64),
            np.asarray([1, 2], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="train scope"):
        pretrain_policy(
            _LinearPolicy(),
            _dataset(),
            config=_config(),
            seed=7,
            split=_split(),
            training_batch_provider=provider,
        )


def test_behavior_cloning_requires_custom_batches_to_cover_train_scope() -> None:
    def provider(
        epoch: int,
        train_indices: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, ...]:
        del epoch, train_indices, batch_size
        return (np.asarray([0, 1], dtype=np.int64),)

    with pytest.raises(ValueError, match="cover the full train scope"):
        pretrain_policy(
            _LinearPolicy(),
            _dataset(),
            config=_config(),
            seed=7,
            split=_split(),
            training_batch_provider=provider,
        )
