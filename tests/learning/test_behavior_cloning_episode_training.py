from __future__ import annotations

import numpy as np
import torch

from trade_rl.integrations.behavior_cloning import pretrain_policy
from trade_rl.learning.behavior_cloning import BehaviorCloningConfig
from trade_rl.learning.episode_teacher_artifact import (
    EpisodeSupervisedPolicyDataset,
)


class _Distribution:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean

    def get_actions(self, *, deterministic: bool = False) -> torch.Tensor:
        assert deterministic is True
        return self.mean


class _RecordingPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.device = torch.device("cpu")
        self.training_sample_ids: list[int] = []

    def get_distribution(self, observations: torch.Tensor) -> _Distribution:
        if torch.is_grad_enabled():
            self.training_sample_ids.extend(
                int(value) for value in observations[:, 0].detach().cpu().tolist()
            )
        mean = torch.tanh(self.bias).expand(len(observations), 1)
        return _Distribution(mean)


def _episode_dataset(*, decision_indices: list[int]) -> EpisodeSupervisedPolicyDataset:
    sample_count = len(decision_indices)
    return EpisodeSupervisedPolicyDataset(
        observations=np.arange(sample_count, dtype=np.float32)[:, None],
        actions=np.zeros((sample_count, 1), dtype=np.float32),
        dataset_id="a" * 64,
        train_start=0,
        train_stop=max(decision_indices) + 2,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
        decision_indices=np.asarray(decision_indices, dtype=np.int64),
        episode_ids=np.repeat(np.arange(sample_count // 2, dtype=np.int64), 2),
    )


def test_episode_behavior_cloning_excludes_purged_rows_from_training_and_metrics() -> (
    None
):
    dataset = _episode_dataset(decision_indices=[0, 1, 4, 5, 6, 7, 8, 9])
    actions = np.asarray(dataset.actions).copy()
    actions[4:6] = 1.0
    dataset = EpisodeSupervisedPolicyDataset(
        observations=dataset.observations,
        actions=actions,
        dataset_id=dataset.dataset_id,
        train_start=dataset.train_start,
        train_stop=dataset.train_stop,
        environment_digest=dataset.environment_digest,
        action_spec_digest=dataset.action_spec_digest,
        teacher_config_digest=dataset.teacher_config_digest,
        decision_indices=dataset.decision_indices,
        episode_ids=dataset.episode_ids,
    )
    policy = _RecordingPolicy()

    result = pretrain_policy(
        policy,
        dataset,
        config=BehaviorCloningConfig(
            epochs=1,
            learning_rate=0.01,
            batch_size=8,
            validation_fraction=0.26,
        ),
        seed=7,
    )

    assert sorted(policy.training_sample_ids) == [0, 1, 2, 3]
    assert result.initial_mse == 0.0
    assert result.final_mse == 0.0
    assert result.sample_count == 8
    assert result.training_sample_count == 4
    assert result.validation_sample_count == 2
    assert result.excluded_sample_count == 2
    assert len(result.split_digest) == 64


def test_behavior_cloning_result_identity_binds_the_explicit_episode_split() -> None:
    config = BehaviorCloningConfig(
        epochs=1,
        learning_rate=1e-12,
        batch_size=8,
        validation_fraction=0.26,
    )
    chronological = pretrain_policy(
        _RecordingPolicy(),
        _episode_dataset(decision_indices=[0, 1, 4, 5, 6, 7, 8, 9]),
        config=config,
        seed=11,
    )
    reordered = pretrain_policy(
        _RecordingPolicy(),
        _episode_dataset(decision_indices=[8, 9, 0, 1, 4, 5, 6, 7]),
        config=config,
        seed=11,
    )

    assert chronological.training_sample_count == reordered.training_sample_count == 4
    assert (
        chronological.validation_sample_count == reordered.validation_sample_count == 2
    )
    assert chronological.excluded_sample_count == reordered.excluded_sample_count == 2
    assert chronological.split_digest != reordered.split_digest
    assert chronological.digest != reordered.digest
