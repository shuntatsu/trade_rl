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


def test_episode_behavior_cloning_excludes_purged_rows_from_training_and_metrics() -> (
    None
):
    observations = np.arange(8, dtype=np.float32)[:, None]
    actions = np.zeros((8, 1), dtype=np.float32)
    actions[4:6] = 1.0
    dataset = EpisodeSupervisedPolicyDataset(
        observations=observations,
        actions=actions,
        dataset_id="a" * 64,
        train_start=0,
        train_stop=11,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
        decision_indices=np.asarray([0, 1, 4, 5, 6, 7, 8, 9], dtype=np.int64),
        episode_ids=np.repeat(np.arange(4, dtype=np.int64), 2),
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
    assert result.validation_sample_count == 2
