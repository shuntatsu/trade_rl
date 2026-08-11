from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset
from trade_rl.rl.training import ResidualTrainingConfig


def _digest(label: str) -> str:
    return content_digest({"label": label})


class _Distribution:
    def __init__(self, actions: torch.Tensor) -> None:
        self._actions = actions

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        assert deterministic
        return self._actions


class _CriticExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.critic_net = nn.Sequential(nn.Linear(1, 8), nn.Tanh())


class _Policy(nn.Module):
    device = torch.device("cpu")

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(3)
        self.actor = nn.Linear(1, 1)
        self.mlp_extractor = _CriticExtractor()
        self.value_net = nn.Linear(8, 1)

    def get_distribution(self, observations: torch.Tensor) -> _Distribution:
        return _Distribution(self.actor(observations.float()))

    def predict_values(self, observations: torch.Tensor) -> torch.Tensor:
        latent = self.mlp_extractor.critic_net(observations.float())
        return self.value_net(latent)


class _RewardEnvironment:
    def __init__(self) -> None:
        self.current_index = 0
        self._remaining = 0
        self._rewards = {0: 1.0, 1: 2.0, 10: 3.0, 11: 4.0}

    def reset(
        self, *, options: dict[str, object]
    ) -> tuple[np.ndarray, dict[str, object]]:
        start = int(options["start_idx"])
        self.current_index = start
        self._remaining = int(options["episode_bars"])
        return np.asarray([float(start)], dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        reward = self._rewards[self.current_index]
        self.current_index += 1
        self._remaining -= 1
        return (
            np.asarray([float(self.current_index)], dtype=np.float32),
            reward,
            self._remaining == 0,
            False,
            {},
        )


def _teacher_dataset() -> SupervisedPolicyDataset:
    return SupervisedPolicyDataset(
        observations=np.asarray([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32),
        actions=np.asarray([[-0.8], [-0.4], [0.4], [0.8]], dtype=np.float32),
        dataset_id=_digest("dataset"),
        train_start=0,
        train_stop=5,
        environment_digest=_digest("environment"),
        action_spec_digest=_digest("action"),
        teacher_config_digest=_digest("teacher"),
    )


def _episode_batch() -> EpisodeOracleBatch:
    dataset_id = _digest("dataset")
    return EpisodeOracleBatch(
        dataset_id=dataset_id,
        teacher_config_digest=_digest("teacher"),
        sampling_config_digest=_digest("sampling"),
        contracts=(
            OracleEpisodeContract(
                dataset_id=dataset_id,
                episode_index=0,
                start=0,
                stop=3,
                initial_state_mode="cash",
                initial_weights=np.zeros(1, dtype=np.float64),
            ),
            OracleEpisodeContract(
                dataset_id=dataset_id,
                episode_index=1,
                start=10,
                stop=13,
                initial_state_mode="cash",
                initial_weights=np.zeros(1, dtype=np.float64),
            ),
        ),
        targets=(
            np.asarray([[0.1], [0.2]], dtype=np.float32),
            np.asarray([[0.3], [0.4]], dtype=np.float32),
        ),
    )


def _split() -> BehaviorCloningSplit:
    return BehaviorCloningSplit(
        train_indices=np.asarray([0, 1], dtype=np.int64),
        validation_indices=np.asarray([2, 3], dtype=np.int64),
        train_episode_ids=np.asarray([0], dtype=np.int64),
        validation_episode_ids=np.asarray([1], dtype=np.int64),
    )


def _config() -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=128,
        gamma=1.0,
        seeds=(7,),
        n_steps=8,
        batch_size=8,
        behavior_cloning_epochs=2,
        behavior_cloning_critic_warm_start_steps=8,
        behavior_cloning_joint_warm_start_steps=4,
        behavior_cloning_critic_warm_start_learning_rate=2e-2,
        behavior_cloning_joint_warm_start_actor_lr_scale=0.1,
    )


def test_configured_critic_warm_start_binds_train_scope_and_artifact(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "trade_rl.integrations.universal_critic_warm_start"
    )
    run_configured = getattr(module, "run_configured_critic_warm_start")
    dataset = _teacher_dataset()
    split = _split()
    config = _config()

    configured = run_configured(
        policy=_Policy(),
        teacher_environment=_RewardEnvironment(),
        teacher_dataset=dataset,
        episode_batch=_episode_batch(),
        split=split,
        config=config,
        observation_provider=None,
        behavior_cloning_seed=7,
        output_root=tmp_path,
    )

    assert configured.warm_start.sample_count == 2
    assert configured.warm_start.actor_max_abs_drift_critic_only == 0.0
    assert configured.artifact_path == tmp_path / "critic-warm-start.json"
    payload = json.loads(configured.artifact_path.read_text(encoding="utf-8"))
    assert payload["artifact_digest"] == configured.artifact_digest
    assert payload["teacher_dataset_digest"] == dataset.artifact_digest
    assert payload["episode_batch_digest"] == _episode_batch().digest
    assert payload["train_sample_count"] == 2
    assert payload["validation_sample_count"] == 2
    assert payload["purged_sample_count"] == 0
    assert payload["training_config_digest"] == content_digest(config.digest_payload())
    assert payload["train_indices_digest"] == content_digest({"indices": (0, 1)})
