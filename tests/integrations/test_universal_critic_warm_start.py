from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.universal_critic_warm_start import (
    collect_episode_return_targets,
    warm_start_policy_actor_critic,
)
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset
from trade_rl.learning.universal_bc import CriticWarmStartPlan


def _digest(label: str) -> str:
    return content_digest({"label": label})


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
        return np.asarray([float(start)], dtype=np.float32), {"start_index": start}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        reward = self._rewards[self.current_index]
        self.current_index += 1
        self._remaining -= 1
        terminated = self._remaining == 0
        return (
            np.asarray([float(self.current_index)], dtype=np.float32),
            reward,
            terminated,
            False,
            {},
        )


def _oracle_batch() -> EpisodeOracleBatch:
    dataset_id = _digest("dataset")
    contracts = (
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
    )
    targets = (
        np.asarray([[0.1], [0.2]], dtype=np.float32),
        np.asarray([[0.3], [0.4]], dtype=np.float32),
    )
    return EpisodeOracleBatch(
        dataset_id=dataset_id,
        teacher_config_digest=_digest("teacher"),
        sampling_config_digest=_digest("sampling"),
        contracts=contracts,
        targets=targets,
    )


def test_collect_episode_return_targets_resets_at_each_episode_boundary() -> None:
    returns = collect_episode_return_targets(
        _RewardEnvironment(),
        _oracle_batch(),
        gamma=1.0,
    )

    np.testing.assert_allclose(returns, np.asarray([3.0, 2.0, 7.0, 4.0]))


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


def _supervised_dataset() -> SupervisedPolicyDataset:
    observations = np.asarray([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
    actions = np.asarray([[-0.8], [-0.4], [0.4], [0.8]], dtype=np.float32)
    return SupervisedPolicyDataset(
        observations=observations,
        actions=actions,
        dataset_id=_digest("dataset"),
        train_start=0,
        train_stop=5,
        environment_digest=_digest("environment"),
        action_spec_digest=_digest("action"),
        teacher_config_digest=_digest("teacher"),
    )


def test_critic_only_phase_preserves_actor_then_joint_phase_improves_value_fit() -> None:
    policy = _Policy()
    dataset = _supervised_dataset()
    result = warm_start_policy_actor_critic(
        policy,
        dataset,
        np.asarray([0.9, 0.5, 0.5, 0.9], dtype=np.float32),
        plan=CriticWarmStartPlan(
            critic_only_steps=80,
            joint_fine_tune_steps=40,
            joint_actor_learning_rate_scale=0.1,
        ),
        batch_size=4,
        learning_rate=2e-2,
        seed=7,
    )

    assert result.actor_max_abs_drift_critic_only == pytest.approx(0.0, abs=1e-12)
    assert result.critic_only_value_mse < result.initial_value_mse
    assert result.final_value_mse <= result.critic_only_value_mse * 1.2
    assert result.actor_max_abs_drift_joint > 0.0


def test_critic_warm_start_rejects_target_count_mismatch() -> None:
    with pytest.raises(ValueError, match="critic target count"):
        warm_start_policy_actor_critic(
            _Policy(),
            _supervised_dataset(),
            np.asarray([1.0, 2.0], dtype=np.float32),
            plan=CriticWarmStartPlan(critic_only_steps=1, joint_fine_tune_steps=1),
            batch_size=2,
            learning_rate=1e-3,
            seed=0,
        )
