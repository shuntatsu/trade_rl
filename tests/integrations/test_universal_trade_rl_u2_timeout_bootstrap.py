from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class _OneStepTimeoutEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.raw_rewards: list[float] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        return np.asarray([0.0], dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        reward = 1.25
        self.raw_rewards.append(reward)
        return (
            np.asarray([2.0], dtype=np.float32),
            reward,
            False,
            True,
            {},
        )


class _Callback(BaseCallback):
    def _on_step(self) -> bool:
        return True


def test_sb3_timeout_bootstrap_is_applied_exactly_once_to_training_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _OneStepTimeoutEnvironment()
    vector = DummyVecEnv([lambda: environment])
    model = PPO(
        "MlpPolicy",
        vector,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=0.9,
        seed=0,
        device="cpu",
    )
    callback = _Callback()
    _total_timesteps, callback = model._setup_learn(
        total_timesteps=2,
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name="u2-timeout",
        progress_bar=False,
    )

    value_observations: list[np.ndarray] = []

    def predict_values(observations: torch.Tensor) -> torch.Tensor:
        value_observations.append(observations.detach().cpu().numpy().copy())
        return torch.full(
            (observations.shape[0], 1),
            2.0,
            dtype=torch.float32,
            device=observations.device,
        )

    monkeypatch.setattr(model.policy, "predict_values", predict_values)

    assert model.env is not None
    assert model.collect_rollouts(
        model.env,
        callback,
        model.rollout_buffer,
        n_rollout_steps=2,
    )

    assert environment.raw_rewards == pytest.approx([1.25, 1.25])
    expected_training_reward = 1.25 + 0.9 * 2.0
    np.testing.assert_allclose(
        model.rollout_buffer.rewards[:, 0],
        np.asarray([expected_training_reward, expected_training_reward]),
        rtol=0.0,
        atol=1e-7,
    )
    assert not np.any(
        np.isclose(
            model.rollout_buffer.rewards[:, 0],
            1.25 + 2.0 * 0.9 * 2.0,
            rtol=0.0,
            atol=1e-7,
        )
    )

    terminal_value_calls = sum(
        np.array_equal(observation, np.asarray([[2.0]], dtype=np.float32))
        for observation in value_observations
    )
    assert terminal_value_calls == 2
