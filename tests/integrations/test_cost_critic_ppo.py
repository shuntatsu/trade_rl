from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.environment_constraints import ConstraintCostVector


class _CostEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, *, include_costs: bool = True, event_step: int = 2) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.include_costs = include_costs
        self.event_step = event_step
        self.step_index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        self.step_index = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        self.step_index += 1
        event = float(self.step_index == self.event_step)
        observation = np.full(3, self.step_index / 10.0, dtype=np.float32)
        info: dict[str, object] = {}
        if self.include_costs:
            info["constraint_costs"] = ConstraintCostVector(
                drawdown_excess=0.01 * self.step_index,
                drawdown_stop_event=event,
                margin_deficit_fraction=0.0,
                forced_liquidation_event=event,
                gross_exposure_request_excess=0.0,
                daily_turnover=0.2,
                execution_cost_fraction=0.001,
                funding_credit_fraction=0.0,
            )
        return observation, 0.1, False, False, info


def _model(environment: Any, *, seed: int = 13) -> Any:
    from trade_rl.integrations.cost_critic_ppo import CostCriticPPO

    return CostCriticPPO(
        "MlpPolicy",
        environment,
        seed=seed,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        cost_schema=canonical_cost_learning_schema(
            auxiliary_event_loss_coefficient=0.25
        ),
        cost_learning_rate=5e-4,
        cost_n_epochs=2,
        cost_batch_size=4,
        cost_continuous_hidden_dims=(12,),
        cost_event_hidden_dims=(10,),
    )


def test_cost_critic_ppo_collects_and_trains_without_actor_penalty() -> None:
    environment = DummyVecEnv([lambda: _CostEnvironment()])
    model = _model(environment)
    try:
        model.learn(total_timesteps=8)

        assert model.algorithm_identifier == "cost_critic_ppo"
        assert model.cost_rollout_storage.finalized is True
        assert model.cost_update_count == 4
        assert np.isfinite(model.last_cost_training_metrics["loss/total"])
        assert model.last_cost_training_metrics["support/drawdown_stop_event"] > 0.0
        assert model.last_cost_training_metrics["support/forced_liquidation_event"] > 0.0
        assert not hasattr(model, "lagrange_multipliers")
    finally:
        environment.close()


def test_cost_critic_ppo_fails_closed_when_constraint_info_is_missing() -> None:
    environment = DummyVecEnv([lambda: _CostEnvironment(include_costs=False)])
    model = _model(environment)
    try:
        with pytest.raises(ValueError, match="constraint_costs"):
            model.learn(total_timesteps=4)
    finally:
        environment.close()


def test_cost_critic_sidecar_does_not_change_reward_ppo_update() -> None:
    ordinary_environment = DummyVecEnv([lambda: _CostEnvironment()])
    cost_environment = DummyVecEnv([lambda: _CostEnvironment()])
    ordinary = PPO(
        "MlpPolicy",
        ordinary_environment,
        seed=29,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
    )
    cost_model = _model(cost_environment, seed=29)
    try:
        ordinary.learn(total_timesteps=4)
        cost_model.learn(total_timesteps=4)

        ordinary_state = ordinary.policy.state_dict()
        cost_state = cost_model.policy.state_dict()
        assert ordinary_state.keys() == cost_state.keys()
        for name in ordinary_state:
            torch.testing.assert_close(
                ordinary_state[name],
                cost_state[name],
                rtol=0.0,
                atol=0.0,
                msg=lambda message, parameter=name: f"{parameter}: {message}",
            )
    finally:
        ordinary_environment.close()
        cost_environment.close()


def test_cost_critic_ppo_counts_event_support_across_vector_environments() -> None:
    environment = DummyVecEnv(
        [
            lambda: _CostEnvironment(event_step=1),
            lambda: _CostEnvironment(event_step=3),
        ]
    )
    model = _model(environment)
    try:
        model.learn(total_timesteps=8)

        assert model.last_cost_training_metrics["support/drawdown_stop_event"] == 2.0
        assert model.last_cost_training_metrics["support/forced_liquidation_event"] == 2.0
    finally:
        environment.close()
