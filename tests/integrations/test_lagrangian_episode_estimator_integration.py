from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from trade_rl.integrations.cost_critic_ppo import CostCriticPPO
from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.environment_constraints import (
    CONSTRAINT_COST_NAMES,
    ConstraintCostVector,
)
from trade_rl.rl.lagrangian import LagrangianSchema, canonical_lagrangian_schema
from trade_rl.rl.lagrangian_episode_estimator import (
    TimeAwareCompletedEpisodeCostAccumulator,
)


class _ElapsedEpisodeEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, *, terminate_after: int | None = 4) -> None:
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
        self.terminate_after = terminate_after
        self.step_index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        self.step_index = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        self.step_index += 1
        terminated = bool(
            self.terminate_after is not None and self.step_index >= self.terminate_after
        )
        costs = ConstraintCostVector(
            drawdown_excess=0.1,
            drawdown_stop_event=0.0,
            margin_deficit_fraction=0.0,
            forced_liquidation_event=0.0,
            gross_exposure_request_excess=0.0,
            daily_turnover=0.0,
            execution_cost_fraction=0.0,
            funding_credit_fraction=0.0,
        )
        info: dict[str, object] = {
            "constraint_costs": costs,
            "transition_elapsed_hours": 6.0,
            "termination_reason": "drawdown_stop" if terminated else None,
        }
        observation = np.asarray(
            [self.step_index / 10.0, 0.5, -0.25],
            dtype=np.float32,
        )
        return observation, 0.1, terminated, False, info


def _lagrangian_schema() -> LagrangianSchema:
    count = len(CONSTRAINT_COST_NAMES)
    return canonical_lagrangian_schema(
        names=CONSTRAINT_COST_NAMES,
        budgets=(0.0,) * count,
        dual_learning_rates=(0.1,) * count,
        ema_betas=(0.0,) * count,
        initial_multipliers=(0.0,) * count,
        max_multipliers=(10.0,) * count,
        warmup_rollouts=(0,) * count,
        update_interval_rollouts=(1,) * count,
    )


def _common_kwargs() -> dict[str, object]:
    return {
        "seed": 41,
        "device": "cpu",
        "n_steps": 4,
        "batch_size": 2,
        "n_epochs": 1,
        "learning_rate": 3e-4,
        "cost_schema": canonical_cost_learning_schema(),
        "cost_learning_rate": 5e-4,
        "cost_n_epochs": 1,
        "cost_batch_size": 2,
        "cost_continuous_hidden_dims": (8,),
        "cost_event_hidden_dims": (8,),
    }


def _lagrangian_model(environment: Any) -> Any:
    from trade_rl.integrations.lagrangian_ppo import LagrangianPPO

    return LagrangianPPO(
        "MlpPolicy",
        environment,
        lagrangian_schema=_lagrangian_schema(),
        **_common_kwargs(),
    )


def test_episode_metadata_storage_is_lagrangian_only() -> None:
    ordinary_environment = DummyVecEnv([_ElapsedEpisodeEnvironment])
    lagrangian_environment = DummyVecEnv([_ElapsedEpisodeEnvironment])
    ordinary = CostCriticPPO(
        "MlpPolicy",
        ordinary_environment,
        **_common_kwargs(),
    )
    constrained = _lagrangian_model(lagrangian_environment)
    try:
        assert ordinary.cost_rollout_storage.store_episode_metadata is False
        assert ordinary.cost_rollout_storage.transition_elapsed_hours is None
        assert constrained.cost_rollout_storage.store_episode_metadata is True
        assert constrained.cost_rollout_storage.transition_elapsed_hours is not None
        assert isinstance(
            constrained.completed_episode_cost_accumulator,
            TimeAwareCompletedEpisodeCostAccumulator,
        )
    finally:
        ordinary_environment.close()
        lagrangian_environment.close()


def test_lagrangian_dual_estimate_uses_elapsed_time_episode_area() -> None:
    environment = DummyVecEnv([_ElapsedEpisodeEnvironment])
    model = _lagrangian_model(environment)
    try:
        model.learn(total_timesteps=4)

        estimate = model.last_constraint_estimates["drawdown_excess"]
        assert estimate is not None
        assert estimate.denominator == 1
        assert estimate.value == pytest.approx(0.1)
        assert model.cost_rollout_storage.transition_elapsed_hours is not None
        np.testing.assert_array_equal(
            model.cost_rollout_storage.transition_elapsed_hours[:, 0],
            np.asarray([6.0, 6.0, 6.0, 6.0]),
        )
        accumulator = model.completed_episode_cost_accumulator
        assert isinstance(accumulator, TimeAwareCompletedEpisodeCostAccumulator)
        assert accumulator.censored_episode_count == 0
    finally:
        environment.close()
