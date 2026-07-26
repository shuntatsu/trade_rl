from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from trade_rl.integrations.lagrangian_ppo import LagrangianPPO
from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.lagrangian import canonical_lagrangian_schema


class _ActorEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        return np.zeros(3, dtype=np.float32), 0.0, False, False, {}


def _schema():
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


def _model(environment: DummyVecEnv) -> LagrangianPPO:
    return LagrangianPPO(
        "MlpPolicy",
        environment,
        seed=83,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        normalize_advantage=True,
        cost_schema=canonical_cost_learning_schema(),
        cost_continuous_hidden_dims=(12,),
        cost_event_hidden_dims=(10,),
        lagrangian_schema=_schema(),
    )


def _cost_matrix() -> np.ndarray:
    costs = np.zeros((4, len(CONSTRAINT_COST_NAMES)), dtype=np.float64)
    costs[:, 0] = np.asarray([0.1, 0.4, 0.2, 0.8])
    costs[:, 1] = np.asarray([3.0, 2.0, 1.0, 4.0])
    return costs


def test_actor_normalizes_only_the_final_raw_combined_advantage() -> None:
    environment = DummyVecEnv([_ActorEnvironment])
    model = _model(environment)
    try:
        multipliers = np.zeros(len(CONSTRAINT_COST_NAMES), dtype=np.float64)
        multipliers[:2] = np.asarray([2.0, 0.3])
        multipliers.flags.writeable = False
        model.frozen_lagrange_multipliers = multipliers
        reward = torch.tensor([0.4, -0.2, 1.3, 0.1], dtype=torch.float32)
        costs = _cost_matrix()

        actual = model._actor_advantages(
            reward_advantages=reward,
            cost_advantages=costs,
        )
        combined_numpy = reward.detach().cpu().numpy() - costs @ multipliers
        raw = torch.as_tensor(combined_numpy, dtype=reward.dtype)
        expected = (raw - raw.mean()) / (raw.std() + 1e-8)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    finally:
        environment.close()


def test_actor_update_is_invariant_to_cost_unit_conversion() -> None:
    environment = DummyVecEnv([_ActorEnvironment])
    model = _model(environment)
    try:
        reward = torch.tensor([0.4, -0.2, 1.3, 0.1], dtype=torch.float32)
        costs = _cost_matrix()
        multipliers = np.zeros(len(CONSTRAINT_COST_NAMES), dtype=np.float64)
        multipliers[:2] = np.asarray([2.0, 0.3])
        multipliers.flags.writeable = False
        model.frozen_lagrange_multipliers = multipliers
        baseline = model._actor_advantages(
            reward_advantages=reward,
            cost_advantages=costs,
        )

        converted_costs = costs.copy()
        converted_costs[:, 0] *= 10.0
        converted_multipliers = multipliers.copy()
        converted_multipliers[0] /= 10.0
        converted_multipliers.flags.writeable = False
        model.frozen_lagrange_multipliers = converted_multipliers
        converted = model._actor_advantages(
            reward_advantages=reward,
            cost_advantages=converted_costs,
        )

        torch.testing.assert_close(converted, baseline, rtol=0.0, atol=0.0)
    finally:
        environment.close()
