from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.lagrangian import (
    ConstraintEstimate,
    LagrangianSchema,
    canonical_lagrangian_schema,
)
from trade_rl.rl.lagrangian_episode import EpisodeCompletionKind


class _CheckpointEnvironment(gym.Env[np.ndarray, np.ndarray]):
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


def _schema(
    *, drawdown_budget: float = 0.1, drawdown_cap: float = 10.0
) -> LagrangianSchema:
    count = len(CONSTRAINT_COST_NAMES)
    return canonical_lagrangian_schema(
        names=CONSTRAINT_COST_NAMES,
        budgets=(drawdown_budget, *(1.0 for _ in range(count - 1))),
        dual_learning_rates=(0.5,) * count,
        ema_betas=(0.5,) * count,
        initial_multipliers=(0.25, *(0.0 for _ in range(count - 1))),
        max_multipliers=(drawdown_cap, *(10.0 for _ in range(count - 1))),
        warmup_rollouts=(0,) * count,
        update_interval_rollouts=(1,) * count,
    )


def _model(environment: object, *, schema: LagrangianSchema):
    from trade_rl.integrations.lagrangian_ppo import LagrangianPPO

    return LagrangianPPO(
        "MlpPolicy",
        environment,
        seed=17,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        cost_schema=canonical_cost_learning_schema(),
        cost_continuous_hidden_dims=(12,),
        cost_event_hidden_dims=(10,),
        lagrangian_schema=schema,
    )


def _estimates(value: float) -> dict[str, ConstraintEstimate]:
    return {
        name: ConstraintEstimate(name=name, numerator=value, denominator=1)
        for name in CONSTRAINT_COST_NAMES
    }


def _initialize_optimizer_state(model: object) -> None:
    critic = model.cost_critic
    optimizer = model.cost_critic_optimizer
    optimizer.zero_grad()
    output = critic(
        torch.ones(
            (2, critic.input_dim),
            dtype=torch.float32,
            device=model.device,
        )
    )
    output.values.square().mean().backward()
    optimizer.step()


def test_lagrangian_checkpoint_identity_binds_schema_budget_and_cap() -> None:
    environment = DummyVecEnv([_CheckpointEnvironment])
    changed_budget_environment = DummyVecEnv([_CheckpointEnvironment])
    changed_cap_environment = DummyVecEnv([_CheckpointEnvironment])
    baseline = _model(environment, schema=_schema())
    changed_budget = _model(
        changed_budget_environment,
        schema=_schema(drawdown_budget=0.2),
    )
    changed_cap = _model(
        changed_cap_environment,
        schema=_schema(drawdown_cap=20.0),
    )
    try:
        identity = baseline.checkpoint_identity_payload()

        assert identity["algorithm"] == "lagrangian_ppo"
        assert identity["lagrangian_cost_names"] == list(CONSTRAINT_COST_NAMES)
        assert identity["lagrangian_schema_digest"] == baseline.lagrangian_schema.digest
        assert identity != changed_budget.checkpoint_identity_payload()
        assert identity != changed_cap.checkpoint_identity_payload()
    finally:
        environment.close()
        changed_budget_environment.close()
        changed_cap_environment.close()


def test_lagrangian_save_load_round_trip_preserves_dual_and_partial_episode_state(
    tmp_path: Path,
) -> None:
    from trade_rl.integrations.lagrangian_ppo import LagrangianPPO

    environment = DummyVecEnv([_CheckpointEnvironment])
    loaded_environment = DummyVecEnv([_CheckpointEnvironment])
    model = _model(environment, schema=_schema())
    try:
        model.lagrangian_controller.update_after_rollout(_estimates(0.5))
        model.lagrangian_controller.update_after_rollout(_estimates(0.3))
        model.frozen_lagrange_multipliers = model.lagrangian_controller.begin_rollout()
        partial_costs = np.zeros((1, 1, len(CONSTRAINT_COST_NAMES)), dtype=np.float64)
        partial_costs[0, 0, 0] = 0.2
        model.completed_episode_cost_accumulator.ingest_rollout(
            costs=partial_costs,
            elapsed_hours=np.asarray([[6.0]], dtype=np.float64),
            completion_kinds=np.asarray([[EpisodeCompletionKind.NONE]], dtype=np.int8),
        )
        _initialize_optimizer_state(model)

        expected_identity = model.checkpoint_identity_payload()
        expected_controller = model.lagrangian_controller.state_dict()
        expected_accumulator = model.completed_episode_cost_accumulator.state_dict()
        expected_frozen = model.frozen_lagrange_multipliers.copy()
        expected_cost_state = {
            name: value.detach().clone()
            for name, value in model.cost_critic.state_dict().items()
        }
        expected_optimizer_state = model.cost_critic_optimizer.state_dict()
        target = tmp_path / "lagrangian-policy"
        model.save(str(target))

        loaded = LagrangianPPO.load(
            str(target),
            env=loaded_environment,
            device="cpu",
        )

        assert loaded.checkpoint_identity_payload() == expected_identity
        assert loaded.lagrangian_controller.state_dict() == expected_controller
        assert (
            loaded.completed_episode_cost_accumulator.state_dict()
            == expected_accumulator
        )
        np.testing.assert_array_equal(
            loaded.frozen_lagrange_multipliers,
            expected_frozen,
        )
        assert loaded.cost_critic.state_dict().keys() == expected_cost_state.keys()
        for name, expected in expected_cost_state.items():
            torch.testing.assert_close(loaded.cost_critic.state_dict()[name], expected)
        assert (
            loaded.cost_critic_optimizer.state_dict()["param_groups"]
            == expected_optimizer_state["param_groups"]
        )
        assert loaded.cost_critic_optimizer.state_dict()["state"].keys() == (
            expected_optimizer_state["state"].keys()
        )

        next_estimates = _estimates(0.4)
        expected_next = model.lagrangian_controller.update_after_rollout(next_estimates)
        loaded_next = loaded.lagrangian_controller.update_after_rollout(next_estimates)
        assert loaded_next == expected_next
    finally:
        environment.close()
        loaded_environment.close()
