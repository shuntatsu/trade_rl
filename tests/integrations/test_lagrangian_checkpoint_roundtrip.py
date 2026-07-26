from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from trade_rl.integrations.lagrangian_ppo import LagrangianPPO
from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.lagrangian import ConstraintEstimate, canonical_lagrangian_schema


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


def _schema(*, budget_shift: float = 0.0, cap_shift: float = 0.0):
    count = len(CONSTRAINT_COST_NAMES)
    budgets = [0.2] * count
    budgets[0] += budget_shift
    caps = [5.0] * count
    caps[0] += cap_shift
    return canonical_lagrangian_schema(
        names=CONSTRAINT_COST_NAMES,
        budgets=tuple(budgets),
        dual_learning_rates=(0.25,) * count,
        ema_betas=(0.5,) * count,
        initial_multipliers=(0.1,) * count,
        max_multipliers=tuple(caps),
        warmup_rollouts=(0,) * count,
        update_interval_rollouts=(1,) * count,
    )


def _model(environment: Any, *, schema: Any | None = None) -> LagrangianPPO:
    return LagrangianPPO(
        "MlpPolicy",
        environment,
        seed=71,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        cost_schema=canonical_cost_learning_schema(),
        cost_learning_rate=5e-4,
        cost_n_epochs=1,
        cost_batch_size=4,
        cost_continuous_hidden_dims=(12,),
        cost_event_hidden_dims=(10,),
        lagrangian_schema=_schema() if schema is None else schema,
    )


def _estimates(scale: float) -> dict[str, ConstraintEstimate]:
    return {
        name: ConstraintEstimate(
            name=name,
            numerator=scale * (index + 1),
            denominator=2,
        )
        for index, name in enumerate(CONSTRAINT_COST_NAMES)
    }


def _assert_nested_state_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        return
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
        return
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_state_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_state_equal(left_value, right_value)
        return
    assert left == right


def test_lagrangian_checkpoint_identity_includes_schema_contract() -> None:
    environment = DummyVecEnv([_CheckpointEnvironment])
    model = _model(environment)
    try:
        identity = model.checkpoint_identity_payload()

        assert identity["algorithm"] == "lagrangian_ppo"
        assert identity["lagrangian_schema_digest"] == model.lagrangian_schema.digest
        assert identity["lagrangian_cost_names"] == list(CONSTRAINT_COST_NAMES)
    finally:
        environment.close()


def test_lagrangian_identity_changes_with_budget_or_cap() -> None:
    baseline_environment = DummyVecEnv([_CheckpointEnvironment])
    budget_environment = DummyVecEnv([_CheckpointEnvironment])
    cap_environment = DummyVecEnv([_CheckpointEnvironment])
    baseline = _model(baseline_environment)
    changed_budget = _model(budget_environment, schema=_schema(budget_shift=0.01))
    changed_cap = _model(cap_environment, schema=_schema(cap_shift=1.0))
    try:
        baseline_identity = baseline.checkpoint_identity_payload()
        budget_identity = changed_budget.checkpoint_identity_payload()
        cap_identity = changed_cap.checkpoint_identity_payload()

        assert baseline_identity != budget_identity
        assert baseline_identity != cap_identity
        assert (
            baseline_identity["lagrangian_schema_digest"]
            != budget_identity["lagrangian_schema_digest"]
        )
        assert (
            baseline_identity["lagrangian_schema_digest"]
            != cap_identity["lagrangian_schema_digest"]
        )
    finally:
        baseline_environment.close()
        budget_environment.close()
        cap_environment.close()


def test_lagrangian_save_load_preserves_dual_accumulator_and_critic_state(
    tmp_path: Path,
) -> None:
    environment = DummyVecEnv([_CheckpointEnvironment])
    loaded_environment = DummyVecEnv([_CheckpointEnvironment])
    model = _model(environment)
    try:
        model.lagrangian_controller.update_after_rollout(_estimates(0.3))
        model.lagrangian_controller.update_after_rollout(_estimates(0.1))
        model.frozen_lagrange_multipliers = model.lagrangian_controller.begin_rollout()

        partial_costs = np.zeros((2, 1, len(CONSTRAINT_COST_NAMES)), dtype=np.float64)
        partial_costs[0, 0, 0] = 0.2
        partial_costs[1, 0, 0] = 0.4
        partial_costs[:, 0, 4:] = 0.1
        accumulator = model.completed_episode_cost_accumulator
        assert accumulator is not None
        accumulator.ingest_rollout(
            costs=partial_costs,
            terminated=np.zeros((2, 1), dtype=np.bool_),
            truncated=np.zeros((2, 1), dtype=np.bool_),
        )

        model.cost_critic_optimizer.zero_grad()
        critic_loss = sum(
            parameter.square().sum() for parameter in model.cost_critic.parameters()
        )
        critic_loss.backward()
        model.cost_critic_optimizer.step()

        expected_identity = model.checkpoint_identity_payload()
        expected_dual_state = model.lagrangian_controller.state_dict()
        expected_accumulator_state = accumulator.state_dict()
        expected_frozen = model.frozen_lagrange_multipliers.copy()
        expected_critic_state = {
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
        assert loaded.lagrangian_controller.state_dict() == expected_dual_state
        loaded_accumulator = loaded.completed_episode_cost_accumulator
        assert loaded_accumulator is not None
        assert loaded_accumulator.state_dict() == expected_accumulator_state
        np.testing.assert_array_equal(
            loaded.frozen_lagrange_multipliers,
            expected_frozen,
        )
        assert loaded.frozen_lagrange_multipliers.flags.writeable is False
        for name, expected in expected_critic_state.items():
            torch.testing.assert_close(
                loaded.cost_critic.state_dict()[name],
                expected,
                rtol=0.0,
                atol=0.0,
            )
        _assert_nested_state_equal(
            loaded.cost_critic_optimizer.state_dict(),
            expected_optimizer_state,
        )

        next_estimates = _estimates(0.4)
        assert model.lagrangian_controller.update_after_rollout(next_estimates) == (
            loaded.lagrangian_controller.update_after_rollout(next_estimates)
        )
    finally:
        environment.close()
        loaded_environment.close()
