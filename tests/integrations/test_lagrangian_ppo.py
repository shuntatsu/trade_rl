from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from trade_rl.integrations.cost_critic_ppo import CostCriticPPO
from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.environment_constraints import (
    CONSTRAINT_COST_NAMES,
    ConstraintCostVector,
)
from trade_rl.rl.lagrangian import (
    LagrangianSchema,
    canonical_lagrangian_schema,
)


class _LagrangianEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        drawdown_cost: float = 0.1,
        terminate_after: int | None = 4,
    ) -> None:
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
        self.drawdown_cost = drawdown_cost
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
        observation = np.asarray(
            [self.step_index / 10.0, 0.5, -0.25],
            dtype=np.float32,
        )
        costs = ConstraintCostVector(
            drawdown_excess=self.drawdown_cost * self.step_index,
            drawdown_stop_event=0.0,
            margin_deficit_fraction=0.0,
            forced_liquidation_event=0.0,
            gross_exposure_request_excess=0.0,
            daily_turnover=0.0,
            execution_cost_fraction=0.0,
            funding_credit_fraction=0.0,
        )
        return observation, 0.1, terminated, False, {"constraint_costs": costs}


def _lagrangian_schema(
    *,
    initial_drawdown_multiplier: float = 0.0,
    drawdown_budget: float = 0.0,
) -> LagrangianSchema:
    cost_count = len(CONSTRAINT_COST_NAMES)
    return canonical_lagrangian_schema(
        names=CONSTRAINT_COST_NAMES,
        budgets=(drawdown_budget, *(100.0 for _ in range(cost_count - 1))),
        dual_learning_rates=(0.5,) * cost_count,
        ema_betas=(0.0,) * cost_count,
        initial_multipliers=(
            initial_drawdown_multiplier,
            *(0.0 for _ in range(cost_count - 1)),
        ),
        max_multipliers=(10.0,) * cost_count,
        warmup_rollouts=(0,) * cost_count,
        update_interval_rollouts=(1,) * cost_count,
    )


def _common_model_kwargs(*, seed: int = 29) -> dict[str, object]:
    return {
        "seed": seed,
        "device": "cpu",
        "n_steps": 4,
        "batch_size": 2,
        "n_epochs": 3,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "cost_schema": canonical_cost_learning_schema(),
        "cost_learning_rate": 5e-4,
        "cost_n_epochs": 1,
        "cost_batch_size": 2,
        "cost_continuous_hidden_dims": (12,),
        "cost_event_hidden_dims": (10,),
    }


def _lagrangian_model(
    environment: Any,
    *,
    schema: LagrangianSchema,
    seed: int = 29,
) -> Any:
    from trade_rl.integrations.lagrangian_ppo import LagrangianPPO

    return LagrangianPPO(
        "MlpPolicy",
        environment,
        lagrangian_schema=schema,
        **_common_model_kwargs(seed=seed),
    )


def _reset_training_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _assert_policy_state_equal(left: Any, right: Any) -> None:
    left_state = left.policy.state_dict()
    right_state = right.policy.state_dict()
    assert left_state.keys() == right_state.keys()
    for name in left_state:
        torch.testing.assert_close(
            left_state[name],
            right_state[name],
            rtol=0.0,
            atol=0.0,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )


def test_zero_multiplier_lagrangian_update_matches_cost_critic_ppo_exactly() -> None:
    ordinary_environment = DummyVecEnv([_LagrangianEnvironment])
    lagrangian_environment = DummyVecEnv([_LagrangianEnvironment])
    ordinary = CostCriticPPO(
        "MlpPolicy",
        ordinary_environment,
        **_common_model_kwargs(seed=29),
    )
    constrained = _lagrangian_model(
        lagrangian_environment,
        schema=_lagrangian_schema(),
        seed=29,
    )
    try:
        _reset_training_rng(29)
        ordinary.learn(total_timesteps=4)
        _reset_training_rng(29)
        constrained.learn(total_timesteps=4)

        _assert_policy_state_equal(ordinary, constrained)
        assert constrained.algorithm_identifier == "lagrangian_ppo"
        assert constrained.frozen_lagrange_multipliers.tolist() == pytest.approx(
            [0.0] * len(CONSTRAINT_COST_NAMES)
        )
    finally:
        ordinary_environment.close()
        lagrangian_environment.close()


def test_multiplier_snapshot_is_frozen_for_every_minibatch_and_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.integrations.lagrangian_ppo as lagrangian_module

    environment = DummyVecEnv([_LagrangianEnvironment])
    model = _lagrangian_model(
        environment,
        schema=_lagrangian_schema(initial_drawdown_multiplier=2.0),
    )
    observed: list[np.ndarray] = []
    original = lagrangian_module.combine_lagrangian_advantages

    def capture_multipliers(
        *,
        reward_advantages: object,
        cost_advantages: object,
        multipliers: object,
    ) -> np.ndarray:
        observed.append(np.asarray(multipliers, dtype=np.float64).copy())
        return original(
            reward_advantages=reward_advantages,
            cost_advantages=cost_advantages,
            multipliers=multipliers,
        )

    monkeypatch.setattr(
        lagrangian_module,
        "combine_lagrangian_advantages",
        capture_multipliers,
    )
    try:
        model.learn(total_timesteps=4)

        assert len(observed) == model.n_epochs * 2
        for snapshot in observed:
            np.testing.assert_array_equal(
                snapshot,
                np.asarray([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
        assert model.frozen_lagrange_multipliers[0] == pytest.approx(2.0)
        assert model.lagrangian_controller.begin_rollout()[0] > 2.0
    finally:
        environment.close()


def test_actor_then_cost_critic_then_dual_update_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = DummyVecEnv([_LagrangianEnvironment])
    model = _lagrangian_model(environment, schema=_lagrangian_schema())
    events: list[str] = []

    original_policy_step = model.policy.optimizer.step
    original_cost_train = model._train_cost_critic
    original_dual_update = model.lagrangian_controller.update_after_rollout

    def policy_step(*args: object, **kwargs: object) -> object:
        events.append("actor")
        return original_policy_step(*args, **kwargs)

    def cost_train() -> None:
        events.append("cost")
        original_cost_train()

    def dual_update(
        estimates: Mapping[str, object],
    ) -> object:
        events.append("dual")
        return original_dual_update(estimates)  # type: ignore[arg-type]

    monkeypatch.setattr(model.policy.optimizer, "step", policy_step)
    monkeypatch.setattr(model, "_train_cost_critic", cost_train)
    monkeypatch.setattr(
        model.lagrangian_controller,
        "update_after_rollout",
        dual_update,
    )
    try:
        model.learn(total_timesteps=4)

        assert events.count("dual") == 1
        assert events.count("cost") == 1
        assert events[-2:] == ["cost", "dual"]
        assert all(event == "actor" for event in events[:-2])
    finally:
        environment.close()


def test_rollout_without_completed_episode_skips_every_dual_update() -> None:
    environment = DummyVecEnv([lambda: _LagrangianEnvironment(terminate_after=None)])
    model = _lagrangian_model(
        environment,
        schema=_lagrangian_schema(initial_drawdown_multiplier=1.0),
    )
    try:
        model.learn(total_timesteps=4)

        assert model.lagrangian_controller.begin_rollout().tolist() == pytest.approx(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        assert tuple(model.last_dual_update_reports) == CONSTRAINT_COST_NAMES
        assert all(
            report.updated is False
            and report.skip_reason == "missing_estimate"
            and report.denominator is None
            for report in model.last_dual_update_reports.values()
        )
    finally:
        environment.close()


def test_safe_and_unsafe_completed_episodes_move_only_the_matching_multiplier() -> None:
    unsafe_environment = DummyVecEnv([_LagrangianEnvironment])
    safe_environment = DummyVecEnv([lambda: _LagrangianEnvironment(drawdown_cost=0.0)])
    unsafe = _lagrangian_model(
        unsafe_environment,
        schema=_lagrangian_schema(),
    )
    safe = _lagrangian_model(
        safe_environment,
        schema=_lagrangian_schema(
            initial_drawdown_multiplier=1.0,
            drawdown_budget=0.5,
        ),
    )
    try:
        unsafe.learn(total_timesteps=4)
        safe.learn(total_timesteps=4)

        unsafe_multipliers = unsafe.lagrangian_controller.begin_rollout()
        safe_multipliers = safe.lagrangian_controller.begin_rollout()
        assert unsafe_multipliers[0] > 0.0
        assert safe_multipliers[0] < 1.0
        np.testing.assert_array_equal(unsafe_multipliers[1:], np.zeros(6))
        np.testing.assert_array_equal(safe_multipliers[1:], np.zeros(6))
    finally:
        unsafe_environment.close()
        safe_environment.close()
