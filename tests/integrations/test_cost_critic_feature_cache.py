from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv

from trade_rl.integrations.cost_critic_ppo import CostCriticPPO
from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.environment_constraints import ConstraintCostVector


class _CostEnvironment(gym.Env[np.ndarray, np.ndarray]):
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
        event = float(self.step_index == 2)
        observation = np.full(3, self.step_index / 10.0, dtype=np.float32)
        return (
            observation,
            0.1,
            False,
            False,
            {
                "constraint_costs": ConstraintCostVector(
                    drawdown_excess=0.01 * self.step_index,
                    drawdown_stop_event=event,
                    margin_deficit_fraction=0.0,
                    forced_liquidation_event=event,
                    gross_exposure_request_excess=0.0,
                    daily_turnover=0.2,
                    execution_cost_fraction=0.001,
                    funding_credit_fraction=0.0,
                )
            },
        )


def _model(
    environment: Any,
    *,
    cost_n_epochs: int = 3,
    cost_batch_size: int = 2,
) -> CostCriticPPO:
    return CostCriticPPO(
        "MlpPolicy",
        environment,
        seed=17,
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
        cost_n_epochs=cost_n_epochs,
        cost_batch_size=cost_batch_size,
        cost_continuous_hidden_dims=(12,),
        cost_event_hidden_dims=(10,),
    )


def test_cost_critic_update_extracts_policy_features_exactly_six_times() -> None:
    environment = DummyVecEnv([lambda: _CostEnvironment()])
    model = _model(environment, cost_n_epochs=3, cost_batch_size=2)
    original = model.policy.extract_features
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.policy.extract_features = counted  # type: ignore[method-assign]
    try:
        model.learn(total_timesteps=4)

        # Four rollout forwards, one PPO minibatch, and one post-PPO cache build.
        # SB3 predict_values() calls BaseModel.extract_features() directly and is
        # intentionally not visible through policy.extract_features wrapping.
        assert calls == 6
        assert model.cost_update_count == 6
    finally:
        environment.close()


def test_policy_feature_capture_returns_exact_detached_tensor_and_restores_binding() -> (
    None
):
    environment = DummyVecEnv([lambda: _CostEnvironment()])
    model = _model(environment)
    raw_observation = np.asarray([[0.1, -0.2, 0.3]], dtype=np.float32)
    observation = obs_as_tensor(raw_observation, model.device)
    original = model.policy.extract_features

    def locally_bound(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs)

    model.policy.extract_features = locally_bound  # type: ignore[method-assign]
    prior_binding = model.policy.extract_features
    model.policy.set_training_mode(False)
    try:
        with torch.no_grad():
            policy_output, captured = model._run_policy_with_cost_features(
                lambda: model.policy(observation)
            )
            fresh = model._cost_features(observation)

        assert len(policy_output) == 3
        assert model.policy.extract_features is prior_binding
        assert captured.requires_grad is False
        assert captured.grad_fn is None
        torch.testing.assert_close(captured, fresh, rtol=0.0, atol=0.0)
    finally:
        environment.close()


def test_value_bootstrap_reuses_exact_sb3_value_features() -> None:
    environment = DummyVecEnv([lambda: _CostEnvironment()])
    model = _model(environment)
    observation = obs_as_tensor(
        np.asarray([[0.1, -0.2, 0.3]], dtype=np.float32),
        model.device,
    )
    model.policy.set_training_mode(False)
    try:
        with torch.no_grad():
            values, captured = model._predict_values_with_cost_features(observation)
            expected_values = model.policy.predict_values(observation)
            expected_features = model._cost_features(observation)

        assert captured.requires_grad is False
        assert captured.grad_fn is None
        torch.testing.assert_close(values, expected_values, rtol=0.0, atol=0.0)
        torch.testing.assert_close(captured, expected_features, rtol=0.0, atol=0.0)
    finally:
        environment.close()


def test_policy_feature_capture_restores_binding_after_operation_failure() -> None:
    environment = DummyVecEnv([lambda: _CostEnvironment()])
    model = _model(environment)
    observation = obs_as_tensor(
        np.asarray([[0.1, -0.2, 0.3]], dtype=np.float32),
        model.device,
    )
    original = model.policy.extract_features

    def locally_bound(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs)

    model.policy.extract_features = locally_bound  # type: ignore[method-assign]
    prior_binding = model.policy.extract_features

    def fail_after_extraction() -> None:
        model.policy.extract_features(observation)
        raise RuntimeError("expected failure")

    try:
        with pytest.raises(RuntimeError, match="expected failure"):
            model._run_policy_with_cost_features(fail_after_extraction)

        assert model.policy.extract_features is prior_binding
    finally:
        environment.close()


def test_full_rollout_cache_matches_fresh_features_with_zero_tolerance() -> None:
    environment = DummyVecEnv([lambda: _CostEnvironment()])
    model = _model(environment, cost_n_epochs=1, cost_batch_size=4)
    try:
        model.learn(total_timesteps=4)
        original = model.policy.extract_features
        calls = 0

        def counted(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        model.policy.extract_features = counted  # type: ignore[method-assign]
        cache = model._build_cost_feature_cache()

        assert calls == 1
        assert cache.requires_grad is False
        assert cache.grad_fn is None
        assert cache.device == model.device
        assert cache.shape[0] == model.n_steps * model.n_envs
        assert bool(torch.isfinite(cache).all())

        indices = np.asarray((0, 3), dtype=np.int64)
        cached = model._cached_cost_features(cache, indices)
        training = model.policy.training
        model.policy.set_training_mode(False)
        try:
            with torch.no_grad():
                fresh = model._cost_features(model._rollout_observations(indices))
        finally:
            model.policy.set_training_mode(training)

        torch.testing.assert_close(cached, fresh, rtol=0.0, atol=0.0)
        assert calls == 2
    finally:
        environment.close()


def test_implicit_cuda_device_resolves_to_current_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert CostCriticPPO._resolved_device(torch.device("cuda")) == torch.device(
        "cuda:0"
    )
    assert CostCriticPPO._resolved_device(torch.device("cuda:1")) == torch.device(
        "cuda:1"
    )
    assert CostCriticPPO._resolved_device(torch.device("cpu")) == torch.device("cpu")


def test_cost_diagnostics_reuse_supplied_cache_without_feature_extraction() -> None:
    environment = DummyVecEnv([lambda: _CostEnvironment()])
    model = _model(environment, cost_n_epochs=1, cost_batch_size=4)
    try:
        model.learn(total_timesteps=4)
        cache = model._build_cost_feature_cache()
        original = model.policy.extract_features
        calls = 0

        def counted(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        model.policy.extract_features = counted  # type: ignore[method-assign]
        reports, family, metrics = model._build_cost_training_diagnostics(cache)

        assert calls == 0
        assert tuple(reports) == model.cost_schema.names
        assert family.continuous_gradient_norm >= 0.0
        assert np.isfinite(metrics["gradient/continuous"])
    finally:
        environment.close()


def test_cache_build_failure_restores_training_modes_and_torch_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = DummyVecEnv([lambda: _CostEnvironment()])
    model = _model(environment, cost_n_epochs=1, cost_batch_size=4)
    try:
        model.learn(total_timesteps=4)
        model.policy.set_training_mode(True)
        model.cost_critic.train(False)
        expected_rng = torch.random.get_rng_state().clone()

        def fail_cache_build() -> torch.Tensor:
            torch.rand(1)
            raise RuntimeError("cache build failed")

        monkeypatch.setattr(model, "_build_cost_feature_cache", fail_cache_build)
        with pytest.raises(RuntimeError, match="cache build failed"):
            model._train_cost_critic()

        assert model.policy.training is True
        assert model.cost_critic.training is False
        assert torch.equal(torch.random.get_rng_state(), expected_rng)
    finally:
        environment.close()
