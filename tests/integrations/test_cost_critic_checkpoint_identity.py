from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from trade_rl.rl.cost_learning import canonical_cost_learning_schema


class _IdentityEnvironment(gym.Env[np.ndarray, np.ndarray]):
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
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        return np.zeros(3, dtype=np.float32), 0.0, False, False, {}


def _model(
    *,
    event_gae_lambda: float = 0.95,
    event_hidden_dims: tuple[int, ...] = (10,),
):
    from trade_rl.integrations.cost_critic_ppo import CostCriticPPO

    environment = DummyVecEnv([_IdentityEnvironment])
    model = CostCriticPPO(
        "MlpPolicy",
        environment,
        seed=17,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        cost_schema=canonical_cost_learning_schema(
            continuous_gae_lambda=0.97,
            event_gae_lambda=event_gae_lambda,
            auxiliary_event_loss_coefficient=0.25,
        ),
        cost_continuous_hidden_dims=(12,),
        cost_event_hidden_dims=event_hidden_dims,
    )
    return model, environment


def test_cost_critic_ppo_exposes_complete_checkpoint_identity() -> None:
    model, environment = _model()
    try:
        identity = model.checkpoint_identity_payload()

        assert identity["algorithm"] == "cost_critic_ppo"
        assert identity["cost_names"] == list(model.cost_schema.names)
        assert identity["cost_schema_digest"] == model.cost_schema.digest
        assert identity["architecture_digest"] == model.cost_critic.architecture_digest
        assert isinstance(identity["rollout_schema_digest"], str)
        assert len(identity["rollout_schema_digest"]) == 64
    finally:
        environment.close()


def test_cost_checkpoint_identity_changes_with_lambda_or_architecture() -> None:
    baseline, baseline_environment = _model()
    changed_lambda, lambda_environment = _model(event_gae_lambda=1.0)
    changed_width, width_environment = _model(event_hidden_dims=(16,))
    try:
        baseline_identity = baseline.checkpoint_identity_payload()
        lambda_identity = changed_lambda.checkpoint_identity_payload()
        width_identity = changed_width.checkpoint_identity_payload()

        assert baseline_identity != lambda_identity
        assert (
            baseline_identity["cost_schema_digest"]
            != lambda_identity["cost_schema_digest"]
        )
        assert (
            baseline_identity["rollout_schema_digest"]
            != lambda_identity["rollout_schema_digest"]
        )
        assert baseline_identity != width_identity
        assert (
            baseline_identity["architecture_digest"]
            != width_identity["architecture_digest"]
        )
    finally:
        baseline_environment.close()
        lambda_environment.close()
        width_environment.close()
