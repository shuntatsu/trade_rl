from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.rl.rollout_memory import estimate_ppo_rollout_buffer_bytes
from trade_rl.rl.training import ResidualTrainingConfig


class _CostTrainingProbe(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}
    environment_digest = "e" * 64
    initial_capital = 1_000.0
    decision_hours = 0.25
    action_names = ("tilt",)
    action_spec_digest = content_digest({"names": action_names})
    alpha_artifact_digest = None
    factor_artifact_digest = None
    normalizer = None

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
        self.close_calls = 0

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

    def close(self) -> None:
        self.close_calls += 1


class _FakeParameter:
    def numel(self) -> int:
        return 2


class _FakePolicy:
    action_distribution_name = "squashed_diag_gaussian"

    def parameters(self) -> tuple[_FakeParameter, ...]:
        return (_FakeParameter(),)


class _FakeCostCriticPPO:
    device = "cpu"

    def __init__(self, policy: object, environment: object, **kwargs: object) -> None:
        self.policy = _FakePolicy()
        self.num_timesteps = 0
        self.policy_identifier = policy
        self.environment = environment
        self.kwargs = kwargs

    def learn(self, **kwargs: object) -> None:
        self.num_timesteps = int(kwargs["total_timesteps"])

    def save(self, target: str) -> None:
        Path(f"{target}.zip").write_bytes(b"cost-policy")


class _ForbiddenPPO:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("ordinary PPO must not construct cost_critic_ppo")


def _config(**overrides: object) -> ResidualTrainingConfig:
    values: dict[str, object] = {
        "timesteps": 4,
        "gamma": 1.0,
        "seeds": (7,),
        "algorithm": "cost_critic_ppo",
        "n_steps": 4,
        "n_envs": 1,
        "batch_size": 4,
        "n_epochs": 1,
        "asset_set_encoder": False,
        "device": "cpu",
        "cost_learning_rate": 7e-4,
        "cost_n_epochs": 2,
        "cost_batch_size": 2,
        "cost_continuous_hidden_dims": (32, 16),
        "cost_event_hidden_dims": (24, 12),
        "cost_max_grad_norm": 0.75,
        "cost_continuous_gae_lambda": 0.97,
        "cost_event_gae_lambda": 1.0,
        "cost_value_loss_coefficient": 0.5,
        "cost_auxiliary_event_loss_coefficient": 0.25,
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)  # type: ignore[arg-type]


def test_backend_constructs_cost_critic_ppo_and_accounts_for_sidecar_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trade_rl.integrations.cost_rollout_buffer import (
        estimate_cost_rollout_storage_bytes,
    )

    probe = _CostTrainingProbe()
    constructed: list[_FakeCostCriticPPO] = []

    def build_cost_model(
        policy: object,
        environment: object,
        **kwargs: object,
    ) -> _FakeCostCriticPPO:
        model = _FakeCostCriticPPO(policy, environment, **kwargs)
        constructed.append(model)
        return model

    monkeypatch.setattr("stable_baselines3.PPO", _ForbiddenPPO)
    monkeypatch.setattr(
        "trade_rl.integrations.cost_critic_ppo.CostCriticPPO",
        build_cost_model,
    )
    monkeypatch.setattr(
        "trade_rl.rl.checkpointing.build_checkpoint_callback",
        lambda **kwargs: object(),
    )

    config = _config()
    result = StableBaselines3Backend(lambda: probe).train(
        seed=7,
        config=config,
        output_path=tmp_path / "policy.zip",
    )

    assert len(constructed) == 1
    model = constructed[0]
    assert model.kwargs["cost_learning_rate"] == pytest.approx(7e-4)
    assert model.kwargs["cost_n_epochs"] == 2
    assert model.kwargs["cost_batch_size"] == 2
    assert model.kwargs["cost_continuous_hidden_dims"] == (32, 16)
    assert model.kwargs["cost_event_hidden_dims"] == (24, 12)
    assert model.kwargs["cost_max_grad_norm"] == pytest.approx(0.75)
    schema = model.kwargs["cost_schema"]
    assert schema["daily_turnover"].gae_lambda == pytest.approx(0.97)
    assert schema["forced_liquidation_event"].gae_lambda == pytest.approx(1.0)
    assert schema["daily_turnover"].value_loss_coefficient == pytest.approx(0.5)
    assert schema[
        "forced_liquidation_event"
    ].auxiliary_event_loss_coefficient == pytest.approx(0.25)

    ordinary_bytes = estimate_ppo_rollout_buffer_bytes(
        probe.observation_space,
        n_steps=4,
        n_envs=1,
        action_dim=1,
    )
    cost_bytes = estimate_cost_rollout_storage_bytes(4, 1, len(schema.names))
    assert result.rollout_buffer_bytes == ordinary_bytes + cost_bytes
    assert result.actual_timesteps == 4
    assert probe.close_calls == 1


def test_backend_rejects_cost_sidecar_when_combined_memory_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe = _CostTrainingProbe()
    monkeypatch.setattr("stable_baselines3.PPO", _ForbiddenPPO)

    ordinary_bytes = estimate_ppo_rollout_buffer_bytes(
        probe.observation_space,
        n_steps=4,
        n_envs=1,
        action_dim=1,
    )
    with pytest.raises(ValueError, match="rollout buffer exceeds"):
        StableBaselines3Backend(lambda: probe).train(
            seed=7,
            config=_config(max_rollout_buffer_bytes=ordinary_bytes),
            output_path=tmp_path / "policy.zip",
        )
