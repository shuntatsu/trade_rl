from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.datasets import DatasetManifest
from trade_rl.integrations.sb3_training import StableBaselines3PPOBackend
from trade_rl.rl.training import (
    PolicyTrainingResult,
    ResidualTrainingConfig,
    gamma_from_half_life,
    train_residual_ensemble,
)

ENVIRONMENT_DIGEST = "e" * 64
ACTION_NAMES = ("fast_tilt", "slow_tilt", "risk_tilt")
ACTION_SPEC_DIGEST = content_digest({"names": ACTION_NAMES})
INITIAL_CAPITAL = 250_000.0


class TinyContinuousEnv(gym.Env):
    metadata = {"render_modes": []}
    environment_digest = ENVIRONMENT_DIGEST
    initial_capital = INITIAL_CAPITAL
    decision_hours = 4.0
    action_names = ACTION_NAMES
    action_spec_digest = ACTION_SPEC_DIGEST

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.steps = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        self.steps += 1
        return (
            np.zeros(2, dtype=np.float32),
            -float(np.square(action).sum()),
            self.steps >= 4,
            False,
            {},
        )


class FakeBackend:
    def train(
        self, *, seed: int, config: ResidualTrainingConfig, output_path: Path
    ) -> PolicyTrainingResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"checkpoint:{seed}".encode())
        return PolicyTrainingResult(
            checkpoint_path=output_path,
            actual_timesteps=config.rounded_timesteps,
            resolved_device="cpu",
            environment_digest=ENVIRONMENT_DIGEST,
            initial_capital=INITIAL_CAPITAL,
            action_size=3,
            action_names=ACTION_NAMES,
            action_spec_digest=ACTION_SPEC_DIGEST,
            observation_size=2,
        )


def manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="a" * 64,
        symbols=("BTC",),
        feature_names=("ret",),
        base_timeframe="1h",
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


def test_gamma_is_bound_to_real_time_decision_interval() -> None:
    gamma = gamma_from_half_life(decision_hours=4.0, half_life_hours=24.0)
    assert gamma**6 == pytest.approx(0.5)
    with pytest.raises(ValueError, match="gamma"):
        ResidualTrainingConfig(
            timesteps=8,
            gamma=0.9,
            seeds=(0,),
            decision_hours=4.0,
            discount_half_life_hours=24.0,
        )


def test_ppo_exploration_and_network_controls_are_part_of_config_identity() -> None:
    first = ResidualTrainingConfig(
        timesteps=8, gamma=0.99, seeds=(0,), n_steps=8, batch_size=8, log_std_init=-1.0
    )
    second = ResidualTrainingConfig(
        timesteps=8, gamma=0.99, seeds=(0,), n_steps=8, batch_size=8, log_std_init=-0.5
    )
    assert content_digest(first.digest_payload()) != content_digest(
        second.digest_payload()
    )


def test_sb3_backend_validates_cadence_and_action_identity(tmp_path: Path) -> None:
    config = ResidualTrainingConfig(
        timesteps=8,
        gamma=0.99,
        seeds=(0,),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        device="cpu",
        decision_hours=4.0,
        asset_set_encoder=False,
    )
    result = StableBaselines3PPOBackend(TinyContinuousEnv).train(
        seed=0, config=config, output_path=tmp_path / "policy.zip"
    )
    assert result.action_names == ACTION_NAMES
    assert result.action_spec_digest == ACTION_SPEC_DIGEST


def test_ensemble_manifest_carries_exact_action_and_observation_identity(
    tmp_path: Path,
) -> None:
    config = ResidualTrainingConfig(
        timesteps=8, gamma=0.99, seeds=(0, 1), n_steps=8, batch_size=8
    )
    result = train_residual_ensemble(
        dataset=manifest(),
        environment_dataset_id="a" * 64,
        config=config,
        backend=FakeBackend(),
        output_dir=tmp_path,
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    assert result.action_names == ACTION_NAMES
    assert result.action_spec_digest == ACTION_SPEC_DIGEST
    assert result.observation_size == 2
