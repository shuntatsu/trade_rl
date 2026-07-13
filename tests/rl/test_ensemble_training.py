from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.domain.datasets import DatasetManifest
from trade_rl.rl.training import (
    PolicyTrainingResult,
    ResidualTrainingConfig,
    gamma_from_half_life,
    train_residual_ensemble,
)


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ResidualTrainingConfig, Path]] = []

    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult:
        self.calls.append((seed, config, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"checkpoint:{seed}".encode())
        return PolicyTrainingResult(
            checkpoint_path=output_path,
            actual_timesteps=config.rounded_timesteps,
            resolved_device="cpu",
        )


def manifest(dataset_id: str = "a" * 64) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=dataset_id,
        symbols=("BTCUSDT", "ETHUSDT"),
        feature_names=("ret", "rsi"),
        base_timeframe="1h",
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


def test_gamma_from_half_life_is_invariant_in_real_time() -> None:
    hourly_decision = gamma_from_half_life(
        decision_hours=1.0,
        half_life_hours=24.0,
    )
    four_hour_decision = gamma_from_half_life(
        decision_hours=4.0,
        half_life_hours=24.0,
    )

    assert hourly_decision**24 == pytest.approx(0.5)
    assert four_hour_decision**6 == pytest.approx(0.5)
    assert four_hour_decision == pytest.approx(hourly_decision**4)


def test_gamma_from_half_life_rejects_invalid_time_values() -> None:
    with pytest.raises(ValueError, match="decision_hours"):
        gamma_from_half_life(decision_hours=0.0, half_life_hours=24.0)
    with pytest.raises(ValueError, match="half_life_hours"):
        gamma_from_half_life(decision_hours=4.0, half_life_hours=0.0)


def test_training_config_rounds_requested_steps_to_complete_rollouts() -> None:
    config = ResidualTrainingConfig(
        timesteps=1_025,
        gamma=0.99,
        seeds=(0,),
        n_steps=1_024,
        batch_size=64,
    )

    assert config.rounded_timesteps == 2_048


def test_training_config_rejects_incoherent_ppo_settings() -> None:
    with pytest.raises(ValueError, match="batch_size must divide n_steps"):
        ResidualTrainingConfig(
            timesteps=10,
            gamma=0.99,
            seeds=(0,),
            n_steps=100,
            batch_size=64,
        )
    with pytest.raises(ValueError, match="learning_rate"):
        ResidualTrainingConfig(
            timesteps=10,
            gamma=0.99,
            seeds=(0,),
            learning_rate=0.0,
        )
    with pytest.raises(ValueError, match="device"):
        ResidualTrainingConfig(
            timesteps=10,
            gamma=0.99,
            seeds=(0,),
            device="",
        )


def test_train_residual_ensemble_creates_one_member_per_seed(tmp_path: Path) -> None:
    backend = FakeBackend()
    created_at = datetime(2026, 7, 13, 7, 0, tzinfo=UTC)
    config = ResidualTrainingConfig(
        timesteps=1_024,
        gamma=0.5,
        seeds=(0, 1, 2),
        n_steps=512,
        batch_size=64,
        device="cpu",
    )

    result = train_residual_ensemble(
        dataset=manifest(),
        environment_dataset_id="a" * 64,
        config=config,
        backend=backend,
        output_dir=tmp_path,
        created_at=created_at,
    )

    assert result.expected_members == 3
    assert tuple(member.seed for member in result.members) == (0, 1, 2)
    assert len({member.checkpoint_digest for member in result.members}) == 3
    assert result.dataset_id == "a" * 64
    assert result.action_schema == "baseline_residual_v1"
    assert result.observation_schema == "baseline_residual_observation_v2"
    assert result.requested_timesteps == 1_024
    assert result.actual_timesteps == 1_024
    assert result.resolved_device == "cpu"
    assert len(backend.calls) == 3
    assert all(call[1] == config for call in backend.calls)


def test_training_configuration_is_bound_into_policy_identity(tmp_path: Path) -> None:
    created_at = datetime(2026, 7, 13, 7, 0, tzinfo=UTC)
    base = ResidualTrainingConfig(
        timesteps=1_024,
        gamma=0.99,
        seeds=(0,),
        learning_rate=3e-4,
    )
    changed = ResidualTrainingConfig(
        timesteps=1_024,
        gamma=0.99,
        seeds=(0,),
        learning_rate=1e-4,
    )

    first = train_residual_ensemble(
        dataset=manifest(),
        environment_dataset_id="a" * 64,
        config=base,
        backend=FakeBackend(),
        output_dir=tmp_path / "first",
        created_at=created_at,
    )
    second = train_residual_ensemble(
        dataset=manifest(),
        environment_dataset_id="a" * 64,
        config=changed,
        backend=FakeBackend(),
        output_dir=tmp_path / "second",
        created_at=created_at,
    )

    assert first.training_config_digest != second.training_config_digest
    assert first.digest != second.digest


def test_training_rejects_dataset_identity_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset identity"):
        train_residual_ensemble(
            dataset=manifest(),
            environment_dataset_id="b" * 64,
            config=ResidualTrainingConfig(timesteps=10, gamma=0.5, seeds=(0,)),
            backend=FakeBackend(),
            output_dir=tmp_path,
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        )


def test_training_config_requires_unique_non_negative_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        ResidualTrainingConfig(timesteps=10, gamma=0.5, seeds=(0, 0))
    with pytest.raises(ValueError, match="non-negative"):
        ResidualTrainingConfig(timesteps=10, gamma=0.5, seeds=(-1,))
