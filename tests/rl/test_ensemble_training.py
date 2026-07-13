from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.domain.datasets import DatasetManifest
from trade_rl.rl.training import (
    ResidualTrainingConfig,
    gamma_from_half_life,
    train_residual_ensemble,
)


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, float, Path]] = []

    def train(
        self,
        *,
        seed: int,
        timesteps: int,
        gamma: float,
        output_path: Path,
    ) -> Path:
        self.calls.append((seed, timesteps, gamma, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"checkpoint:{seed}".encode())
        return output_path


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


def test_train_residual_ensemble_creates_one_member_per_seed(tmp_path: Path) -> None:
    backend = FakeBackend()
    created_at = datetime(2026, 7, 13, 7, 0, tzinfo=UTC)

    result = train_residual_ensemble(
        dataset=manifest(),
        environment_dataset_id="a" * 64,
        config=ResidualTrainingConfig(
            timesteps=1_024,
            gamma=0.5,
            seeds=(0, 1, 2),
        ),
        backend=backend,
        output_dir=tmp_path,
        created_at=created_at,
    )

    assert result.expected_members == 3
    assert tuple(member.seed for member in result.members) == (0, 1, 2)
    assert len({member.checkpoint_digest for member in result.members}) == 3
    assert result.dataset_id == "a" * 64
    assert result.action_schema == "baseline_residual_v1"
    assert len(backend.calls) == 3
    assert all(call[1] == 1_024 for call in backend.calls)
    assert all(call[2] == 0.5 for call in backend.calls)


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
