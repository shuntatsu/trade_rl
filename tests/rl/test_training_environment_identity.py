from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.datasets import DatasetManifest
from trade_rl.rl.training import (
    PolicyTrainingResult,
    ResidualTrainingConfig,
    train_residual_ensemble,
)

DATASET_ID = "a" * 64
ENVIRONMENT_DIGEST = "b" * 64


def manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_id=DATASET_ID,
        symbols=("BTCUSDT", "ETHUSDT"),
        feature_names=("ret",),
        base_timeframe="1h",
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


class EnvironmentBackend:
    def __init__(self, capitals: tuple[float, ...]) -> None:
        self.capitals = capitals
        self.call_index = 0

    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult:
        del seed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"checkpoint:{self.call_index}".encode())
        capital = self.capitals[self.call_index]
        self.call_index += 1
        return PolicyTrainingResult(
            checkpoint_path=output_path,
            actual_timesteps=config.rounded_timesteps,
            resolved_device="cpu",
            environment_digest=ENVIRONMENT_DIGEST,
            initial_capital=capital,
            action_size=3,
            action_names=("fast_tilt", "slow_tilt", "risk_tilt"),
            action_spec_digest=content_digest(
                {"names": ("fast_tilt", "slow_tilt", "risk_tilt")}
            ),
            observation_size=8,
        )


def config() -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=8,
        gamma=0.99,
        seeds=(0, 1),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        device="cpu",
    )


def test_policy_manifest_records_environment_and_aum(tmp_path: Path) -> None:
    result = train_residual_ensemble(
        dataset=manifest(),
        environment_dataset_id=DATASET_ID,
        config=config(),
        backend=EnvironmentBackend((250_000.0, 250_000.0)),
        output_dir=tmp_path,
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    assert result.environment_digest == ENVIRONMENT_DIGEST
    assert result.initial_capital == pytest.approx(250_000.0)


def test_ensemble_rejects_inconsistent_aum_across_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="initial capital"):
        train_residual_ensemble(
            dataset=manifest(),
            environment_dataset_id=DATASET_ID,
            config=config(),
            backend=EnvironmentBackend((250_000.0, 500_000.0)),
            output_dir=tmp_path,
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        )
