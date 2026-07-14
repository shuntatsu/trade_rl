from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.rl.checkpointing import (
    load_checkpoint_manifest,
    publish_checkpoint,
)
from trade_rl.rl.training import ResidualTrainingConfig


class FakeModel:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def save(self, target: str) -> None:
        if self.fail:
            raise RuntimeError("save failed")
        Path(target).with_suffix(".zip").write_bytes(b"checkpoint")


def test_checkpoint_interval_is_derived_and_can_be_disabled() -> None:
    default = ResidualTrainingConfig(
        timesteps=100, gamma=0.99, seeds=(0,), n_steps=20, batch_size=20
    )
    disabled = ResidualTrainingConfig(
        timesteps=100,
        gamma=0.99,
        seeds=(0,),
        n_steps=20,
        batch_size=20,
        checkpoint_interval_steps=0,
    )

    assert default.resolved_checkpoint_interval == 20
    assert disabled.resolved_checkpoint_interval == 0


def test_publish_checkpoint_is_atomic_and_content_addressed(tmp_path: Path) -> None:
    manifest = publish_checkpoint(
        model=FakeModel(),
        checkpoint_root=tmp_path / "checkpoints",
        algorithm="ppo",
        seed=7,
        requested_timestep=20,
        observed_timestep=21,
        environment_digest="e" * 64,
        training_config_digest="a" * 64,
    )

    root = tmp_path / "checkpoints" / "step-000000000021"
    assert manifest.policy_path == root / "policy.zip"
    assert manifest.policy_path.read_bytes() == b"checkpoint"
    restored = load_checkpoint_manifest(root / "checkpoint.json")
    assert restored == manifest
    assert len(restored.policy_digest) == 64
    assert len(restored.digest) == 64


def test_failed_checkpoint_publish_removes_staging(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"

    with pytest.raises(RuntimeError, match="save failed"):
        publish_checkpoint(
            model=FakeModel(fail=True),
            checkpoint_root=root,
            algorithm="ppo",
            seed=0,
            requested_timestep=10,
            observed_timestep=10,
            environment_digest="e" * 64,
            training_config_digest="a" * 64,
        )

    assert not root.exists() or not tuple(root.iterdir())


def test_load_checkpoint_manifest_rejects_policy_digest_mismatch(
    tmp_path: Path,
) -> None:
    manifest = publish_checkpoint(
        model=FakeModel(),
        checkpoint_root=tmp_path / "checkpoints",
        algorithm="ppo",
        seed=0,
        requested_timestep=10,
        observed_timestep=10,
        environment_digest="e" * 64,
        training_config_digest="a" * 64,
    )
    manifest.policy_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="policy digest"):
        load_checkpoint_manifest(manifest.policy_path.parent / "checkpoint.json")
