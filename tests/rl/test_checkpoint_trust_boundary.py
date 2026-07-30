from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_rl.rl.checkpointing import (
    load_checkpoint_manifest,
    publish_checkpoint,
    verified_checkpoint_policy_copy,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="symlink semantics require POSIX")


class FakeModel:
    def save(self, target: str) -> None:
        Path(target).with_suffix(".zip").write_bytes(b"checkpoint")


def _publish(tmp_path: Path) -> Path:
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
    return manifest.policy_path.parent


def test_checkpoint_manifest_must_not_be_a_symlink(tmp_path: Path) -> None:
    root = _publish(tmp_path)
    manifest_path = root / "checkpoint.json"
    external = tmp_path / "external-checkpoint.json"
    manifest_path.replace(external)
    manifest_path.symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        load_checkpoint_manifest(manifest_path)


def test_checkpoint_policy_must_not_be_a_symlink(tmp_path: Path) -> None:
    root = _publish(tmp_path)
    policy_path = root / "policy.zip"
    external = tmp_path / "external-policy.zip"
    policy_path.replace(external)
    policy_path.symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        load_checkpoint_manifest(root / "checkpoint.json")


def test_checkpoint_deserialization_uses_a_private_verified_copy(
    tmp_path: Path,
) -> None:
    root = _publish(tmp_path)
    manifest = load_checkpoint_manifest(root / "checkpoint.json")

    with verified_checkpoint_policy_copy(manifest) as verified:
        assert verified != manifest.policy_path
        assert verified.read_bytes() == manifest.policy_path.read_bytes()
        private_root = verified.parent

    assert not private_root.exists()
