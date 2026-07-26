from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.rl.checkpointing import (
    load_checkpoint_manifest,
    publish_checkpoint,
    validate_checkpoint_algorithm_identity,
)


class _IdentityModel:
    def __init__(self, payload: dict[str, object] | None) -> None:
        self.payload = payload

    def save(self, target: str) -> None:
        Path(target).with_suffix(".zip").write_bytes(b"checkpoint")

    def checkpoint_identity_payload(self) -> dict[str, object] | None:
        return self.payload


def _cost_identity() -> dict[str, object]:
    return {
        "algorithm": "cost_critic_ppo",
        "architecture_digest": "b" * 64,
        "cost_names": ["drawdown_excess", "drawdown_stop_event"],
        "cost_schema_digest": "c" * 64,
        "rollout_schema_digest": "d" * 64,
    }


def _publish(tmp_path: Path, payload: dict[str, object] | None):
    return publish_checkpoint(
        model=_IdentityModel(payload),
        checkpoint_root=tmp_path / "checkpoints",
        algorithm="cost_critic_ppo" if payload is not None else "ppo",
        seed=7,
        requested_timestep=20,
        observed_timestep=20,
        environment_digest="e" * 64,
        training_config_digest="a" * 64,
    )


def test_checkpoint_round_trips_algorithm_identity(tmp_path: Path) -> None:
    identity = _cost_identity()
    manifest = _publish(tmp_path, identity)

    restored = load_checkpoint_manifest(manifest.policy_path.parent / "checkpoint.json")

    assert restored.algorithm_identity == identity
    assert restored.algorithm_identity_digest is not None
    validate_checkpoint_algorithm_identity(restored, identity)


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("cost_names", ["drawdown_stop_event", "drawdown_excess"]),
        ("cost_schema_digest", "f" * 64),
        ("architecture_digest", "1" * 64),
        ("rollout_schema_digest", "2" * 64),
    ],
)
def test_checkpoint_rejects_cost_identity_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    identity = _cost_identity()
    manifest = _publish(tmp_path, identity)
    expected = {**identity, field: replacement}

    with pytest.raises(ValueError, match="algorithm identity mismatch"):
        validate_checkpoint_algorithm_identity(manifest, expected)


def test_checkpoint_requires_cost_identity_when_expected(tmp_path: Path) -> None:
    manifest = _publish(tmp_path, None)

    with pytest.raises(ValueError, match="algorithm identity is missing"):
        validate_checkpoint_algorithm_identity(manifest, _cost_identity())


def test_ordinary_checkpoint_remains_identity_free(tmp_path: Path) -> None:
    manifest = _publish(tmp_path, None)
    restored = load_checkpoint_manifest(manifest.policy_path.parent / "checkpoint.json")

    assert restored.algorithm_identity is None
    assert restored.algorithm_identity_digest is None
    validate_checkpoint_algorithm_identity(restored, None)
