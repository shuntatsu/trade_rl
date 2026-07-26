from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.rl.checkpointing import (
    publish_checkpoint,
    validate_checkpoint_algorithm_identity,
)


class _IdentityModel:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def save(self, target: str) -> None:
        Path(target).with_suffix(".zip").write_bytes(b"checkpoint")

    def checkpoint_identity_payload(self) -> dict[str, object]:
        return self.payload


def _identity() -> dict[str, object]:
    return {
        "algorithm": "lagrangian_ppo",
        "architecture_digest": "a" * 64,
        "cost_names": ["drawdown_excess", "drawdown_stop_event"],
        "cost_schema_digest": "b" * 64,
        "rollout_schema_digest": "c" * 64,
        "lagrangian_schema_digest": "d" * 64,
        "lagrangian_cost_names": ["drawdown_excess", "drawdown_stop_event"],
    }


def _publish(tmp_path: Path, identity: dict[str, object]):
    return publish_checkpoint(
        model=_IdentityModel(identity),
        checkpoint_root=tmp_path / "checkpoints",
        algorithm="lagrangian_ppo",
        seed=13,
        requested_timestep=40,
        observed_timestep=40,
        environment_digest="e" * 64,
        training_config_digest="f" * 64,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("lagrangian_schema_digest", "1" * 64),
        (
            "lagrangian_cost_names",
            ["drawdown_stop_event", "drawdown_excess"],
        ),
        ("architecture_digest", "2" * 64),
    ],
)
def test_checkpoint_rejects_lagrangian_identity_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    identity = _identity()
    manifest = _publish(tmp_path, identity)
    expected = {**identity, field: replacement}

    with pytest.raises(ValueError, match="algorithm identity mismatch"):
        validate_checkpoint_algorithm_identity(manifest, expected)


def test_checkpoint_accepts_exact_lagrangian_identity(tmp_path: Path) -> None:
    identity = _identity()
    manifest = _publish(tmp_path, identity)

    validate_checkpoint_algorithm_identity(manifest, identity)
