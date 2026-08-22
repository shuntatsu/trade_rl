from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_training_runner import train_universal_seeds


def _digest(label: str) -> str:
    return content_digest(label)


@dataclass(frozen=True)
class _TrainingResult:
    checkpoint_path: Path
    environment_digest: str
    architecture_digest: str | None
    actual_timesteps: int


class _Backend:
    def __init__(
        self,
        result_identity: Callable[[int], tuple[str, str | None, int]],
    ) -> None:
        self._result_identity = result_identity

    def train(self, *, seed: int, config: object, output_path: Path) -> object:
        del config
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"policy:{seed}".encode())
        environment, architecture, timesteps = self._result_identity(seed)
        return _TrainingResult(
            checkpoint_path=output_path,
            environment_digest=environment,
            architecture_digest=architecture,
            actual_timesteps=timesteps,
        )


def _runtime() -> object:
    return SimpleNamespace(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        catalog_digest=_digest("catalog"),
        partition_digest=_digest("partition"),
        split_manifest_digest=_digest("split"),
        feature_schema_digest=_digest("features"),
        statistics_digest=_digest("statistics"),
        instrument_context_schema_digest=_digest("context"),
        training_contract_digest=_digest("training-contract"),
        pretraining_artifact_digest=_digest("pretraining"),
    )


def _training() -> object:
    return SimpleNamespace(
        seeds=(3, 5),
        digest_payload=lambda: {"schema_version": "training-test-v1"},
    )


def _run(tmp_path: Path, backend: object) -> None:
    train_universal_seeds(
        runtime=_runtime(),
        training=_training(),
        backend=backend,
        output_root=tmp_path,
        architecture_name="u_medium_direct",
    )


def test_member_environment_digest_must_be_sha256_before_manifest_publish(
    tmp_path: Path,
) -> None:
    backend = _Backend(lambda _seed: ("not-a-digest", _digest("arch"), 32))

    with pytest.raises(ValueError, match="environment"):
        _run(tmp_path, backend)

    assert not (tmp_path / "universal-training.json").exists()


def test_member_architecture_digest_is_mandatory_before_manifest_publish(
    tmp_path: Path,
) -> None:
    backend = _Backend(lambda _seed: (_digest("env"), None, 32))

    with pytest.raises(ValueError, match="architecture"):
        _run(tmp_path, backend)

    assert not (tmp_path / "universal-training.json").exists()


def test_member_identity_cannot_drift_between_seeds(tmp_path: Path) -> None:
    backend = _Backend(
        lambda seed: (
            _digest("env"),
            _digest(f"architecture:{seed}"),
            32,
        )
    )

    with pytest.raises(ValueError, match="architecture"):
        _run(tmp_path, backend)

    assert not (tmp_path / "universal-training.json").exists()


def test_actual_timesteps_must_be_positive_before_manifest_publish(
    tmp_path: Path,
) -> None:
    backend = _Backend(lambda _seed: (_digest("env"), _digest("arch"), 0))

    with pytest.raises(ValueError, match="actual_timesteps"):
        _run(tmp_path, backend)

    assert not (tmp_path / "universal-training.json").exists()


def test_member_environment_identity_cannot_drift_between_seeds(
    tmp_path: Path,
) -> None:
    backend = _Backend(
        lambda seed: (
            _digest(f"environment:{seed}"),
            _digest("architecture"),
            32,
        )
    )

    with pytest.raises(ValueError, match="environment"):
        _run(tmp_path, backend)

    assert not (tmp_path / "universal-training.json").exists()
