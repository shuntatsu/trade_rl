from pathlib import Path

import pytest

from mars_lite.serving.bundle import build_manifest
from mars_lite.serving.registry import ModelRegistry


def create_bundle(root: Path, version: str, payload: bytes) -> Path:
    root.mkdir()
    (root / "model.zip").write_bytes(payload)
    (root / "metadata.json").write_text(
        '{"schema_version":1,"model_version":"%s","git_sha":"abc123",'
        '"symbols":["BTCUSDT"],"observation_schema_version":1,'
        '"observation_progress_mode":"zero","observation_dim":5,"run_config":{}}'
        % version,
        encoding="utf-8",
    )
    (root / "preprocessing.json").write_text(
        '{"feature_names":["ret"],"global_feature_names":[],"feature_norm":"none",'
        '"feature_mask":[true],"post_mask_dim":1}',
        encoding="utf-8",
    )
    (root / "risk.json").write_text(
        '{"guardrails":{},"pre_trade":{}}', encoding="utf-8"
    )
    build_manifest(root)
    return root


def test_register_activate_and_rollback_are_atomic(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    v1 = create_bundle(tmp_path / "v1", version="v1", payload=b"one")
    v2 = create_bundle(tmp_path / "v2", version="v2", payload=b"two")

    registry.register(v1)
    registry.activate("v1", evidence_identity="run-1")
    assert registry.get_active_bundle().version == "v1"

    registry.register(v2)
    registry.activate("v2", evidence_identity="run-2")
    assert registry.get_active_bundle().version == "v2"

    registry.rollback()
    assert registry.get_active_bundle().version == "v1"


def test_failed_activation_preserves_old_active(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    v1 = create_bundle(tmp_path / "v1", version="v1", payload=b"one")
    registry.register(v1)
    registry.activate("v1", evidence_identity="run-1")

    with pytest.raises(KeyError):
        registry.activate("missing", evidence_identity="run-x")
    assert registry.get_active_bundle().version == "v1"


def test_registered_versions_are_immutable(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    v1 = create_bundle(tmp_path / "v1", version="v1", payload=b"one")
    registry.register(v1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(v1)


def test_registry_cli_register_activate_and_show(tmp_path: Path, capsys) -> None:
    from scripts.manage_registry import main

    candidate = create_bundle(tmp_path / "candidate", version="v1", payload=b"one")
    registry_dir = tmp_path / "registry"

    assert main(["--registry-dir", str(registry_dir), "register", str(candidate)]) == 0
    assert (
        main(
            [
                "--registry-dir",
                str(registry_dir),
                "activate",
                "v1",
                "--evidence-identity",
                "run-1",
            ]
        )
        == 0
    )
    assert main(["--registry-dir", str(registry_dir), "show-active"]) == 0
    output = capsys.readouterr().out
    assert '"version": "v1"' in output
