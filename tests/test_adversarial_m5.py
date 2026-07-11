import threading
from pathlib import Path

import pytest

from mars_lite.serving.candidate import create_candidate_bundle
from mars_lite.serving.registry import ModelRegistry


def _candidate(root: Path, version: str, payload: bytes) -> Path:
    model = root.parent / f"{root.name}.zip"
    model.write_bytes(payload)
    return create_candidate_bundle(
        destination=root,
        model_source=model,
        version=version,
        git_sha="a" * 40,
        symbols=("BTCUSDT",),
        feature_names=("ret",),
        global_feature_names=(),
        feature_norm="none",
        feature_mask=None,
        observation_dim=5,
        observation_schema_version=1,
        post_processor={},
        run_config={"observation_progress_mode": "zero"},
        metrics={},
        guardrails={},
        pre_trade={},
    )


def test_concurrent_registration_preserves_both_versions(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    first = _candidate(tmp_path / "candidate-v1", "v1", b"one")
    second = _candidate(tmp_path / "candidate-v2", "v2", b"two")
    errors: list[Exception] = []

    def register(path: Path) -> None:
        try:
            registry.register(path)
        except Exception as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    threads = [
        threading.Thread(target=register, args=(first,)),
        threading.Thread(target=register, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert registry.list_versions() == ["v1", "v2"]


def test_failed_copy_removes_partial_temporary_directory(
    tmp_path: Path, monkeypatch
) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    candidate = _candidate(tmp_path / "candidate", "v1", b"one")

    def fail_copy(source, destination):
        Path(destination).mkdir(parents=True)
        (Path(destination) / "partial").write_text("partial", encoding="utf-8")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("mars_lite.serving.registry.shutil.copytree", fail_copy)
    with pytest.raises(OSError, match="No space left"):
        registry.register(candidate)

    assert registry.list_versions() == []
    assert not any(
        path.name.startswith(".v1") for path in registry.versions_dir.iterdir()
    )


def test_failed_active_pointer_write_preserves_previous_version(
    tmp_path: Path, monkeypatch
) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(_candidate(tmp_path / "candidate-v1", "v1", b"one"))
    registry.register(_candidate(tmp_path / "candidate-v2", "v2", b"two"))
    registry.activate("v1", evidence_identity="run-1")

    def fail_write(path, value):
        raise OSError("simulated active pointer failure")

    monkeypatch.setattr(registry, "_atomic_write_json", fail_write)
    with pytest.raises(OSError, match="active pointer"):
        registry.activate("v2", evidence_identity="run-2")

    assert registry.get_active_bundle().version == "v1"


def test_same_version_with_different_digest_is_rejected(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(_candidate(tmp_path / "candidate-one", "v1", b"one"))
    conflicting = _candidate(tmp_path / "candidate-two", "v1", b"two")

    with pytest.raises(ValueError, match="already registered"):
        registry.register(conflicting)
    assert registry.list_versions() == ["v1"]
