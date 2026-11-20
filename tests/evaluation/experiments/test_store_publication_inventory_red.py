"""Synthetic RED: a post-builder stage mutation must not become trusted Study evidence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments import store as store_module
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.store import StudyStore


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires Linux descriptor-relative publication",
)
def test_directory_publish_rejects_new_file_added_during_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StudyStore(tmp_path / "study")
    original = store_module.atomic_rename_directory
    injected = False

    def inject_before_rename(source: str | Path, target: str | Path) -> Path:
        nonlocal injected
        if Path(target).name == "synthetic" and not injected:
            injected = True
            (Path(source) / "forged.json").write_text(
                '{"trusted":false}', encoding="utf-8"
            )
        return original(source, target)

    monkeypatch.setattr(store_module, "atomic_rename_directory", inject_before_rename)

    with pytest.raises(ArtifactIntegrityError, match="(contents|integrity|changed)"):
        store.publish_directory_once(
            "evidence/synthetic",
            lambda stage: (stage / "genuine.json").write_text(
                '{"trusted":true}', encoding="utf-8"
            ),
        )
    assert injected, "injection must execute inside the publication window"
    assert not (store.root / "evidence/synthetic").exists()


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires Linux descriptor-relative publication",
)
def test_directory_publish_rejects_payload_change_during_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StudyStore(tmp_path / "study")
    original = store_module.atomic_rename_directory
    injected = False

    def inject_before_rename(source: str | Path, target: str | Path) -> Path:
        nonlocal injected
        if Path(target).name == "synthetic" and not injected:
            injected = True
            (Path(source) / "genuine.json").write_text(
                '{"trusted":false}', encoding="utf-8"
            )
        return original(source, target)

    monkeypatch.setattr(store_module, "atomic_rename_directory", inject_before_rename)

    with pytest.raises(ArtifactIntegrityError, match="(contents|integrity|changed)"):
        store.publish_directory_once(
            "evidence/synthetic",
            lambda stage: (stage / "genuine.json").write_text(
                '{"trusted":true}', encoding="utf-8"
            ),
        )
    assert injected, "modification must execute inside the publication window"
    assert not (store.root / "evidence/synthetic").exists()


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires Linux descriptor-relative publication",
)
def test_directory_publish_rejects_post_rename_extra_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The published inode can be changed after rename, before commit validation."""
    store = StudyStore(tmp_path / "study")
    original = store_module.atomic_rename_directory
    injected = False

    def inject_after_rename(source: str | Path, target: str | Path) -> Path:
        nonlocal injected
        result = original(source, target)
        if Path(target).name == "synthetic" and not injected:
            injected = True
            (Path(target) / "forged.json").write_text(
                '{"trusted":false}', encoding="utf-8"
            )
        return result

    monkeypatch.setattr(store_module, "atomic_rename_directory", inject_after_rename)
    with pytest.raises(ArtifactIntegrityError, match="(contents|integrity|changed)"):
        store.publish_directory_once(
            "evidence/synthetic",
            lambda stage: (stage / "genuine.json").write_text(
                '{"trusted":true}', encoding="utf-8"
            ),
        )
    assert injected
    assert not (store.root / "evidence/synthetic").exists()


def test_directory_publish_preserves_nested_binary_and_empty_directory(
    tmp_path: Path,
) -> None:
    """The integrity inventory must not reject a valid nested builder tree."""
    store = StudyStore(tmp_path / "study")

    def builder(stage: Path) -> None:
        (stage / "runs" / "seed-1").mkdir(parents=True)
        (stage / "runs" / "seed-1" / "weights.bin").write_bytes(b"\0\xff\x01")
        (stage / "empty").mkdir()
        (stage / "manifest.json").write_text('{"valid":true}', encoding="utf-8")

    published = store.publish_directory_once("evidence/synthetic", builder)
    assert (published / "runs/seed-1/weights.bin").read_bytes() == b"\0\xff\x01"
    assert (published / "empty").is_dir()
    assert store.read_json("evidence/synthetic/manifest.json") == {"valid": True}
