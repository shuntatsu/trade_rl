from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.errors import (
    ArtifactIntegrityError,
    InvalidExperimentStateError,
)
from trade_rl.evaluation.experiments.store import StudyStore


def test_publish_json_once_is_canonical_and_never_overwrites(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study")
    payload = {"z": 2, "a": [1, 3]}

    published = store.publish_json_once("plan.json", payload)

    assert published == tmp_path / "study" / "plan.json"
    assert published.read_bytes() == canonical_json_bytes(payload)
    original = published.read_bytes()

    with pytest.raises(InvalidExperimentStateError, match="already exists"):
        store.publish_json_once("plan.json", {"different": True})

    assert published.read_bytes() == original


def test_publish_directory_once_removes_failed_staging(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study")

    def fail_builder(stage: Path) -> None:
        (stage / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("builder failed")

    with pytest.raises(RuntimeError, match="builder failed"):
        store.publish_directory_once("evidence/baseline", fail_builder)

    assert not (tmp_path / "study" / "evidence" / "baseline").exists()
    leftovers = [
        path for path in (tmp_path / "study").rglob("*") if ".staging-" in path.name
    ]
    assert leftovers == []


def test_store_rejects_absolute_and_parent_traversal_paths(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study")

    with pytest.raises(ValueError, match="relative"):
        store.publish_json_once(tmp_path / "escape.json", {"x": 1})
    with pytest.raises(ValueError, match="parent traversal"):
        store.publish_json_once("../escape.json", {"x": 1})

    assert not (tmp_path / "escape.json").exists()


def test_store_rejects_symlinked_parent_beneath_study_root(tmp_path: Path) -> None:
    root = tmp_path / "study"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    store = StudyStore(root)
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        store.publish_json_once("linked/escape.json", {"x": 1})

    assert not (outside / "escape.json").exists()


def test_read_json_rejects_non_object_payload(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study")
    store.publish_json_once("value.json", [1, 2, 3])

    with pytest.raises(ArtifactIntegrityError, match="JSON object"):
        store.read_json("value.json")
