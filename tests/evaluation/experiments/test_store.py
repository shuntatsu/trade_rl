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


def test_publish_json_once_preserves_late_competing_file_after_last_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An external writer can claim the target just after our last false check."""
    store = StudyStore(tmp_path / "study")
    target = store.root / "plan.json"
    competitor = canonical_json_bytes({"competitor": "keep"})
    original_is_symlink = Path.is_symlink
    injected = False

    def race_after_final_false_symlink_check(path: Path) -> bool:
        nonlocal injected
        was_symlink = original_is_symlink(path)
        if (
            path == target
            and not injected
            and tuple(store.root.glob(".plan.json.staging-*"))
        ):
            assert not target.exists()
            target.write_bytes(competitor)
            injected = True
        return was_symlink

    monkeypatch.setattr(Path, "is_symlink", race_after_final_false_symlink_check)
    with pytest.raises(InvalidExperimentStateError, match="already exists"):
        store.publish_json_once("plan.json", {"owner": "this writer"})

    assert injected, "the after-check contender was not injected"
    assert target.read_bytes() == competitor
    assert not tuple(store.root.glob(".plan.json.staging-*"))


def test_publish_json_once_exclusive_rename_rejects_late_competing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A competitor after the helper's precheck must lose at the OS syscall."""
    import trade_rl.artifacts.atomic_write as atomic_write

    store = StudyStore(tmp_path / "study")
    target = store.root / "plan.json"
    competitor = canonical_json_bytes({"competitor": "claimed"})
    original_rename = atomic_write._rename_directory_no_replace
    injected = False

    def race_at_os_rename(source: Path, destination: Path) -> None:
        nonlocal injected
        assert destination == target
        assert not injected
        assert source.is_file()
        assert not target.exists()
        target.write_bytes(competitor)
        injected = True
        original_rename(source, destination)

    monkeypatch.setattr(atomic_write, "_rename_directory_no_replace", race_at_os_rename)
    with pytest.raises(InvalidExperimentStateError, match="already exists"):
        store.publish_json_once("plan.json", {"owner": "this writer"})

    assert injected, "the exclusive OS rename was not reached"
    assert target.read_bytes() == competitor
    assert not tuple(store.root.glob(".plan.json.staging-*"))


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
