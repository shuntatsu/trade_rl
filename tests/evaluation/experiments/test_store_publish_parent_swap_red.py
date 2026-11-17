"""Test-only RED: Study publication must not follow a swapped parent path.

All paths and payloads belong to pytest's disposable tmp_path; no Study or
market artifact is consumed or modified.
"""

from __future__ import annotations

import errno
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments import store as store_module
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.store import StudyStore

_POSIX_NOFOLLOW = os.name == "posix" and hasattr(os, "O_NOFOLLOW")


def _swap_parent_after_validation(
    store: StudyStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    parent = store.root / "evidence"
    previous = store.root / "evidence-before-swap"
    outside = tmp_path / "outside"
    outside.mkdir()
    checked_mkdir = store._mkdir_checked_parents

    def replace_parent(target: Path) -> None:
        checked_mkdir(target)
        assert parent.is_dir() and not parent.is_symlink()
        parent.rename(previous)
        parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(store, "_mkdir_checked_parents", replace_parent)
    return parent, previous, outside


@pytest.mark.skipif(not _POSIX_NOFOLLOW, reason="requires POSIX no-follow")
def test_json_publication_refuses_parent_swapped_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StudyStore(tmp_path / "study")
    parent, previous, outside = _swap_parent_after_validation(
        store, tmp_path, monkeypatch
    )

    with pytest.raises(ArtifactIntegrityError):
        store.publish_json_once("evidence/synthetic.json", {"synthetic": True})

    assert parent.is_symlink()
    assert list(outside.iterdir()) == []
    assert not (previous / "synthetic.json").exists()


@pytest.mark.skipif(not _POSIX_NOFOLLOW, reason="requires POSIX no-follow")
def test_directory_publication_refuses_parent_swapped_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StudyStore(tmp_path / "study")
    parent, previous, outside = _swap_parent_after_validation(
        store, tmp_path, monkeypatch
    )

    def build(stage: Path) -> None:
        (stage / "synthetic.txt").write_text("synthetic", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError):
        store.publish_directory_once("evidence/synthetic", build)

    assert parent.is_symlink()
    assert list(outside.iterdir()) == []
    assert not (previous / "synthetic").exists()


@pytest.mark.skipif(not _POSIX_NOFOLLOW, reason="requires POSIX no-follow")
def test_directory_builder_cannot_escape_when_study_root_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-check Study-root swap must not redirect the builder's first write.

    Inject at the Path.write_text sink, after the store's last pre-builder
    pathname checks. All directories and the marker are disposable.
    """
    store = StudyStore(tmp_path / "study")
    outside = tmp_path / "outside"
    outside.mkdir()
    original_root = tmp_path / "study-before-swap"
    original_write_text = Path.write_text
    swapped = False

    def write_after_swap(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        nonlocal swapped
        if path.name == "builder-marker.txt" and not swapped:
            assert ".staging-" in path.parent.name
            swapped = True
            store.root.rename(original_root)
            store.root.symlink_to(outside, target_is_directory=True)
            # Model an attacker-controlled replacement that contains the same
            # transient stage basename as the legitimate private builder stage.
            (outside / path.parent.name).mkdir()
        return original_write_text(
            path, data, encoding=encoding, errors=errors, newline=newline
        )

    monkeypatch.setattr(Path, "write_text", write_after_swap)

    def build(stage: Path) -> None:
        (stage / "builder-marker.txt").write_text("synthetic", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError):
        store.publish_directory_once("evidence/synthetic", build)

    assert swapped
    assert not list(outside.rglob("builder-marker.txt"))
    assert not (original_root / "evidence" / "synthetic").exists()


@pytest.mark.skipif(not _POSIX_NOFOLLOW, reason="requires POSIX no-follow")
def test_directory_builder_keeps_nested_study_store_nofollow_contract(
    tmp_path: Path,
) -> None:
    """A safe stage path must still support nested locks and descriptor readers."""
    outer = StudyStore(tmp_path / "study")
    outside = tmp_path / "outside-nested"
    outside.mkdir()

    def build(stage: Path) -> None:
        nested = StudyStore(stage)
        with nested.mutation_lock():
            nested.publish_json_once("nested.json", {"synthetic": True})
            assert nested.read_json("nested.json") == {"synthetic": True}
            (stage / "linked").symlink_to(outside, target_is_directory=True)
            with pytest.raises(ArtifactIntegrityError):
                nested.read_json("linked/foreign.json")
            with pytest.raises(ArtifactIntegrityError):
                nested.publish_json_once("linked/foreign.json", {"invalid": True})
            (stage / "linked").unlink()
        # Mirror the existing evidence builder's removal of its staging lock.
        (stage / ".mutation.lock").unlink()

    published = outer.publish_directory_once("evidence/synthetic", build)
    assert published == outer.root / "evidence" / "synthetic"
    assert StudyStore(published).read_json("nested.json") == {"synthetic": True}
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(not _POSIX_NOFOLLOW, reason="requires POSIX no-follow")
def test_study_root_refuses_raw_symlink_dotdot_before_any_publication(
    tmp_path: Path,
) -> None:
    """A root symlink/.. must not relocate evidence to a lexical abspath."""
    outside = tmp_path / "outside"
    (outside / "nested").mkdir(parents=True)
    (outside / "safe").mkdir()
    (tmp_path / "safe").mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside / "nested", target_is_directory=True)

    root_with_parent_traversal = linked / ".." / "safe"
    assert ".." in root_with_parent_traversal.parts
    assert root_with_parent_traversal.resolve() == outside / "safe"
    assert os.path.abspath(root_with_parent_traversal) == str(tmp_path / "safe")

    with pytest.raises(ArtifactIntegrityError, match="parent traversal"):
        StudyStore(root_with_parent_traversal)

    assert list((outside / "safe").iterdir()) == []
    assert list((tmp_path / "safe").iterdir()) == []


@pytest.mark.skipif(not _POSIX_NOFOLLOW, reason="requires POSIX no-follow")
def test_builder_enotdir_is_not_a_parent_integrity_error(tmp_path: Path) -> None:
    """Do not rewrite a builder's ENOTDIR into a parent-tamper diagnosis."""
    store = StudyStore(tmp_path / "study")

    def fail_builder(_stage: Path) -> None:
        raise NotADirectoryError(errno.ENOTDIR, "synthetic builder failure")

    with pytest.raises(NotADirectoryError, match="synthetic builder failure"):
        store.publish_directory_once("evidence/synthetic", fail_builder)

    assert not (store.root / "evidence" / "synthetic").exists()
    assert not list(store.root.glob(".synthetic.staging-*"))


@pytest.mark.skipif(not _POSIX_NOFOLLOW, reason="requires POSIX no-follow")
@pytest.mark.parametrize("kind", ("json", "directory"))
def test_publication_rejects_swapped_parent_with_external_same_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """A post-pin symlink swap must be integrity failure, not duplicate-file."""
    store = StudyStore(tmp_path / "study")
    parent = store.root / "evidence"
    parent.mkdir()
    earlier = store.root / "evidence-before-swap"
    outside = tmp_path / "outside"
    outside.mkdir()
    name = "synthetic.json" if kind == "json" else "synthetic"
    external_target = outside / name
    if kind == "json":
        external_target.write_text("external", encoding="utf-8")
    else:
        external_target.mkdir()

    original_pin = store_module._pinned_publication_parent
    swapped = False

    @contextmanager
    def replace_after_descriptor_open(
        candidate: Path,
    ) -> Iterator[tuple[Path, Callable[[], None]]]:
        nonlocal swapped
        with original_pin(candidate) as pinned:
            if candidate == parent and not swapped:
                parent.rename(earlier)
                parent.symlink_to(outside, target_is_directory=True)
                swapped = True
            yield pinned

    monkeypatch.setattr(
        store_module, "_pinned_publication_parent", replace_after_descriptor_open
    )

    def build(stage: Path) -> None:
        (stage / "marker.txt").write_text("synthetic", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError):
        if kind == "json":
            store.publish_json_once("evidence/synthetic.json", {"synthetic": True})
        else:
            store.publish_directory_once("evidence/synthetic", build)

    assert swapped
    assert parent.is_symlink()
    assert not (earlier / name).exists()
    if kind == "json":
        assert external_target.read_text(encoding="utf-8") == "external"
    else:
        assert list(external_target.iterdir()) == []
