"""Test-only RED: Study publication must not follow a swapped parent path.

All paths and payloads belong to pytest's disposable tmp_path; no Study or
market artifact is consumed or modified.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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
