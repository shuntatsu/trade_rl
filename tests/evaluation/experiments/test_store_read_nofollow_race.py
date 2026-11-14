"""Deterministic read-side path-swap regression for Study JSON artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.store import StudyStore


@pytest.mark.skipif(os.name == "nt", reason="symlink creation may require Windows privileges")
def test_study_read_refuses_symlink_inserted_after_regular_file_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No later name lookup may follow a path swapped after a file check."""
    store = StudyStore(tmp_path / "study")
    target = store.publish_json_once("plan.json", {"trusted": True})
    outside = tmp_path / "outside.json"
    outside.write_text('{"trusted": false}\n', encoding="utf-8")
    original_is_file = Path.is_file
    swapped = False

    def swap_after_check(path: Path) -> bool:
        nonlocal swapped
        checked = original_is_file(path)
        if path == target and not swapped:
            assert checked
            target.unlink()
            target.symlink_to(outside)
            swapped = True
        return checked

    monkeypatch.setattr(Path, "is_file", swap_after_check)
    with pytest.raises(ArtifactIntegrityError):
        store.read_json("plan.json")


@pytest.mark.skipif(os.name == "nt", reason="symlink creation may require Windows privileges")
def test_study_read_refuses_existing_symlink(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study")
    outside = tmp_path / "outside.json"
    outside.write_text('{"trusted": false}\n', encoding="utf-8")
    (store.root / "plan.json").symlink_to(outside)

    with pytest.raises(ArtifactIntegrityError, match="regular file"):
        store.read_json("plan.json")
