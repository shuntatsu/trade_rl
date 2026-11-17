"""Test-only RED: replacing the staging leaf must not redirect builder writes.

The symlink, staging directory, and outside marker live entirely in pytest's
disposable tmp_path. Historical Study and market artifacts are never accessed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.store import StudyStore


@pytest.mark.skipif(
    os.name != "posix"
    or not getattr(os, "O_NOFOLLOW", 0)
    or not Path("/proc/self/fd").is_dir(),
    reason="requires POSIX directory-fd publication",
)
def test_directory_builder_refuses_replaced_staging_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An attacker-writable staging name is not a pinned directory capability."""
    store = StudyStore(tmp_path / "study")
    outside = tmp_path / "outside-stage-leaf"
    outside.mkdir()
    detached_stage = tmp_path / "original-stage-leaf"
    real_write_text = Path.write_text
    injected = False

    def replace_leaf_before_builder_write(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        nonlocal injected
        if path.name == "stage-leaf-marker.txt" and not injected:
            stages = list(store.root.glob(".synthetic.staging-*"))
            assert len(stages) == 1
            original_stage = stages[0]
            assert original_stage.is_dir() and not original_stage.is_symlink()
            original_stage.rename(detached_stage)
            original_stage.symlink_to(outside, target_is_directory=True)
            injected = True
        return real_write_text(
            path, data, encoding=encoding, errors=errors, newline=newline
        )

    monkeypatch.setattr(Path, "write_text", replace_leaf_before_builder_write)

    def build(stage: Path) -> None:
        (stage / "stage-leaf-marker.txt").write_text("synthetic", encoding="utf-8")

    with pytest.raises((ArtifactIntegrityError, ValueError, OSError)):
        store.publish_directory_once("evidence/synthetic", build)

    assert injected, "the concurrent stage-leaf substitution was not exercised"
    assert not (outside / "stage-leaf-marker.txt").exists(), (
        "builder wrote outside Study through a replaced staging leaf"
    )
    assert not (store.root / "evidence" / "synthetic").exists()
