"""Synthetic regression: never recursively delete a substituted staging basename."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments import store as store_module
from trade_rl.evaluation.experiments.store import StudyStore


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires POSIX directory fd",
)
def test_failed_directory_publish_keeps_unrelated_substituted_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StudyStore(tmp_path / "study")
    detached = tmp_path / "verified-original-detached"
    original_rename = store_module.atomic_rename_directory
    swap_exercised = False

    def swap_and_conflict(source: str | Path, target: str | Path) -> Path:
        nonlocal swap_exercised
        source_name = store.root / Path(source).name
        assert source_name.is_dir()
        source_name.rename(detached)
        source_name.mkdir()
        (source_name / "unrelated-keep.txt").write_text("KEEP", encoding="utf-8")
        Path(target).mkdir()
        (Path(target) / "already-published.txt").write_text(
            "existing", encoding="utf-8"
        )
        swap_exercised = True
        return original_rename(source, target)

    monkeypatch.setattr(store_module, "atomic_rename_directory", swap_and_conflict)

    def build(stage: Path) -> None:
        (stage / "genuine.txt").write_text("GENUINE", encoding="utf-8")

    with pytest.raises((FileExistsError, OSError)):
        store.publish_directory_once("evidence/synthetic", build)

    assert swap_exercised
    assert (detached / "genuine.txt").read_text(encoding="utf-8") == "GENUINE"
    original_name = store.root / "evidence" / "synthetic"
    assert (original_name / "already-published.txt").read_text(
        encoding="utf-8"
    ) == "existing"
    substituted = list(store.root.glob(".synthetic.staging-*"))
    assert len(substituted) == 1, (
        "cleanup must not delete an unrelated stage replacement"
    )
    assert (substituted[0] / "unrelated-keep.txt").read_text(encoding="utf-8") == "KEEP"
