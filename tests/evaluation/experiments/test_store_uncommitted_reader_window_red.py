"""Synthetic RED: readers must not accept a forged directory before publication commit."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments import store as store_module
from trade_rl.evaluation.experiments.errors import (
    ArtifactIntegrityError,
    InvalidExperimentStateError,
    PublishedArtifactCleanupError,
)
from trade_rl.evaluation.experiments.store import StudyStore


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires POSIX descriptor-relative directory publication",
)
def test_reader_rejects_forged_staging_inode_before_publisher_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed post-rename inode check must not expose forged bytes to readers."""
    store = StudyStore(tmp_path / "study")
    detached = tmp_path / "validated-stage-detached"
    original_rename = store_module.atomic_rename_directory
    interleaved: dict[str, object] = {}
    swap_exercised = False

    def swap_then_read(source: str | Path, target: str | Path) -> Path:
        nonlocal swap_exercised
        if not swap_exercised:
            stage = Path(source)
            assert stage.name.startswith(".synthetic.staging-")
            stage.rename(detached)
            stage.mkdir()
            (stage / "forged.json").write_text(
                '{"is_validated":false}', encoding="utf-8"
            )
            swap_exercised = True
            result = original_rename(source, target)
            # Interleave an actual repository reader precisely after rename,
            # before the writer's post-rename inode check and quarantine.
            try:
                interleaved["payload"] = store.read_json(
                    "evidence/synthetic/forged.json"
                )
            except (ArtifactIntegrityError, OSError, ValueError):
                interleaved["rejected"] = True
            return result
        return original_rename(source, target)

    monkeypatch.setattr(store_module, "atomic_rename_directory", swap_then_read)

    def build(stage: Path) -> None:
        (stage / "genuine.json").write_text('{"is_validated":true}', encoding="utf-8")

    try:
        store.publish_directory_once("evidence/synthetic", build)
    except (ArtifactIntegrityError, PublishedArtifactCleanupError, OSError):
        pass

    assert swap_exercised, "the staging-source attack must be exercised"
    assert (detached / "genuine.json").is_file()
    assert interleaved.get("rejected"), (
        f"Reader accepted uncommitted forged evidence: {interleaved!r}"
    )


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires POSIX descriptor-relative directory publication",
)
def test_independent_store_reader_cannot_observe_precommit_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second StudyStore must refuse even genuine bytes before commit."""
    writer = StudyStore(tmp_path / "study")
    reader = StudyStore(writer.root)
    original_rename = store_module.atomic_rename_directory
    interleaved: dict[str, object] = {}

    def publish_then_read(source: str | Path, target: str | Path) -> Path:
        result = original_rename(source, target)
        try:
            interleaved["payload"] = reader.read_json("evidence/synthetic/genuine.json")
        except ArtifactIntegrityError:
            interleaved["rejected"] = True
        return result

    monkeypatch.setattr(store_module, "atomic_rename_directory", publish_then_read)

    writer.publish_directory_once(
        "evidence/synthetic",
        lambda stage: (stage / "genuine.json").write_text(
            '{"is_validated":true}', encoding="utf-8"
        ),
    )
    assert interleaved.get("rejected"), (
        f"An independent Study reader saw the directory before commit: {interleaved!r}"
    )
    assert reader.read_json("evidence/synthetic/genuine.json") == {"is_validated": True}


def test_same_store_cannot_recursively_publish_over_pending_directory(
    tmp_path: Path,
) -> None:
    """An inner call must not erase the outer pending-read reservation."""
    store = StudyStore(tmp_path / "study")

    def build(stage: Path) -> None:
        with pytest.raises(InvalidExperimentStateError, match="in progress"):
            store.publish_directory_once(
                "evidence/synthetic",
                lambda child: (child / "forged.json").write_text(
                    '{"is_validated":false}', encoding="utf-8"
                ),
            )
        (stage / "genuine.json").write_text('{"is_validated":true}', encoding="utf-8")

    store.publish_directory_once("evidence/synthetic", build)
    assert store.read_json("evidence/synthetic/genuine.json") == {"is_validated": True}
    assert not (store.root / "evidence" / "synthetic" / "forged.json").exists()
