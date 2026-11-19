"""Synthetic regressions for file-stage cleanup after a source-name substitution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments import store as store_module
from trade_rl.evaluation.experiments.errors import (
    InvalidExperimentStateError,
    PublishedArtifactCleanupError,
)
from trade_rl.evaluation.experiments.store import StudyStore


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires proc file descriptors for deterministic file-stage swap",
)
@pytest.mark.parametrize("conflicting_target", (False, True))
def test_json_stage_replacement_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conflicting_target: bool
) -> None:
    store = StudyStore(tmp_path / "study")
    detached = tmp_path / "verified-staging-detached"
    source_name: Path | None = None
    real_fsync = os.fsync

    def replace_after_verified_fsync(fd: int) -> None:
        nonlocal source_name
        real_fsync(fd)
        if source_name is not None:
            return
        stage = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if not stage.name.startswith(".synthetic.json.staging-"):
            return
        stage.rename(detached)
        stage.write_bytes(b"UNRELATED_REPLACEMENT")
        source_name = stage
        if conflicting_target:
            target = store.root / "evidence" / "synthetic.json"
            target.write_bytes(b"EXISTING_TARGET")

    monkeypatch.setattr(store_module.os, "fsync", replace_after_verified_fsync)
    if conflicting_target:
        with pytest.raises(InvalidExperimentStateError, match="already exists"):
            store.publish_json_once("evidence/synthetic.json", {"validated": True})
    else:
        with pytest.raises(PublishedArtifactCleanupError, match="cleanup failed"):
            store.publish_json_once("evidence/synthetic.json", {"validated": True})
        assert store.read_json("evidence/synthetic.json") == {"validated": True}

    assert source_name is not None
    assert detached.read_bytes() == b'{"validated":true}'
    assert source_name.read_bytes() == b"UNRELATED_REPLACEMENT", (
        "JSON cleanup deleted a file it did not create"
    )
    if conflicting_target:
        assert (store.root / "evidence" / "synthetic.json").read_bytes() == (
            b"EXISTING_TARGET"
        )


def test_json_staging_name_collision_does_not_delete_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StudyStore(tmp_path / "study")
    (store.root / "evidence").mkdir()
    collision = store.root / "evidence" / ".synthetic.json.staging-fixed"
    collision.write_bytes(b"PREEXISTING")
    monkeypatch.setattr(store_module, "_temporary_name_suffix", lambda: "fixed")

    with pytest.raises(FileExistsError):
        store.publish_json_once("evidence/synthetic.json", {"validated": True})

    assert collision.read_bytes() == b"PREEXISTING", (
        "failed exclusive open must not clean up a file it never created"
    )
    assert not (store.root / "evidence" / "synthetic.json").exists()
