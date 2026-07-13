from __future__ import annotations

import json
from pathlib import Path

import pytest
from trade_rl.artifacts.store import ArtifactStore


def write_marker(path: Path, value: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "marker.txt").write_text(value, encoding="utf-8")


def test_publish_run_moves_validated_stage_and_updates_latest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    stage = store.stage_run("run-001")
    write_marker(stage, "ok")

    published = store.publish_run(
        "run-001",
        validate=lambda path: (path / "marker.txt").read_text(encoding="utf-8")
        == "ok",
    )

    assert published == tmp_path / "runs" / "run-001"
    assert published.is_dir()
    assert not stage.exists()
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest == {"path": "runs/run-001", "run_id": "run-001"}


def test_failed_validation_preserves_previous_latest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    first = store.stage_run("run-001")
    write_marker(first, "first")
    store.publish_run("run-001", validate=lambda _: True)

    second = store.stage_run("run-002")
    write_marker(second, "broken")

    with pytest.raises(ValueError, match="validation"):
        store.publish_run("run-002", validate=lambda _: False)

    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == "run-001"
    assert second.is_dir()
    assert not (tmp_path / "runs" / "run-002").exists()


def test_mark_failed_isolates_partial_run(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    stage = store.stage_run("run-003")
    write_marker(stage, "partial")

    failed = store.mark_failed("run-003")

    assert failed == tmp_path / "failed" / "run-003"
    assert (failed / "marker.txt").read_text(encoding="utf-8") == "partial"
    assert not stage.exists()
