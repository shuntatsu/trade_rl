from __future__ import annotations

import json
from pathlib import Path

from mars_lite.pipeline.residual_run_artifacts import (
    build_run_manifest,
    sha256_file,
    write_run_manifest,
)


def test_run_manifest_is_deterministic_and_content_bound(tmp_path: Path) -> None:
    (tmp_path / "fold_0").mkdir()
    model = tmp_path / "fold_0" / "model.zip"
    report = tmp_path / "report.json"
    model.write_bytes(b"model")
    report.write_text('{"ok":true}\n', encoding="utf-8")

    first = build_run_manifest(
        tmp_path,
        run_id="run-1",
        config={"dataset_identity": "data-1"},
    )
    second = build_run_manifest(
        tmp_path,
        run_id="run-1",
        config={"dataset_identity": "data-1"},
    )

    assert first == second
    assert [entry["path"] for entry in first["files"]] == [
        "fold_0/model.zip",
        "report.json",
    ]
    assert first["files"][0]["sha256"] == sha256_file(model)

    model.write_bytes(b"changed")
    changed = build_run_manifest(
        tmp_path,
        run_id="run-1",
        config={"dataset_identity": "data-1"},
    )
    assert changed != first


def test_write_run_manifest_excludes_itself(tmp_path: Path) -> None:
    (tmp_path / "artifact.txt").write_text("payload", encoding="utf-8")

    manifest = write_run_manifest(
        tmp_path,
        run_id="run-2",
        config={"effective_decision_every": 4},
    )
    persisted = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))

    assert persisted == manifest
    assert [entry["path"] for entry in manifest["files"]] == ["artifact.txt"]
