from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from tools import tmp_issue577_basis_publisher as publisher


def test_frozen_source_drift_fails_before_dataset_or_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frozen = tmp_path / "frozen-source-manifest.json"
    frozen.write_bytes(b"frozen-authority")

    monkeypatch.setattr(publisher, "load_index_preflight", lambda _path: {})
    monkeypatch.setattr(
        publisher,
        "download_all_sources",
        lambda _preflight: ([], {}, {}),
    )
    monkeypatch.setattr(
        publisher,
        "build_source_manifest",
        lambda _entries: {"content_digest": "a" * 64},
    )
    monkeypatch.setattr(publisher, "manifest_bytes", lambda _manifest: b"changed-source")
    build_dataset = Mock(side_effect=AssertionError("Dataset must not be built"))
    build_result = Mock(side_effect=AssertionError("training must not run"))
    monkeypatch.setattr(publisher, "build_dataset", build_dataset)
    monkeypatch.setattr(publisher, "build_result", build_result)

    with pytest.raises(RuntimeError, match="frozen pre-training authority"):
        publisher.execute(
            output_dir=tmp_path / "out",
            preflight_path=tmp_path / "index-preflight.json",
            expected_source_manifest=frozen,
            workflow_run_id=123,
        )

    build_dataset.assert_not_called()
    build_result.assert_not_called()
    assert not (tmp_path / "out").exists()
