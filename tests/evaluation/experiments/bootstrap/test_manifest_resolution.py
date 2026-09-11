"""Bootstrap manifest v1/v2 source-resolution compatibility contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.evaluation.experiments.bootstrap.test_workflow import (
    _config_path,
    _install_fakes,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.workflow import (
    bootstrap_canonical_m2_study,
    inspect_canonical_m2_bootstrap,
)


def _publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _install_fakes(monkeypatch)
    output = tmp_path / "canonical-m2"
    bootstrap_canonical_m2_study(_config_path(tmp_path), output)
    return output


def _read_manifest(root: Path) -> dict[str, object]:
    payload = json.loads((root / "bootstrap-manifest.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_manifest(root: Path, payload: dict[str, object]) -> None:
    body = dict(payload)
    body.pop("bootstrap_digest", None)
    payload = dict(body)
    payload["bootstrap_digest"] = content_digest(body)
    (root / "bootstrap-manifest.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_new_bootstrap_manifest_v2_binds_vision_resolution_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _publish(tmp_path, monkeypatch)
    manifest = _read_manifest(root)
    resolution = json.loads(
        (root / "source" / "vision-resolution.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == "canonical_m2_bootstrap_manifest_v2"
    assert manifest["vision_resolution_digest"] == content_digest(resolution)


def test_legacy_v1_bootstrap_without_resolution_remains_inspectable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _publish(tmp_path, monkeypatch)
    manifest = _read_manifest(root)
    manifest["schema_version"] = "canonical_m2_bootstrap_manifest_v1"
    manifest.pop("vision_resolution_digest", None)
    _write_manifest(root, manifest)
    (root / "source" / "vision-resolution.json").unlink()

    result = inspect_canonical_m2_bootstrap(root)

    assert result.root == root


def test_v1_manifest_rejects_resolution_file_schema_confusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _publish(tmp_path, monkeypatch)
    manifest = _read_manifest(root)
    manifest["schema_version"] = "canonical_m2_bootstrap_manifest_v1"
    manifest.pop("vision_resolution_digest", None)
    _write_manifest(root, manifest)

    with pytest.raises(ValueError, match="legacy|v1|resolution"):
        inspect_canonical_m2_bootstrap(root)


def test_v2_manifest_rejects_resolution_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _publish(tmp_path, monkeypatch)
    manifest = _read_manifest(root)
    manifest["schema_version"] = "canonical_m2_bootstrap_manifest_v2"
    manifest["vision_resolution_digest"] = "0" * 64
    _write_manifest(root, manifest)

    with pytest.raises(ValueError, match="resolution digest"):
        inspect_canonical_m2_bootstrap(root)
