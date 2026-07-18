from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.serving.helpers import create_bundle
from trade_rl.release.attestation import default_attestation_path
from trade_rl.serving.bundle import load_serving_bundle


def test_release_attestation_binds_candidate_without_changing_bundle_digest(
    tmp_path: Path,
) -> None:
    candidate = load_serving_bundle(
        create_bundle(tmp_path / "candidate", release_digest=None)
    )
    released = load_serving_bundle(create_bundle(tmp_path / "released"))

    assert released.manifest.bundle_digest == candidate.manifest.bundle_digest
    assert released.release is not None
    assert released.release.bundle_digest == released.manifest.bundle_digest


def test_fake_embedded_release_pointer_is_rejected(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "candidate", release_digest=None)
    path = root / "bundle.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["release_digest"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest|schema|fields"):
        load_serving_bundle(root)


def test_tampered_release_attestation_is_rejected(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "released")
    path = default_attestation_path(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bundle_digest"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="release|digest"):
        load_serving_bundle(root)
