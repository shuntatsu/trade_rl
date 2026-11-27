"""File-backed and verified-byte Dataset readers share strict JSON semantics."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.data.test_market_dataset_artifact import _dataset
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    load_market_dataset_artifact,
    write_market_dataset_files,
)
from trade_rl.data.artifacts.verified_bytes import load_dataset_from_verified_bytes


@pytest.mark.parametrize("reader", ["file", "bytes"])
@pytest.mark.parametrize("case", ["root", "nested", "escaped"])
def test_both_dataset_readers_reject_duplicate_manifest_members(
    tmp_path: Path, case: str, reader: str
) -> None:
    artifact = write_market_dataset_files(tmp_path, _dataset())
    original = artifact.manifest_path.read_text(encoding="utf-8")
    if case == "root":
        needle = f'"artifact_digest":"{artifact.artifact_digest}"'
        duplicate = '"artifact_digest":"' + "0" * 64 + '"'
    elif case == "nested":
        needle = '"dtype":"<f8"'
        duplicate = needle
    else:
        needle = f'"artifact_digest":"{artifact.artifact_digest}"'
        duplicate = needle.replace("artifact_digest", r"\u0061rtifact_digest")
    assert needle in original
    changed = original.replace(needle, duplicate + "," + needle, 1).encode("utf-8")
    artifact.manifest_path.write_bytes(changed)
    with pytest.raises(ValueError, match="duplicate JSON member"):
        if reader == "file":
            load_market_dataset_artifact(tmp_path)
        else:
            load_dataset_from_verified_bytes(changed, artifact.arrays_path.read_bytes())


def test_dataset_readers_preserve_valid_json_identity(tmp_path: Path) -> None:
    original = _dataset()
    artifact = write_market_dataset_files(tmp_path, original)
    encoded = json.dumps(
        json.loads(artifact.manifest_path.read_bytes()), indent=2
    ).encode()
    artifact.manifest_path.write_bytes(encoded)
    restored, digest = load_dataset_from_verified_bytes(
        encoded, artifact.arrays_path.read_bytes()
    )
    assert digest == artifact.artifact_digest
    assert restored.dataset_id == original.dataset_id
    assert load_market_dataset_artifact(tmp_path).dataset_id == original.dataset_id


def _duplicate_identity_member(identity: str) -> str:
    return identity.replace("{", '{"schema":"wrong",', 1)


def test_dataset_constructor_rejects_ambiguous_identity_json() -> None:
    dataset = _dataset()
    assert dataset.identity_payload_json is not None
    with pytest.raises(ValueError, match="duplicate JSON member"):
        replace(
            dataset,
            identity_payload_json=_duplicate_identity_member(
                dataset.identity_payload_json
            ),
        )


@pytest.mark.parametrize("reader", ["file", "bytes"])
def test_dataset_readers_reject_embedded_ambiguous_identity(
    tmp_path: Path, reader: str
) -> None:
    artifact = write_market_dataset_files(tmp_path, _dataset())
    manifest = json.loads(artifact.manifest_path.read_bytes())
    identity = manifest["dataset"]["identity_payload_json"]
    manifest["dataset"]["identity_payload_json"] = _duplicate_identity_member(identity)
    manifest.pop("artifact_digest")
    manifest["artifact_digest"] = content_digest(manifest)
    changed = canonical_json_bytes(manifest)
    artifact.manifest_path.write_bytes(changed)
    with pytest.raises(ValueError, match="duplicate JSON member"):
        if reader == "file":
            load_market_dataset_artifact(tmp_path)
        else:
            load_dataset_from_verified_bytes(changed, artifact.arrays_path.read_bytes())
