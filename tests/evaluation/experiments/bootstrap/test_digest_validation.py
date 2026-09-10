from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.workflow import (
    CanonicalM2BootstrapResult,
    _inspect_manifest,
)

_VALID = {
    "bootstrap_config_digest": "1" * 64,
    "vision_plan_digest": "2" * 64,
    "raw_source_roster_digest": "3" * 64,
    "dataset_id": "4" * 64,
    "dataset_artifact_digest": "5" * 64,
    "study_digest": "6" * 64,
    "implementation_digest": "7" * 64,
    "runtime_environment_digest": "8" * 64,
}
_INVALID_DIGESTS = (
    "A" * 64,
    "+" + "1" * 63,
    " " + "1" * 63,
)


@pytest.mark.parametrize("invalid_digest", _INVALID_DIGESTS)
def test_bootstrap_result_rejects_noncanonical_sha256(
    tmp_path: Path,
    invalid_digest: str,
) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        CanonicalM2BootstrapResult(
            root=tmp_path,
            config_digest=invalid_digest,
            bootstrap_digest="2" * 64,
            dataset_id="3" * 64,
            dataset_artifact_digest="4" * 64,
            study_digest="5" * 64,
        )


def _write_manifest(root: Path, *, config_digest: str) -> None:
    body: dict[str, object] = {
        "schema_version": "canonical_m2_bootstrap_manifest_v1",
        "bootstrap_config_digest": config_digest,
        "vision_plan_digest": _VALID["vision_plan_digest"],
        "raw_source_roster": [
            {
                "url": "https://example.invalid/archive.zip",
                "sha256": "9" * 64,
                "size_bytes": 1,
            }
        ],
        "raw_source_roster_digest": content_digest(
            [
                {
                    "url": "https://example.invalid/archive.zip",
                    "sha256": "9" * 64,
                    "size_bytes": 1,
                }
            ]
        ),
        "metadata_evidence": {},
        "dataset_id": _VALID["dataset_id"],
        "dataset_artifact_schema": "market_dataset_artifact_v1",
        "dataset_artifact_digest": _VALID["dataset_artifact_digest"],
        "study_digest": _VALID["study_digest"],
        "implementation_digest": _VALID["implementation_digest"],
        "runtime_environment_digest": _VALID["runtime_environment_digest"],
    }
    payload = {**body, "bootstrap_digest": content_digest(body)}
    (root / "bootstrap-manifest.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


@pytest.mark.parametrize("invalid_digest", _INVALID_DIGESTS)
def test_bootstrap_manifest_loader_rejects_noncanonical_sha256(
    tmp_path: Path,
    invalid_digest: str,
) -> None:
    _write_manifest(tmp_path, config_digest=invalid_digest)

    with pytest.raises(ValueError, match="SHA-256"):
        _inspect_manifest(tmp_path)


def test_bootstrap_digest_validation_accepts_canonical_lowercase_sha256(
    tmp_path: Path,
) -> None:
    canonical = "a" * 64
    result = CanonicalM2BootstrapResult(
        root=tmp_path,
        config_digest=canonical,
        bootstrap_digest="2" * 64,
        dataset_id="3" * 64,
        dataset_artifact_digest="4" * 64,
        study_digest="5" * 64,
    )
    assert result.config_digest == canonical

    _write_manifest(tmp_path, config_digest=canonical)
    manifest = _inspect_manifest(tmp_path)
    assert manifest["bootstrap_config_digest"] == canonical
