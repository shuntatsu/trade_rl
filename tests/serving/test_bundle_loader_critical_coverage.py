from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.serving.helpers import (
    ACTION_NAMES,
    ACTION_SPEC_DIGEST,
    INITIAL_CAPITAL,
    OBSERVATION_SIZE,
    create_bundle,
)
from trade_rl.domain.selection import PolicyMode
from trade_rl.release.attestation import (
    ReleaseAttestation,
    default_attestation_path,
    write_release_attestation,
)
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.serving.bundle import (
    ServingBundleManifest,
    load_serving_bundle,
    write_serving_bundle_manifest,
)

NOW = datetime(2026, 7, 14, tzinfo=UTC)


def _custom_manifest(
    root: Path,
    *,
    artifact_paths: tuple[str, ...],
    normalizer_digest: str | None,
) -> ServingBundleManifest:
    return ServingBundleManifest.build(
        root=root,
        dataset_id="a" * 64,
        action_schema=ACTION_SCHEMA,
        observation_schema=OBSERVATION_SCHEMA,
        observation_size=OBSERVATION_SIZE,
        environment_digest="d" * 64,
        initial_capital=INITIAL_CAPITAL,
        policy_mode=PolicyMode.BASELINE_ONLY,
        policy_digest=None,
        signal_digest="b" * 64,
        selection_digest="c" * 64,
        release_digest=None,
        artifact_paths=artifact_paths,
        created_at=NOW,
        action_size=len(ACTION_NAMES),
        action_names=ACTION_NAMES,
        action_spec_digest=ACTION_SPEC_DIGEST,
        normalizer_digest=normalizer_digest,
    )


def test_loader_rejects_missing_or_nonmapping_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest is missing"):
        load_serving_bundle(tmp_path / "missing")

    root = tmp_path / "nonmapping"
    root.mkdir()
    (root / "bundle.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_serving_bundle(root)


def test_loader_rejects_missing_unbound_and_mismatched_normalizer(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-normalizer"
    missing.mkdir()
    (missing / "dataset.json").write_text("{}", encoding="utf-8")
    manifest = _custom_manifest(
        missing,
        artifact_paths=("dataset.json",),
        normalizer_digest="f" * 64,
    )
    write_serving_bundle_manifest(missing, manifest)
    with pytest.raises(ValueError, match="declare.*normalizer"):
        load_serving_bundle(missing)

    unbound = tmp_path / "unbound-normalizer"
    unbound.mkdir()
    (unbound / "dataset.json").write_text("{}", encoding="utf-8")
    (unbound / "normalizer.json").write_text("{}", encoding="utf-8")
    manifest = _custom_manifest(
        unbound,
        artifact_paths=("dataset.json", "normalizer.json"),
        normalizer_digest=None,
    )
    write_serving_bundle_manifest(unbound, manifest)
    with pytest.raises(ValueError, match="unbound normalizer"):
        load_serving_bundle(unbound)

    mismatch = create_bundle(
        tmp_path / "normalizer-mismatch",
        release_digest=None,
    )
    loaded = load_serving_bundle(mismatch)
    artifact_paths = tuple(item.path for item in loaded.manifest.files)
    wrong = ServingBundleManifest.build(
        root=mismatch,
        dataset_id=loaded.manifest.dataset_id,
        action_schema=loaded.manifest.action_schema,
        observation_schema=loaded.manifest.observation_schema,
        observation_size=loaded.manifest.observation_size,
        environment_digest=loaded.manifest.environment_digest,
        initial_capital=loaded.manifest.initial_capital,
        policy_mode=loaded.manifest.policy_mode,
        policy_digest=loaded.manifest.policy_digest,
        signal_digest=loaded.manifest.signal_digest,
        selection_digest=loaded.manifest.selection_digest,
        release_digest=None,
        artifact_paths=artifact_paths,
        created_at=loaded.manifest.created_at,
        action_size=loaded.manifest.action_size,
        action_names=loaded.manifest.action_names,
        action_spec_digest=loaded.manifest.action_spec_digest,
        normalizer_digest="f" * 64,
    )
    write_serving_bundle_manifest(mismatch, wrong)
    with pytest.raises(ValueError, match="normalizer digest"):
        load_serving_bundle(mismatch)


def test_loader_rejects_legacy_release_pointer_mismatch(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "legacy")
    manifest = load_serving_bundle(root).manifest
    write_serving_bundle_manifest(root, replace(manifest, release_digest="f" * 64))
    with pytest.raises(ValueError, match="pointer mismatch"):
        load_serving_bundle(root)


def _external_attestation(
    root: Path,
    *,
    bundle_digest: str,
    dataset_id: str,
    selected_policy_digest: str | None,
) -> None:
    attestation = ReleaseAttestation.create(
        bundle_digest=bundle_digest,
        dataset_id=dataset_id,
        selection_evaluation_digest="1" * 64,
        gate_evaluation_digest="2" * 64,
        gate_evidence_digest="3" * 64,
        selected_policy_digest=selected_policy_digest,
        git_commit="4" * 40,
        dependency_digest="5" * 64,
        approver="coverage-test",
        approved_at=NOW,
        key_id="bundle-loader-key",
        signing_key=b"bundle-loader-signing-key",
    )
    write_release_attestation(default_attestation_path(root), attestation)


@pytest.mark.parametrize("mismatch", ["bundle", "dataset", "policy"])
def test_loader_rejects_external_attestation_identity_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    root = create_bundle(tmp_path / mismatch, release_digest=None)
    manifest = load_serving_bundle(root).manifest
    _external_attestation(
        root,
        bundle_digest=("f" * 64 if mismatch == "bundle" else manifest.bundle_digest),
        dataset_id=("e" * 64 if mismatch == "dataset" else manifest.dataset_id),
        selected_policy_digest=("d" * 64 if mismatch == "policy" else None),
    )
    with pytest.raises(ValueError, match=f"external release attestation {mismatch}"):
        load_serving_bundle(root)


def test_loader_rejects_symlink_missing_size_digest_and_undeclared_files(
    tmp_path: Path,
) -> None:
    symlink_root = create_bundle(tmp_path / "symlink")
    artifact = symlink_root / "dataset.json"
    target = tmp_path / "outside.json"
    target.write_bytes(artifact.read_bytes())
    artifact.unlink()
    try:
        artifact.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="symlink"):
        load_serving_bundle(symlink_root)

    missing_root = create_bundle(tmp_path / "missing-file")
    (missing_root / "dataset.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        load_serving_bundle(missing_root)

    size_root = create_bundle(tmp_path / "size")
    (size_root / "dataset.json").write_text("longer payload", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch"):
        load_serving_bundle(size_root)

    digest_root = create_bundle(tmp_path / "digest")
    original = (digest_root / "dataset.json").read_bytes()
    replacement = b"x" * len(original)
    (digest_root / "dataset.json").write_bytes(replacement)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_serving_bundle(digest_root)

    undeclared_root = create_bundle(tmp_path / "undeclared")
    (undeclared_root / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="undeclared"):
        load_serving_bundle(undeclared_root)


def test_loader_rejects_malformed_file_entry(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "malformed")
    path = root / "bundle.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["files"] = [1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_serving_bundle(root)
