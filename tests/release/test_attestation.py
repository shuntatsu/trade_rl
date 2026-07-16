from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade_rl.release.attestation import ReleaseAttestation


def test_release_attestation_binds_existing_bundle_without_circular_hash() -> None:
    attestation = ReleaseAttestation.create(
        bundle_digest="a" * 64,
        dataset_id="b" * 64,
        selection_evaluation_digest="c" * 64,
        gate_evaluation_digest="d" * 64,
        gate_evidence_digest="e" * 64,
        selected_policy_digest=None,
        git_commit="f" * 40,
        dependency_digest="1" * 64,
        approver="risk-committee",
        approved_at=datetime(2026, 7, 14, tzinfo=UTC),
        key_id="release-test-key",
        signing_key=b"release-test-signing-key",
    )
    assert attestation.bundle_digest == "a" * 64
    assert attestation.attestation_digest != attestation.bundle_digest


def test_attestation_rejects_digest_tampering() -> None:
    with pytest.raises(ValueError, match="digest mismatch"):
        ReleaseAttestation(
            attestation_digest="0" * 64,
            bundle_digest="a" * 64,
            dataset_id="b" * 64,
            selection_evaluation_digest="c" * 64,
            gate_evaluation_digest="d" * 64,
            gate_evidence_digest="e" * 64,
            selected_policy_digest=None,
            git_commit="f" * 40,
            dependency_digest="1" * 64,
            approver="risk-committee",
            approved_at=datetime(2026, 7, 14, tzinfo=UTC),
            key_id="release-test-key",
            signature="0" * 64,
        )
