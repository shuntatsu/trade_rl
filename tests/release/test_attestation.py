from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trade_rl.release.asymmetric import PublicVerificationKey
from trade_rl.release.attestation import ReleaseAttestation
from trade_rl.release.offline_approval import create_release_attestation
from trade_rl.release.offline_signing import public_key_bytes

NOW = datetime(2026, 7, 14, tzinfo=UTC)
PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x22" * 32)
PUBLIC_KEY = PublicVerificationKey(
    key_id="release-test-key",
    public_key=public_key_bytes(PRIVATE_KEY),
    purpose="release-verification",
    valid_from=NOW - timedelta(days=1),
    valid_until=NOW + timedelta(days=365),
)


def _attestation() -> ReleaseAttestation:
    return create_release_attestation(
        bundle_digest="a" * 64,
        dataset_id="b" * 64,
        training_run_digest=None,
        run_kind="baseline_release",
        selection_proposal_digest=None,
        selection_authorization_digest=None,
        walk_forward_run_digest=None,
        gate_evidence_digest=None,
        confirmation_evidence_digest=None,
        selected_policy_digest=None,
        git_commit="f" * 40,
        dependency_digest="1" * 64,
        approver="risk-committee",
        approved_at=NOW,
        expires_at=NOW + timedelta(days=30),
        key_id=PUBLIC_KEY.key_id,
        private_key=PRIVATE_KEY,
    )


def test_release_attestation_binds_existing_bundle_without_circular_hash() -> None:
    attestation = _attestation()
    attestation.verify({PUBLIC_KEY.key_id: PUBLIC_KEY}, trusted_at=NOW)
    assert attestation.bundle_digest == "a" * 64
    assert attestation.attestation_digest != attestation.bundle_digest
    assert not hasattr(ReleaseAttestation, "create")


def test_attestation_rejects_digest_tampering() -> None:
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(_attestation(), attestation_digest="0" * 64)


def test_attestation_rejects_signature_and_expiration() -> None:
    attestation = _attestation()
    with pytest.raises(ValueError, match="signature"):
        replace(attestation, signature="A" * 88).verify(
            {PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_at=NOW,
        )
    with pytest.raises(ValueError, match="expired"):
        attestation.verify(
            {PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_at=NOW + timedelta(days=31),
        )
