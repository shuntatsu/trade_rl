"""Offline-only release approval helpers.

Runtime and registry code import verification contracts only; private signing
material is accepted solely by this module.
"""

from __future__ import annotations

from datetime import datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trade_rl.artifacts.hashing import content_digest
from trade_rl.release.attestation import (
    RELEASE_ATTESTATION_SCHEMA,
    RELEASE_PURPOSE,
    ReleaseAttestation,
)
from trade_rl.release.offline_signing import sign_payload


def create_release_attestation(
    *,
    bundle_digest: str,
    dataset_id: str,
    training_run_digest: str | None,
    run_kind: str,
    selection_proposal_digest: str | None,
    selection_authorization_digest: str | None,
    walk_forward_run_digest: str | None,
    gate_evidence_digest: str | None,
    confirmation_evidence_digest: str | None,
    selected_policy_digest: str | None,
    git_commit: str,
    dependency_digest: str,
    approver: str,
    approved_at: datetime,
    expires_at: datetime,
    key_id: str,
    private_key: Ed25519PrivateKey,
) -> ReleaseAttestation:
    """Create one externally signed release attestation."""

    payload = {
        "approved_at": approved_at,
        "approver": approver,
        "bundle_digest": bundle_digest,
        "confirmation_evidence_digest": confirmation_evidence_digest,
        "dataset_id": dataset_id,
        "dependency_digest": dependency_digest,
        "expires_at": expires_at,
        "gate_evidence_digest": gate_evidence_digest,
        "git_commit": git_commit,
        "key_id": key_id,
        "run_kind": run_kind,
        "schema_version": RELEASE_ATTESTATION_SCHEMA,
        "selected_policy_digest": selected_policy_digest,
        "selection_authorization_digest": selection_authorization_digest,
        "selection_proposal_digest": selection_proposal_digest,
        "training_run_digest": training_run_digest,
        "walk_forward_run_digest": walk_forward_run_digest,
    }
    envelope = sign_payload(
        payload,
        key_id=key_id,
        purpose=RELEASE_PURPOSE,
        private_key=private_key,
        signed_at=approved_at,
    )
    if envelope.payload_digest != content_digest(payload):
        raise RuntimeError("release signing payload digest mismatch")
    return ReleaseAttestation(
        attestation_digest=envelope.payload_digest,
        bundle_digest=bundle_digest,
        dataset_id=dataset_id,
        training_run_digest=training_run_digest,
        run_kind=run_kind,
        selection_proposal_digest=selection_proposal_digest,
        selection_authorization_digest=selection_authorization_digest,
        walk_forward_run_digest=walk_forward_run_digest,
        gate_evidence_digest=gate_evidence_digest,
        confirmation_evidence_digest=confirmation_evidence_digest,
        selected_policy_digest=selected_policy_digest,
        git_commit=git_commit,
        dependency_digest=dependency_digest,
        approver=approver,
        approved_at=approved_at,
        expires_at=expires_at,
        key_id=key_id,
        signature=envelope.signature,
    )


__all__ = ["create_release_attestation"]
