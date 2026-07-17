"""Offline-only release approval helpers.

Runtime and registry code must import verification contracts, never this module.
"""

from __future__ import annotations

from datetime import datetime

from trade_rl.release.attestation import ReleaseAttestation


def create_release_attestation(
    *,
    bundle_digest: str,
    dataset_id: str,
    selection_evaluation_digest: str,
    gate_evaluation_digest: str,
    gate_evidence_digest: str,
    selected_policy_digest: str | None,
    git_commit: str,
    dependency_digest: str,
    approver: str,
    approved_at: datetime,
    key_id: str,
    signing_key: bytes | bytearray | memoryview,
) -> ReleaseAttestation:
    """Create one release attestation in a separately controlled process."""

    return ReleaseAttestation.create(
        bundle_digest=bundle_digest,
        dataset_id=dataset_id,
        selection_evaluation_digest=selection_evaluation_digest,
        gate_evaluation_digest=gate_evaluation_digest,
        gate_evidence_digest=gate_evidence_digest,
        selected_policy_digest=selected_policy_digest,
        git_commit=git_commit,
        dependency_digest=dependency_digest,
        approver=approver,
        approved_at=approved_at,
        key_id=key_id,
        signing_key=signing_key,
    )


__all__ = ["create_release_attestation"]
