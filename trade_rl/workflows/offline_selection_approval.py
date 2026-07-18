"""Offline-only creation of signed selection authorizations."""

from __future__ import annotations

from datetime import datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trade_rl.release.offline_signing import sign_payload
from trade_rl.workflows.selection_authorization import (
    SELECTION_AUTHORIZATION_SCHEMA,
    SelectionAuthorization,
    SelectionProposal,
)

_SELECTION_PURPOSE = "selection-authorization"


def create_selection_authorization(
    proposal: SelectionProposal,
    *,
    approver: str,
    approved_at: datetime,
    expires_at: datetime,
    key_id: str,
    private_key: Ed25519PrivateKey,
) -> SelectionAuthorization:
    """Sign one immutable proposal in an offline approval context."""

    unsigned = SelectionAuthorization(
        proposal_digest=proposal.digest,
        approver=approver,
        approved_at=approved_at,
        expires_at=expires_at,
        key_id=key_id,
        signature="pending",
    )
    envelope = sign_payload(
        unsigned.signed_payload(),
        key_id=key_id,
        purpose=_SELECTION_PURPOSE,
        private_key=private_key,
        signed_at=approved_at,
    )
    if unsigned.schema_version != SELECTION_AUTHORIZATION_SCHEMA:
        raise RuntimeError("selection authorization schema mismatch")
    return SelectionAuthorization(
        proposal_digest=unsigned.proposal_digest,
        approver=unsigned.approver,
        approved_at=unsigned.approved_at,
        expires_at=unsigned.expires_at,
        key_id=envelope.key_id,
        signature=envelope.signature,
    )


__all__ = ["create_selection_authorization"]
