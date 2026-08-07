"""Offline-only Ed25519 key generation and detached signing helpers."""

from __future__ import annotations

import base64
from datetime import datetime

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.release.asymmetric import SignedEvidenceEnvelope

_SIGNATURE_BYTES = 64


def generate_private_key() -> Ed25519PrivateKey:
    """Generate a private key inside an explicit offline signing process."""

    return Ed25519PrivateKey.generate()


def public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    """Derive raw public verification bytes from one offline private key."""

    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("private_key must be an Ed25519PrivateKey")
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def sign_payload(
    payload: object,
    *,
    key_id: str,
    purpose: str,
    private_key: Ed25519PrivateKey,
    signed_at: datetime,
) -> SignedEvidenceEnvelope:
    """Sign canonical evidence in an explicit offline signing context."""

    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("private_key must be an Ed25519PrivateKey")
    unsigned = SignedEvidenceEnvelope(
        key_id=key_id,
        purpose=purpose,
        payload_digest=content_digest(payload),
        signed_at=signed_at,
        signature=base64.b64encode(b"\0" * _SIGNATURE_BYTES).decode("ascii"),
    )
    signature = private_key.sign(canonical_json_bytes(unsigned.signing_payload()))
    return SignedEvidenceEnvelope(
        key_id=unsigned.key_id,
        purpose=unsigned.purpose,
        payload_digest=unsigned.payload_digest,
        signed_at=unsigned.signed_at,
        signature=base64.b64encode(signature).decode("ascii"),
    )


__all__ = ["generate_private_key", "public_key_bytes", "sign_payload"]
