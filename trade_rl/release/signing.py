"""Authenticated canonical evidence envelopes using trusted HMAC-SHA256 keys."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from dataclasses import dataclass

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.domain.common import require_non_empty, require_sha256

AUTHENTICATED_ENVELOPE_SCHEMA = "authenticated_evidence_hmac_sha256_v1"


def _key_bytes(value: bytes | bytearray | memoryview) -> bytes:
    key = bytes(value)
    if len(key) < 16:
        raise ValueError("signing key must contain at least 16 bytes")
    return key


def _message(*, key_id: str, payload_digest: str) -> bytes:
    return canonical_json_bytes(
        {
            "key_id": key_id,
            "payload_digest": payload_digest,
            "schema_version": AUTHENTICATED_ENVELOPE_SCHEMA,
        }
    )


@dataclass(frozen=True, slots=True)
class AuthenticatedEnvelope:
    key_id: str
    payload_digest: str
    signature: str
    schema_version: str = AUTHENTICATED_ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        require_non_empty(self.key_id, field="key_id")
        require_sha256(self.payload_digest, field="payload_digest")
        require_sha256(self.signature, field="signature")
        if self.schema_version != AUTHENTICATED_ENVELOPE_SCHEMA:
            raise ValueError("unsupported authenticated evidence schema")


def sign_payload(
    payload: object,
    *,
    key_id: str,
    signing_key: bytes | bytearray | memoryview,
) -> AuthenticatedEnvelope:
    require_non_empty(key_id, field="key_id")
    payload_digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    signature = hmac.new(
        _key_bytes(signing_key),
        _message(key_id=key_id, payload_digest=payload_digest),
        hashlib.sha256,
    ).hexdigest()
    return AuthenticatedEnvelope(
        key_id=key_id,
        payload_digest=payload_digest,
        signature=signature,
    )


def verify_payload(
    payload: object,
    envelope: AuthenticatedEnvelope,
    *,
    trusted_keys: Mapping[str, bytes | bytearray | memoryview],
) -> None:
    key = trusted_keys.get(envelope.key_id)
    if key is None:
        raise ValueError("authenticated evidence key is not trusted")
    observed_digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if not hmac.compare_digest(observed_digest, envelope.payload_digest):
        raise ValueError("authenticated evidence payload digest mismatch")
    expected = hmac.new(
        _key_bytes(key),
        _message(key_id=envelope.key_id, payload_digest=envelope.payload_digest),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, envelope.signature):
        raise ValueError("authenticated evidence signature mismatch")


__all__ = [
    "AUTHENTICATED_ENVELOPE_SCHEMA",
    "AuthenticatedEnvelope",
    "sign_payload",
    "verify_payload",
]
