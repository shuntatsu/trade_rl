"""Authenticated canonical evidence envelopes using purpose-bound HMAC keys."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from dataclasses import dataclass

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.domain.common import require_non_empty, require_sha256

AUTHENTICATED_ENVELOPE_SCHEMA = "authenticated_evidence_hmac_sha256_v1"
HMAC_SHA256_ALGORITHM = "hmac-sha256"
_ALLOWED_VERIFICATION_PURPOSES = {
    "release-verification",
    "metadata-verification",
    "confirmation-verification",
}


def _key_bytes(value: bytes | bytearray | memoryview) -> bytes:
    key = bytes(value)
    if len(key) < 16:
        raise ValueError("signing key must contain at least 16 bytes")
    return key


@dataclass(frozen=True, slots=True)
class VerificationKey:
    """Verification-only key material with an explicit evidence purpose."""

    key_id: str
    key: bytes | bytearray | memoryview
    purpose: str
    algorithm: str = HMAC_SHA256_ALGORITHM

    def __post_init__(self) -> None:
        require_non_empty(self.key_id, field="key_id")
        if self.purpose not in _ALLOWED_VERIFICATION_PURPOSES:
            raise ValueError("verification key purpose is unsupported")
        if self.algorithm != HMAC_SHA256_ALGORITHM:
            raise ValueError("verification key algorithm is unsupported")
        object.__setattr__(self, "key", _key_bytes(self.key))


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
    """Create an envelope in an explicit offline signing context."""

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


def _verification_material(
    *,
    envelope: AuthenticatedEnvelope,
    trusted_keys: Mapping[
        str,
        VerificationKey | bytes | bytearray | memoryview,
    ],
    required_purpose: str | None,
) -> bytes:
    value = trusted_keys.get(envelope.key_id)
    if value is None:
        raise ValueError("authenticated evidence key is not trusted")
    if isinstance(value, VerificationKey):
        if value.key_id != envelope.key_id:
            raise ValueError("verification key identity does not match mapping key")
        if required_purpose is not None and value.purpose != required_purpose:
            raise ValueError("verification key purpose does not match evidence")
        return bytes(value.key)
    if required_purpose is not None:
        raise ValueError("purpose-bound verification requires VerificationKey")
    return _key_bytes(value)


def verify_payload(
    payload: object,
    envelope: AuthenticatedEnvelope,
    *,
    trusted_keys: Mapping[
        str,
        VerificationKey | bytes | bytearray | memoryview,
    ],
    required_purpose: str | None = None,
) -> None:
    key = _verification_material(
        envelope=envelope,
        trusted_keys=trusted_keys,
        required_purpose=required_purpose,
    )
    observed_digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if not hmac.compare_digest(observed_digest, envelope.payload_digest):
        raise ValueError("authenticated evidence payload digest mismatch")
    expected = hmac.new(
        key,
        _message(key_id=envelope.key_id, payload_digest=envelope.payload_digest),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, envelope.signature):
        raise ValueError("authenticated evidence signature mismatch")


__all__ = [
    "AUTHENTICATED_ENVELOPE_SCHEMA",
    "HMAC_SHA256_ALGORITHM",
    "AuthenticatedEnvelope",
    "VerificationKey",
    "sign_payload",
    "verify_payload",
]
