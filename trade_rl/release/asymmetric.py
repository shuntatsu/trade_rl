"""Ed25519-signed, purpose-bound evidence envelopes.

Runtime and trainer processes receive public verification material only. Private
key generation and signing live in :mod:`trade_rl.release.offline_signing`.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)

SIGNED_EVIDENCE_SCHEMA = "signed_evidence_ed25519_v1"
ED25519_ALGORITHM = "ed25519"
_SIGNATURE_BYTES = 64
_PUBLIC_KEY_BYTES = 32


def _utc(value: datetime, *, field: str) -> datetime:
    return require_aware_datetime(value, field=field).astimezone(UTC)


def _strict_b64(value: str, *, field: str, expected_size: int) -> bytes:
    require_non_empty(value, field=field)
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{field} must be valid base64") from error
    if len(decoded) != expected_size:
        raise ValueError(f"{field} has an invalid byte length")
    return decoded


@dataclass(frozen=True, slots=True)
class PublicVerificationKey:
    """Public verification material with purpose and validity boundaries."""

    key_id: str
    public_key: bytes | bytearray | memoryview
    purpose: str
    valid_from: datetime
    valid_until: datetime
    algorithm: str = ED25519_ALGORITHM

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "key_id", require_non_empty(self.key_id, field="key_id")
        )
        object.__setattr__(
            self, "purpose", require_non_empty(self.purpose, field="purpose")
        )
        raw = bytes(self.public_key)
        if len(raw) != _PUBLIC_KEY_BYTES:
            raise ValueError("public_key must contain 32 raw Ed25519 bytes")
        object.__setattr__(self, "public_key", raw)
        start = _utc(self.valid_from, field="valid_from")
        end = _utc(self.valid_until, field="valid_until")
        if end <= start:
            raise ValueError("verification key valid_until must be after valid_from")
        object.__setattr__(self, "valid_from", start)
        object.__setattr__(self, "valid_until", end)
        if self.algorithm != ED25519_ALGORITHM:
            raise ValueError("verification key algorithm is unsupported")

    def verifier(self) -> Ed25519PublicKey:
        return Ed25519PublicKey.from_public_bytes(bytes(self.public_key))


@dataclass(frozen=True, slots=True)
class SignedEvidenceEnvelope:
    """Detached signature over a canonical payload digest and signing context."""

    key_id: str
    purpose: str
    payload_digest: str
    signed_at: datetime
    signature: str
    schema_version: str = SIGNED_EVIDENCE_SCHEMA
    algorithm: str = ED25519_ALGORITHM

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "key_id", require_non_empty(self.key_id, field="key_id")
        )
        object.__setattr__(
            self, "purpose", require_non_empty(self.purpose, field="purpose")
        )
        require_sha256(self.payload_digest, field="payload_digest")
        object.__setattr__(self, "signed_at", _utc(self.signed_at, field="signed_at"))
        _strict_b64(self.signature, field="signature", expected_size=_SIGNATURE_BYTES)
        if self.schema_version != SIGNED_EVIDENCE_SCHEMA:
            raise ValueError("unsupported signed evidence schema")
        if self.algorithm != ED25519_ALGORITHM:
            raise ValueError("signed evidence algorithm is unsupported")

    def signing_payload(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "payload_digest": self.payload_digest,
            "purpose": self.purpose,
            "schema_version": self.schema_version,
            "signed_at": self.signed_at,
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self.signing_payload(), "signature": self.signature}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> SignedEvidenceEnvelope:
        try:
            return cls(
                key_id=_strict_string(raw, "key_id"),
                purpose=_strict_string(raw, "purpose"),
                payload_digest=_strict_string(raw, "payload_digest"),
                signed_at=_parse_datetime(raw, "signed_at"),
                signature=_strict_string(raw, "signature"),
                schema_version=_strict_string(raw, "schema_version"),
                algorithm=_strict_string(raw, "algorithm"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("signed evidence envelope is invalid") from error


def _strict_string(raw: Mapping[str, object], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _parse_datetime(raw: Mapping[str, object], field: str) -> datetime:
    value = _strict_string(raw, field)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from error


def verify_signed_payload(
    payload: object,
    envelope: SignedEvidenceEnvelope,
    *,
    trusted_keys: Mapping[str, PublicVerificationKey],
    trusted_at: datetime,
    required_purpose: str,
) -> None:
    """Verify payload identity, signer purpose, key validity and signature."""

    now = _utc(trusted_at, field="trusted_at")
    if envelope.purpose != required_purpose:
        raise ValueError("signed evidence purpose mismatch")
    key = trusted_keys.get(envelope.key_id)
    if key is None:
        raise ValueError("signed evidence key is not trusted")
    if key.key_id != envelope.key_id:
        raise ValueError("verification key identity does not match mapping key")
    if key.purpose != required_purpose:
        raise ValueError("verification key purpose does not match evidence")
    if key.algorithm != envelope.algorithm:
        raise ValueError("verification key algorithm does not match evidence")
    if not key.valid_from <= envelope.signed_at <= key.valid_until:
        raise ValueError("signed evidence was created outside key validity")
    if now < envelope.signed_at:
        raise ValueError("signed evidence is from the future")
    if now > key.valid_until:
        raise ValueError("verification key is expired at trusted time")
    if content_digest(payload) != envelope.payload_digest:
        raise ValueError("signed evidence payload digest mismatch")
    signature = _strict_b64(
        envelope.signature,
        field="signature",
        expected_size=_SIGNATURE_BYTES,
    )
    try:
        key.verifier().verify(
            signature,
            canonical_json_bytes(envelope.signing_payload()),
        )
    except InvalidSignature as error:
        raise ValueError("signed evidence signature mismatch") from error


def load_public_verification_keys(path: str | Path) -> dict[str, PublicVerificationKey]:
    """Load a strict public-key trust store from canonical JSON."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("public verification key store must be an object")
    keys_raw = raw.get("keys")
    if raw.get(
        "schema_version"
    ) != "public_verification_key_store_v1" or not isinstance(keys_raw, list):
        raise ValueError("public verification key store schema is invalid")
    result: dict[str, PublicVerificationKey] = {}
    for index, item in enumerate(keys_raw):
        if not isinstance(item, dict):
            raise ValueError(f"keys[{index}] must be an object")
        try:
            key_id = _strict_string(item, "key_id")
            public_key_text = _strict_string(item, "public_key")
            key = PublicVerificationKey(
                key_id=key_id,
                public_key=_strict_b64(
                    public_key_text,
                    field=f"keys[{index}].public_key",
                    expected_size=_PUBLIC_KEY_BYTES,
                ),
                purpose=_strict_string(item, "purpose"),
                valid_from=_parse_datetime(item, "valid_from"),
                valid_until=_parse_datetime(item, "valid_until"),
                algorithm=_strict_string(item, "algorithm"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"public verification key keys[{index}] is invalid"
            ) from error
        if key_id in result:
            raise ValueError("public verification key IDs must be unique")
        result[key_id] = key
    if not result:
        raise ValueError("public verification key store must not be empty")
    return result


__all__ = [
    "ED25519_ALGORITHM",
    "SIGNED_EVIDENCE_SCHEMA",
    "PublicVerificationKey",
    "SignedEvidenceEnvelope",
    "load_public_verification_keys",
    "verify_signed_payload",
]
