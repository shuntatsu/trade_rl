"""Strict offline loading of Ed25519 private signing material."""

from __future__ import annotations

import base64
import binascii
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

_PRIVATE_KEY_SCHEMA = "ed25519_private_key_v1"
_FIELDS = {
    "algorithm",
    "key_id",
    "private_key",
    "purpose",
    "schema_version",
}


@dataclass(frozen=True, slots=True)
class OfflineSigningKey:
    key_id: str
    purpose: str
    private_key: Ed25519PrivateKey


def _string(raw: dict[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"offline signing key {field} must be a non-empty string")
    return value


def _require_private_permissions(path: Path) -> None:
    if path.is_symlink():
        raise PermissionError("offline signing key cannot be a symlink")
    if os.name != "posix":
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(
            "offline signing key permissions must deny group and others"
        )


def load_offline_signing_key(
    path: str | Path,
    *,
    required_purpose: str,
) -> OfflineSigningKey:
    """Load one raw Ed25519 key only from a private offline file."""

    source = Path(path)
    _require_private_permissions(source)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError("offline signing key is invalid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("offline signing key must be an object")
    if set(raw) != _FIELDS:
        raise ValueError("offline signing key fields are invalid")
    if _string(raw, "schema_version") != _PRIVATE_KEY_SCHEMA:
        raise ValueError("offline signing key schema is unsupported")
    if _string(raw, "algorithm") != "ed25519":
        raise ValueError("offline signing key algorithm is unsupported")
    purpose = _string(raw, "purpose")
    if purpose != required_purpose:
        raise ValueError("offline signing key purpose mismatch")
    encoded = _string(raw, "private_key")
    try:
        private_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            "offline signing key private_key must be valid base64"
        ) from exc
    if len(private_bytes) != 32:
        raise ValueError("offline signing key must contain 32 raw bytes")
    return OfflineSigningKey(
        key_id=_string(raw, "key_id"),
        purpose=purpose,
        private_key=Ed25519PrivateKey.from_private_bytes(private_bytes),
    )


__all__ = ["OfflineSigningKey", "load_offline_signing_key"]
