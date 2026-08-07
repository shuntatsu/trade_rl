"""Content digests for canonical artifacts."""

from __future__ import annotations

from hashlib import sha256

from trade_rl.artifacts.codec import canonical_json_bytes


def content_digest(value: object) -> str:
    """Return the lowercase SHA-256 digest of canonical JSON content."""

    return sha256(canonical_json_bytes(value)).hexdigest()
