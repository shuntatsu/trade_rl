from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade_rl.release import __all__ as release_exports
from trade_rl.release.asymmetric import (
    PublicVerificationKey,
    verify_signed_payload,
)
from trade_rl.release.offline_signing import (
    generate_private_key,
    public_key_bytes,
    sign_payload,
)

NOW = datetime(2026, 7, 18, tzinfo=UTC)


def test_release_package_does_not_export_signing_helper() -> None:
    assert "sign_payload" not in release_exports


def test_verification_key_rejects_invalid_window_and_algorithm() -> None:
    private_key = generate_private_key()
    with pytest.raises(ValueError, match="valid_until"):
        PublicVerificationKey(
            key_id="release-key",
            public_key=public_key_bytes(private_key),
            purpose="release-verification",
            valid_from=NOW,
            valid_until=NOW,
        )
    with pytest.raises(ValueError, match="algorithm"):
        PublicVerificationKey(
            key_id="release-key",
            public_key=public_key_bytes(private_key),
            purpose="release-verification",
            valid_from=NOW,
            valid_until=NOW + timedelta(days=1),
            algorithm="rsa",
        )


def test_verify_payload_requires_declared_purpose() -> None:
    private_key = generate_private_key()
    payload = {"value": 1}
    envelope = sign_payload(
        payload,
        key_id="release-key",
        purpose="release-verification",
        private_key=private_key,
        signed_at=NOW,
    )
    key = PublicVerificationKey(
        key_id="release-key",
        public_key=public_key_bytes(private_key),
        purpose="release-verification",
        valid_from=NOW - timedelta(days=1),
        valid_until=NOW + timedelta(days=1),
    )
    verify_signed_payload(
        payload,
        envelope,
        trusted_keys={"release-key": key},
        trusted_at=NOW,
        required_purpose="release-verification",
    )
    with pytest.raises(ValueError, match="purpose"):
        verify_signed_payload(
            payload,
            envelope,
            trusted_keys={"release-key": key},
            trusted_at=NOW,
            required_purpose="metadata-verification",
        )
