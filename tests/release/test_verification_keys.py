from __future__ import annotations

import pytest

from trade_rl.release import __all__ as release_exports
from trade_rl.release.signing import VerificationKey, sign_payload, verify_payload


def test_release_package_does_not_export_signing_helper() -> None:
    assert "sign_payload" not in release_exports


def test_verification_key_rejects_wrong_purpose_and_algorithm() -> None:
    with pytest.raises(ValueError, match="purpose"):
        VerificationKey(
            key_id="release-key",
            key=b"0123456789abcdef",
            purpose="release-signing",
        )
    with pytest.raises(ValueError, match="algorithm"):
        VerificationKey(
            key_id="release-key",
            key=b"0123456789abcdef",
            purpose="release-verification",
            algorithm="sha256",
        )


def test_verify_payload_requires_declared_purpose_when_requested() -> None:
    payload = {"value": 1}
    envelope = sign_payload(
        payload,
        key_id="release-key",
        signing_key=b"0123456789abcdef",
    )
    key = VerificationKey(
        key_id="release-key",
        key=b"0123456789abcdef",
        purpose="release-verification",
    )

    verify_payload(
        payload,
        envelope,
        trusted_keys={"release-key": key},
        required_purpose="release-verification",
    )

    with pytest.raises(ValueError, match="purpose"):
        verify_payload(
            payload,
            envelope,
            trusted_keys={
                "release-key": VerificationKey(
                    key_id="release-key",
                    key=b"0123456789abcdef",
                    purpose="metadata-verification",
                )
            },
            required_purpose="release-verification",
        )
