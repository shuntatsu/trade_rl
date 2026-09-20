from __future__ import annotations

import pytest

from trade_rl._validation import require_sha256


@pytest.mark.parametrize(
    "value",
    (
        None,
        True,
        123,
        b"0" * 64,
        ["0" * 64],
        {"digest": "0" * 64},
    ),
)
def test_require_sha256_rejects_non_strings_as_validation_error(value: object) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        require_sha256(value, field="digest")  # type: ignore[arg-type]


def test_require_sha256_accepts_exact_lowercase_digest() -> None:
    digest = "a" * 64

    assert require_sha256(digest, field="digest") == digest
