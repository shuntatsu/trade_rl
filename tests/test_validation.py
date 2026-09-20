from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trade_rl._validation import (
    require_aware_datetime,
    require_git_sha,
    require_non_empty,
    require_sha256,
    require_unique_non_empty,
)


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


@pytest.mark.parametrize(
    "value",
    (
        None,
        True,
        123,
        b"0" * 40,
        ["0" * 40],
        {"sha": "0" * 40},
    ),
)
def test_require_git_sha_rejects_non_strings_as_validation_error(value: object) -> None:
    with pytest.raises(ValueError, match="40-character Git SHA"):
        require_git_sha(value, field="commit")  # type: ignore[arg-type]


def test_require_git_sha_accepts_exact_lowercase_digest() -> None:
    digest = "b" * 40

    assert require_git_sha(digest, field="commit") == digest


@pytest.mark.parametrize("value", (None, True, 123, b"name", ["name"], {"name": "x"}))
def test_require_non_empty_rejects_non_strings_as_validation_error(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        require_non_empty(value, field="name")  # type: ignore[arg-type]


def test_require_non_empty_preserves_trim_semantics() -> None:
    assert require_non_empty("  value  ", field="name") == "value"


@pytest.mark.parametrize(
    "value",
    (
        None,
        True,
        123,
        "2026-01-01T00:00:00Z",
        object(),
    ),
)
def test_require_aware_datetime_rejects_non_datetimes_as_validation_error(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        require_aware_datetime(value, field="timestamp")  # type: ignore[arg-type]


def test_require_aware_datetime_accepts_timezone_aware_value() -> None:
    value = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert require_aware_datetime(value, field="timestamp") is value


def test_require_unique_non_empty_rejects_non_string_members_as_validation_error() -> (
    None
):
    values = ("first", 2)

    with pytest.raises(ValueError, match="non-empty"):
        require_unique_non_empty(values, field="names")  # type: ignore[arg-type]


def test_require_unique_non_empty_preserves_trim_and_uniqueness_semantics() -> None:
    assert require_unique_non_empty((" a ", "b"), field="names") == ("a", "b")
    with pytest.raises(ValueError, match="unique"):
        require_unique_non_empty(("a", " a "), field="names")
