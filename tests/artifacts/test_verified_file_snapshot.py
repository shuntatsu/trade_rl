from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import trade_rl.artifacts.verified_file as verified_file


def test_file_digest_and_size_use_one_open_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"different bytes on path")
    snapshot = b"single opened snapshot"

    @contextmanager
    def fake_open_regular_binary(
        _path: Path,
        *,
        field: str,
    ) -> Iterator[io.BytesIO]:
        assert field == "test artifact"
        yield io.BytesIO(snapshot)

    monkeypatch.setattr(
        verified_file,
        "open_regular_binary",
        fake_open_regular_binary,
    )

    digest, size = verified_file.file_digest_and_size(
        path,
        field="test artifact",
    )

    assert digest == hashlib.sha256(snapshot).hexdigest()
    assert size == len(snapshot)


@pytest.mark.parametrize("bad_digest", (None, 123, True, b"0" * 64))
def test_verified_private_copy_rejects_non_string_digest_as_validation_error(
    tmp_path: Path,
    bad_digest: object,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"artifact")

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        with verified_file.verified_private_copy(
            path,
            expected_digest=bad_digest,  # type: ignore[arg-type]
            field="test artifact",
            filename="artifact.bin",
        ):
            pytest.fail("invalid digest must fail before yielding a private copy")


@pytest.mark.parametrize("reader", ("open", "digest", "read", "private_copy"))
@pytest.mark.parametrize("parent_kind", ("top", "nested"))
def test_verified_artifact_read_rejects_symlinked_directory_ancestor(
    tmp_path: Path,
    reader: str,
    parent_kind: str,
) -> None:
    """A regular final file must not be trusted through a linked parent."""

    outside = tmp_path / "untrusted"
    outside.mkdir()
    payload = b"independent synthetic artifact bytes"
    (outside / "policy.zip").write_bytes(payload)

    if parent_kind == "top":
        linked_parent = tmp_path / "bundle-link"
        source = linked_parent / "policy.zip"
    else:
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        linked_parent = bundle / "policy-link"
        source = linked_parent / "policy.zip"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"directory symlinks unavailable on this filesystem: {error}")

    digest = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="symlink"):
        if reader == "open":
            with verified_file.open_regular_binary(source, field="policy"):
                pytest.fail("a symlinked parent must be refused before reading")
        elif reader == "digest":
            verified_file.file_digest_and_size(source, field="policy")
        elif reader == "read":
            verified_file.read_verified_bytes(
                source,
                expected_digest=digest,
                expected_size_bytes=len(payload),
                field="policy",
            )
        else:
            with verified_file.verified_private_copy(
                source,
                expected_digest=digest,
                expected_size_bytes=len(payload),
                field="policy",
                filename="policy.zip",
            ):
                pytest.fail("a symlinked parent must not yield a private copy")


def test_verified_artifact_read_keeps_regular_nested_parents_working(
    tmp_path: Path,
) -> None:
    """The hardened reader does not modify valid artifact bytes or hashes."""

    bundle = tmp_path / "bundle" / "nested"
    bundle.mkdir(parents=True)
    policy = bundle / "policy.zip"
    payload = b"literal synthetic policy, not a model"
    policy.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    with verified_file.open_regular_binary(policy, field="policy") as handle:
        assert handle.read() == payload
    assert verified_file.file_digest_and_size(policy, field="policy") == (
        digest,
        len(payload),
    )
    assert (
        verified_file.read_verified_bytes(
            policy,
            expected_digest=digest,
            expected_size_bytes=len(payload),
            field="policy",
        )
        == payload
    )
    with verified_file.verified_private_copy(
        policy,
        expected_digest=digest,
        expected_size_bytes=len(payload),
        field="policy",
        filename="policy.zip",
    ) as private_copy:
        assert private_copy.read_bytes() == payload
        assert private_copy != policy


def test_verified_artifact_read_rejects_parent_traversal(tmp_path: Path) -> None:
    """Dot-dot must not bypass the symlink-component check."""

    root = tmp_path / "bundle"
    root.mkdir()
    outside = tmp_path / "untrusted"
    outside.mkdir()
    (outside / "policy.zip").write_bytes(b"synthetic")
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"directory symlinks unavailable on this filesystem: {error}")

    with pytest.raises(ValueError, match="parent traversal"):
        with verified_file.open_regular_binary(
            link / ".." / "policy.zip",
            field="policy",
        ):
            pytest.fail("parent traversal must fail closed")
