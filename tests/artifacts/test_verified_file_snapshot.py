from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
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


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "mkfifo") or not getattr(os, "O_NONBLOCK", 0),
    reason="POSIX FIFO nonblocking open required",
)
@pytest.mark.parametrize("force_portable_fallback", (False, True))
def test_verified_open_rejects_fifo_without_waiting_for_writer(
    tmp_path: Path,
    force_portable_fallback: bool,
) -> None:
    """A FIFO must be type-rejected before a blocking O_RDONLY can wait."""

    fifo = tmp_path / "untrusted-named-pipe"
    os.mkfifo(fifo)
    script = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from trade_rl.artifacts.verified_file import open_regular_binary\n"
        "if sys.argv[2] == 'fallback':\n"
        "    os.supports_dir_fd = frozenset()\n"
        "try:\n"
        "    with open_regular_binary(Path(sys.argv[1]), field='FIFO evidence'):\n"
        "        raise AssertionError('a FIFO is not a regular file')\n"
        "except ValueError as error:\n"
        "    assert 'regular file' in str(error), str(error)\n"
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(fifo),
                "fallback" if force_portable_fallback else "native",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("FIFO open blocked before the regular-file type check")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert fifo.is_fifo()


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


@pytest.mark.skipif(
    os.name != "posix"
    or not Path("/proc/self/fd").is_dir()
    or not getattr(os, "O_NOFOLLOW", 0)
    or os.open not in os.supports_dir_fd,
    reason="requires Linux-style process FD paths and POSIX descriptor no-follow",
)
@pytest.mark.parametrize("reader", ("open", "digest", "read", "private_copy"))
def test_process_fd_alias_requires_scoped_trust(tmp_path: Path, reader: str) -> None:
    """An arbitrary caller-held FD must not bypass artifact ancestor no-follow.

    The matching positive case is StudyStore's internally pinned nested stage.
    These bytes and descriptors are disposable and never reference real evidence.
    """
    bundle = tmp_path / "synthetic-source"
    bundle.mkdir()
    payload = b"synthetic untrusted process FD alias"
    (bundle / "policy.zip").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    descriptor = os.open(bundle, os.O_RDONLY | os.O_DIRECTORY)
    try:
        alias = Path("/proc/self/fd") / str(descriptor) / "policy.zip"
        assert alias.is_file(), "the test requires a live readable FD alias"
        with pytest.raises(ValueError, match="symlink|capability"):
            if reader == "open":
                with verified_file.open_regular_binary(alias, field="policy"):
                    pytest.fail("untrusted FD alias bypassed the reader")
            elif reader == "digest":
                verified_file.file_digest_and_size(alias, field="policy")
            elif reader == "read":
                verified_file.read_verified_bytes(
                    alias,
                    expected_digest=digest,
                    expected_size_bytes=len(payload),
                    field="policy",
                )
            else:
                with verified_file.verified_private_copy(
                    alias,
                    expected_digest=digest,
                    expected_size_bytes=len(payload),
                    field="policy",
                    filename="policy.zip",
                ):
                    pytest.fail("untrusted FD alias yielded a private copy")
    finally:
        os.close(descriptor)


@pytest.mark.skipif(
    os.name != "posix"
    or not getattr(os, "O_NOFOLLOW", 0)
    or not getattr(os, "O_DIRECTORY", 0)
    or os.open not in os.supports_dir_fd,
    reason="requires POSIX descriptor-relative no-follow",
)
@pytest.mark.parametrize("use_relative_path", (False, True))
def test_directory_descriptor_refuses_raw_parent_traversal_before_abspath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_relative_path: bool,
) -> None:
    """Raw symlink/.. cannot be collapsed before per-component no-follow."""
    (tmp_path / "safe").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    if use_relative_path:
        monkeypatch.chdir(tmp_path)
        source = Path("linked") / ".." / "safe"
    else:
        source = tmp_path / "linked" / ".." / "safe"

    with pytest.raises(OSError, match="parent traversal"):
        descriptor = verified_file._open_directory_descriptor_nofollow(source)
        os.close(descriptor)


@pytest.mark.skipif(
    os.name != "posix"
    or not Path("/proc/self/fd").is_dir()
    or not getattr(os, "O_NOFOLLOW", 0)
    or os.open not in os.supports_dir_fd,
    reason="requires POSIX descriptor-based process FD paths",
)
@pytest.mark.parametrize(
    "invalid_fd", ("9" * 30, "9" * 5000, "2147483648", "0" * 12 + "3")
)
def test_oversized_process_fd_alias_fails_with_safe_oserror(
    invalid_fd: str,
) -> None:
    """A numeric alias exceeding the native FD range must fail closed normally."""
    import errno

    with pytest.raises(OSError) as error:
        descriptor = verified_file._open_directory_descriptor_nofollow(
            Path("/proc/self/fd") / invalid_fd
        )
        os.close(descriptor)
    assert error.value.errno in (errno.ELOOP, errno.ENOTDIR)


@pytest.mark.skipif(
    os.name != "posix"
    or not Path("/proc/self/fd").is_dir()
    or not getattr(os, "O_NOFOLLOW", 0)
    or os.open not in os.supports_dir_fd,
    reason="requires POSIX descriptor-based process FD paths",
)
def test_reader_reports_invalid_large_fd_alias_as_parent_integrity_error() -> None:
    """A regular-file caller must not receive raw OverflowError from os.dup."""
    untrusted = Path("/proc/self/fd") / ("9" * 30) / "synthetic.json"
    with pytest.raises(ValueError, match="parent"):
        with verified_file.open_regular_binary(untrusted, field="synthetic"):
            pytest.fail("malformed FD alias bypassed safe parent admission")


@pytest.mark.skipif(
    os.name != "posix"
    or not Path("/proc/self/fd").is_dir()
    or not getattr(os, "O_NOFOLLOW", 0),
    reason="requires POSIX process FD path handling",
)
def test_trusted_fd_cannot_be_spelled_as_noncanonical_process_alias(
    tmp_path: Path,
) -> None:
    """Even a registered FD cannot bless a nonexistent zero-padded alias."""
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with verified_file._trusted_process_directory_fd(descriptor):
            with pytest.raises(OSError, match="untrusted"):
                opened = verified_file._open_directory_descriptor_nofollow(
                    Path("/proc/self/fd") / f"0{descriptor}"
                )
                os.close(opened)
    finally:
        os.close(descriptor)
