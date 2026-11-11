"""Synthetic size-admission regression checks for verified artifact snapshots."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import trade_rl.artifacts.verified_file as verified_file


class _ObservedBytes(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = super().read(size)
        self.bytes_read += len(chunk)
        return chunk


@pytest.mark.parametrize("operation", ("read", "copy"))
def test_verified_snapshot_rejects_oversize_before_consuming_or_copying_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
) -> None:
    """A claimed one-byte artifact must not consume or publish a multi-MiB body."""

    payload = b"synthetic bytes only" * (1024 * 160)
    source = _ObservedBytes(payload)

    @contextmanager
    def fake_open_regular_binary(
        _path: Path, *, field: str
    ) -> Iterator[io.BytesIO]:
        assert field == "snapshot"
        yield source

    monkeypatch.setattr(verified_file, "open_regular_binary", fake_open_regular_binary)
    expected_digest = hashlib.sha256(payload).hexdigest()

    if operation == "read":
        with pytest.raises(ValueError, match="snapshot size mismatch"):
            verified_file.read_verified_bytes(
                tmp_path / "untrusted.bin",
                expected_digest=expected_digest,
                expected_size_bytes=1,
                field="snapshot",
            )
    else:
        private_dir = tmp_path / "preserved-private"
        private_dir.mkdir()

        @contextmanager
        def fake_temporary_directory(*, prefix: str) -> Iterator[str]:
            assert prefix == "trade-rl-verified-"
            yield str(private_dir)

        with monkeypatch.context() as patch:
            patch.setattr(
                verified_file.tempfile,
                "TemporaryDirectory",
                fake_temporary_directory,
            )
            with pytest.raises(
                ValueError, match="snapshot size changed during verified copy"
            ):
                with verified_file.verified_private_copy(
                    tmp_path / "untrusted.bin",
                    expected_digest=expected_digest,
                    expected_size_bytes=1,
                    field="snapshot",
                    filename="private.bin",
                ):
                    pytest.fail("oversize input must not yield a private copy")

        assert (private_dir / "private.bin").read_bytes() == b""

    assert source.bytes_read == 2


@pytest.mark.parametrize("operation", ("read", "copy"))
@pytest.mark.parametrize("scenario", ("exact", "short", "wrong_digest"))
def test_verified_snapshot_preserves_exact_and_negative_size_digest_contracts(
    tmp_path: Path,
    operation: str,
    scenario: str,
) -> None:
    payload = b"synthetic verified snapshot" * 29
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(payload)
    expected_size = len(payload) + (1 if scenario == "short" else 0)
    expected_digest = hashlib.sha256(
        b"some other synthetic bytes" if scenario == "wrong_digest" else payload
    ).hexdigest()

    if operation == "read":
        def invoke() -> bytes:
            return verified_file.read_verified_bytes(
                artifact,
                expected_digest=expected_digest,
                expected_size_bytes=expected_size,
                field="snapshot",
            )
    else:
        def invoke() -> bytes:
            with verified_file.verified_private_copy(
                artifact,
                expected_digest=expected_digest,
                expected_size_bytes=expected_size,
                field="snapshot",
                filename="private.bin",
            ) as private_copy:
                return private_copy.read_bytes()

    if scenario == "exact":
        assert invoke() == payload
    else:
        error = (
            "snapshot size" if scenario == "short" else "snapshot"
        )
        with pytest.raises(ValueError, match=error):
            invoke()
