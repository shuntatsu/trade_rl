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
