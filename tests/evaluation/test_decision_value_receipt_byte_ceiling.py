"""Negative oracle for total size limits on untrusted TD8 failure receipts.

The reader must request a finite number of bytes before JSON parsing, rather
than asking the OS for all bytes of an attacker-sized regular file.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from trade_rl.evaluation import decision_value_dev_study as study


def test_strict_receipt_reader_never_uses_unbounded_file_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "incomplete-pairs.json").write_bytes(b"{}")
    requested_sizes: list[int] = []

    class ReadProbe:
        def read(self, size: int = -1) -> bytes:
            if size < 0:
                raise AssertionError(
                    "strict receipt reader requested an unbounded read"
                )
            requested_sizes.append(size)
            return b"{}"

    @contextmanager
    def checked_open(path: Path, *, field: str) -> Iterator[ReadProbe]:
        assert path == root / "incomplete-pairs.json"
        assert field == "failure receipt"
        yield ReadProbe()

    monkeypatch.setattr(study, "open_regular_binary", checked_open)
    # Once a bounded read succeeds, a malformed receipt must still fail closed
    # at schema validation rather than being admitted as financial evidence.
    with pytest.raises(ValueError, match="failure receipt fields are malformed"):
        study.load_incomplete_pair_attempt(root)

    assert requested_sizes
    assert all(0 < size <= 64 * 1024 * 1024 + 1 for size in requested_sizes)
