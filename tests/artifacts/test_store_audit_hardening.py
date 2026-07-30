from __future__ import annotations

from pathlib import Path

from trade_rl.artifacts.store import _atomic_write


def test_atomic_write_does_not_reuse_or_remove_a_stale_temporary_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "latest.json"
    stale = tmp_path / ".latest.json.tmp"
    stale.write_bytes(b"stale")

    _atomic_write(target, b"current")

    assert target.read_bytes() == b"current"
    assert stale.read_bytes() == b"stale"
