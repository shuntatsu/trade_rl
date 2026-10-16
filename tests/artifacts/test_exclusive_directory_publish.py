from __future__ import annotations

from pathlib import Path

import pytest

import trade_rl.artifacts.atomic_write as atomic_write


def test_atomic_directory_publish_preserves_target_created_after_precheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "published"
    staging.mkdir()
    payload = staging / "receipt.json"
    payload.write_bytes(b'{"complete":true}\n')
    payload_before = payload.read_bytes()

    original_is_symlink = Path.is_symlink
    competitor_identity: tuple[int, int] | None = None
    intercepted = False

    def create_competitor_after_negative_check(path: Path) -> bool:
        nonlocal competitor_identity, intercepted
        observed = original_is_symlink(path)
        if path == target and not intercepted:
            assert observed is False
            target.mkdir()
            stat = target.stat()
            competitor_identity = (stat.st_dev, stat.st_ino)
            intercepted = True
        return observed

    monkeypatch.setattr(Path, "is_symlink", create_competitor_after_negative_check)

    try:
        with pytest.raises(FileExistsError):
            atomic_write.atomic_rename_directory(staging, target)
    finally:
        assert intercepted
        assert competitor_identity is not None
        stat = target.stat()
        assert (stat.st_dev, stat.st_ino) == competitor_identity
        assert target.is_dir()
        assert not tuple(target.iterdir())
        assert staging.is_dir()
        assert payload.read_bytes() == payload_before
