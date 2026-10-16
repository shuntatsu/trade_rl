from __future__ import annotations

import os
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


@pytest.mark.parametrize(
    "source_kind", ["missing", "regular_file", "directory_symlink"]
)
def test_atomic_directory_publish_rejects_invalid_source_before_native_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "published"
    file_bytes = b"this is not a directory"
    referent = tmp_path / "directory-referent"
    referent_bytes = b"preserve the symlink referent"
    link_target: Path | None = None

    if source_kind == "regular_file":
        staging.write_bytes(file_bytes)
    elif source_kind == "directory_symlink":
        referent.mkdir()
        (referent / "payload.bin").write_bytes(referent_bytes)
        try:
            staging.symlink_to(referent, target_is_directory=True)
        except OSError as error:
            if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                pytest.skip("Windows symlink creation privilege is unavailable")
            raise
        link_target = staging.readlink()
    else:
        assert source_kind == "missing"

    def fail_if_native_rename_is_reached(source: Path, destination: Path) -> None:
        pytest.fail(
            f"native directory rename unexpectedly reached for {source} -> {destination}"
        )

    monkeypatch.setattr(
        atomic_write, "_rename_directory_noreplace", fail_if_native_rename_is_reached
    )

    with pytest.raises(ValueError) as error:
        atomic_write.atomic_rename_directory(staging, target)

    assert (
        str(error.value) == "directory publication source must be a regular directory"
    )
    assert not target.exists()
    assert not target.is_symlink()
    if source_kind == "missing":
        assert not staging.exists()
        assert not staging.is_symlink()
    elif source_kind == "regular_file":
        assert staging.is_file()
        assert not staging.is_symlink()
        assert staging.read_bytes() == file_bytes
    else:
        assert source_kind == "directory_symlink"
        assert link_target is not None
        assert staging.is_symlink()
        assert staging.readlink() == link_target
        assert staging.resolve() == referent
        assert (referent / "payload.bin").read_bytes() == referent_bytes
