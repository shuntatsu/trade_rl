"""Real-filesystem failure-boundary regressions for the shared file writer."""

from pathlib import Path

import pytest

import trade_rl.artifacts.atomic_write as atomic_write


def _fixed_stage(target: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(atomic_write, "_temporary_name_suffix", lambda: "fixed")
    return target.with_name(f".{target.name}.tmp-{atomic_write.os.getpid()}-fixed")


@pytest.mark.parametrize("existing_output", [False, True])
def test_exclusive_stage_collision_preserves_foreign_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_output: bool
) -> None:
    target = tmp_path / "receipt.json"
    if existing_output:
        target.write_bytes(b"OLD")
    stage = _fixed_stage(target, monkeypatch)
    stage.write_bytes(b"OTHER_WRITER")

    with pytest.raises(FileExistsError):
        atomic_write.atomic_write_bytes(target, b"NEW")

    assert stage.read_bytes() == b"OTHER_WRITER"
    if existing_output:
        assert target.read_bytes() == b"OLD"
    else:
        assert not target.exists()


@pytest.mark.parametrize("directory_sync_fails", [False, True])
def test_published_write_does_not_unlink_reoccupied_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory_sync_fails: bool
) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"OLD")
    stage = _fixed_stage(target, monkeypatch)
    replace = atomic_write.os.replace

    def replace_and_reoccupy(source: Path, destination: Path) -> None:
        replace(source, destination)
        assert source == stage
        stage.write_bytes(b"OTHER_WRITER")

    def sync_directory(path: Path) -> None:
        assert path == tmp_path
        if directory_sync_fails:
            raise OSError("injected post-replace directory sync failure")

    monkeypatch.setattr(atomic_write.os, "replace", replace_and_reoccupy)
    monkeypatch.setattr(atomic_write, "_fsync_directory", sync_directory)
    if directory_sync_fails:
        with pytest.raises(OSError, match="post-replace directory sync failure"):
            atomic_write.atomic_write_bytes(target, b"NEW")
    else:
        assert atomic_write.atomic_write_bytes(target, b"NEW") == target

    assert target.read_bytes() == b"NEW"
    assert stage.read_bytes() == b"OTHER_WRITER"


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_prepublication_failure_cleans_owned_stage_and_preserves_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"OLD")
    stage = _fixed_stage(target, monkeypatch)

    def fail(*_args: object) -> None:
        raise OSError("injected prepublication failure")

    monkeypatch.setattr(atomic_write.os, failure, fail)
    with pytest.raises(OSError, match="prepublication failure"):
        atomic_write.atomic_write_bytes(target, b"NEW")

    assert target.read_bytes() == b"OLD"
    assert not stage.exists()
