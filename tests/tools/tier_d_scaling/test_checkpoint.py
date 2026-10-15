"""Profile checkpoints preserve completed calls and publish atomically."""

from __future__ import annotations

import cProfile
import importlib
import marshal
import pstats
from pathlib import Path
from typing import Any

import pytest


def _checkpoint_module() -> Any:
    return importlib.import_module("tools.tier_d_scaling.checkpoint")


def _entry(stats: pstats.Stats, function: str) -> tuple[Any, ...]:
    entries = getattr(stats, "stats")
    return next(entry for key, entry in entries.items() if key[2] == function)


def test_checkpoint_is_loadable_and_leaves_the_active_profile_running(
    tmp_path: Path,
) -> None:
    module = _checkpoint_module()
    profile = cProfile.Profile()
    checkpoint = tmp_path / "snapshot.prof"

    def nested_child() -> int:
        return sum(range(1000))

    def nested_parent() -> int:
        return nested_child()

    def after_checkpoint() -> int:
        return sum(range(100))

    def active_frame() -> None:
        nested_parent()
        module.publish_checkpoint(profile, checkpoint)
        after_checkpoint()

    profile.enable()
    active_frame()
    profile.disable()
    profile.snapshot_stats()

    captured = pstats.Stats(str(checkpoint))
    parent_call = _entry(captured, "nested_parent")
    child_call = _entry(captured, "nested_child")
    assert parent_call[1] == 1
    assert parent_call[3] > 0
    assert child_call[1] == 1
    assert child_call[3] > 0

    final_parent_call = _entry(pstats.Stats(profile), "active_frame")
    final_post_checkpoint_call = _entry(pstats.Stats(profile), "after_checkpoint")
    assert final_parent_call[1] == 1
    assert final_post_checkpoint_call[1] == 1


def test_missing_parent_is_not_created(tmp_path: Path) -> None:
    module = _checkpoint_module()
    profile = cProfile.Profile()
    destination = tmp_path / "missing" / "snapshot.prof"

    with pytest.raises(FileNotFoundError):
        module.publish_checkpoint(profile, destination)

    assert not destination.parent.exists()


def test_existing_target_is_not_overwritten(tmp_path: Path) -> None:
    module = _checkpoint_module()
    profile = cProfile.Profile()
    destination = tmp_path / "snapshot.prof"
    original = b"existing checkpoint bytes"
    destination.write_bytes(original)

    with pytest.raises(FileExistsError):
        module.publish_checkpoint(profile, destination)

    assert destination.read_bytes() == original


def test_existing_symlink_target_is_not_replaced(tmp_path: Path) -> None:
    module = _checkpoint_module()
    profile = cProfile.Profile()
    target = tmp_path / "target.prof"
    destination = tmp_path / "snapshot.prof"
    original = b"symlink target bytes"
    target.write_bytes(original)
    try:
        destination.symlink_to(target)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(FileExistsError):
        module.publish_checkpoint(profile, destination)

    assert destination.is_symlink()
    assert target.read_bytes() == original


def test_marshal_failure_cleans_staging_file_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _checkpoint_module()
    profile = cProfile.Profile()
    destination = tmp_path / "snapshot.prof"
    staging: list[Path] = []

    def fail_dump(value: Any, stream: Any) -> None:
        staging.append(Path(stream.name))
        raise OSError("marshal failure")

    monkeypatch.setattr(module.marshal, "dump", fail_dump)
    with pytest.raises(OSError, match="marshal failure"):
        module.publish_checkpoint(profile, destination)

    assert not destination.exists()
    assert len(staging) == 1
    assert staging[0].parent == tmp_path
    assert not staging[0].exists()


def test_link_failure_cleans_staging_file_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _checkpoint_module()
    profile = cProfile.Profile()
    profile.run("sum(range(100))")
    destination = tmp_path / "snapshot.prof"
    staging: list[Path] = []

    def fail_link(source: Any, target: Any) -> None:
        staging.append(Path(source))
        assert Path(target) == destination
        with Path(source).open("rb") as stream:
            assert isinstance(marshal.load(stream), dict)
        raise OSError("link failure")

    monkeypatch.setattr(module.os, "link", fail_link)
    with pytest.raises(OSError, match="link failure"):
        module.publish_checkpoint(profile, destination)

    assert not destination.exists()
    assert len(staging) == 1
    assert staging[0].parent == tmp_path
    assert not staging[0].exists()
