from __future__ import annotations

from pathlib import Path

import pytest

import trade_rl.artifacts.store as store_module
from trade_rl.artifacts.store import ArtifactStore, _atomic_write


def test_atomic_write_does_not_reuse_or_remove_a_stale_temporary_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "latest.json"
    stale = tmp_path / ".latest.json.tmp"
    stale.write_bytes(b"stale")

    _atomic_write(target, b"current")

    assert target.read_bytes() == b"current"
    assert stale.read_bytes() == b"stale"


def test_latest_pointer_failure_rolls_published_run_back_to_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "store")
    stage = store.stage_run("run")
    (stage / "artifact").write_bytes(b"payload")

    def fail_pointer_write(path: Path, payload: bytes) -> None:
        del path, payload
        raise OSError("pointer write failed")

    monkeypatch.setattr(store_module, "_atomic_write", fail_pointer_write)

    with pytest.raises(OSError, match="pointer write failed"):
        store.publish_run("run", validate=lambda _: True)

    assert (store.staging_root / "run").is_dir()
    assert not (store.runs_root / "run").exists()


def test_post_replace_fsync_failure_keeps_published_run_and_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import json

    import trade_rl.artifacts.atomic_pointer as pointer_module
    from trade_rl.artifacts.atomic_pointer import AtomicReplaceDurabilityError

    store = ArtifactStore(tmp_path / "store")
    stage = store.stage_run("run")
    (stage / "artifact").write_bytes(b"payload")

    monkeypatch.setattr(
        pointer_module,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("fsync failed")),
    )

    with pytest.raises(AtomicReplaceDurabilityError, match="durability"):
        store.publish_run("run", validate=lambda _: True)

    published = store.runs_root / "run"
    assert published.is_dir()
    assert not (store.staging_root / "run").exists()
    pointer = json.loads((store.root / "latest.json").read_text(encoding="utf-8"))
    assert pointer["path"] == "runs/run"
