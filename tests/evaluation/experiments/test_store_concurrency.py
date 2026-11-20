from __future__ import annotations

import multiprocessing
from pathlib import Path

from trade_rl.evaluation.experiments.store import StudyStore


def _hold_lock(root_text: str, ready, release) -> None:
    root = Path(root_text)
    store = StudyStore(root)
    with store.mutation_lock():
        (root / "lock-order.txt").write_text("holder\n", encoding="utf-8")
        ready.set()
        if not release.wait(10.0):
            raise RuntimeError("parent did not release holder")


def _wait_for_lock(root_text: str, acquired) -> None:
    root = Path(root_text)
    store = StudyStore(root)
    with store.mutation_lock():
        with (root / "lock-order.txt").open("a", encoding="utf-8") as handle:
            handle.write("waiter\n")
        acquired.set()


def test_mutation_lock_serializes_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "study"
    ready = context.Event()
    release = context.Event()
    acquired = context.Event()

    holder = context.Process(target=_hold_lock, args=(str(root), ready, release))
    waiter = context.Process(target=_wait_for_lock, args=(str(root), acquired))

    holder.start()
    assert ready.wait(10.0)
    waiter.start()
    assert not acquired.wait(0.3)

    release.set()
    assert acquired.wait(10.0)

    holder.join(10.0)
    waiter.join(10.0)
    assert holder.exitcode == 0
    assert waiter.exitcode == 0
    assert (root / "lock-order.txt").read_text(encoding="utf-8").splitlines() == [
        "holder",
        "waiter",
    ]


def _reader_checks_mutation_window(root_text: str, outcome) -> None:
    from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError

    store = StudyStore(root_text)
    try:
        payload = store.read_json("plan.json")
    except ArtifactIntegrityError:
        outcome.put("refused")
    else:
        outcome.put(payload)


def test_other_process_study_reader_refuses_active_mutation_lock(
    tmp_path: Path,
) -> None:
    """An unrelated reader process must fail closed while the writer holds lock."""
    import os

    import pytest

    if os.name != "posix":
        pytest.skip("POSIX nonblocking shared locks required")
    root = tmp_path / "study"
    writer = StudyStore(root)
    writer.publish_json_once("plan.json", {"committed": True})
    ctx = multiprocessing.get_context("spawn")
    outcome = ctx.Queue()
    reader = ctx.Process(
        target=_reader_checks_mutation_window, args=(str(root), outcome)
    )
    with writer.mutation_lock():
        reader.start()
        assert outcome.get(timeout=10) == "refused"
        reader.join(timeout=10)
        assert reader.exitcode == 0

    assert StudyStore(root).read_json("plan.json") == {"committed": True}


def _pause_writer_after_directory_rename(root_text: str, ready, release) -> None:
    from trade_rl.evaluation.experiments import store as store_module

    writer = StudyStore(root_text)
    actual_rename = store_module.atomic_rename_directory

    def pause_after_rename(source: str | Path, target: str | Path) -> Path:
        published = actual_rename(source, target)
        ready.set()
        if not release.wait(10.0):
            raise RuntimeError("publisher was not released before inode verification")
        return published

    store_module.atomic_rename_directory = pause_after_rename
    writer.publish_directory_once(
        "evidence/synthetic",
        lambda stage: (stage / "genuine.json").write_text(
            '{"committed":true}', encoding="utf-8"
        ),
    )


def test_cross_process_reader_rejects_interleaved_uncommitted_directory(
    tmp_path: Path,
) -> None:
    """Read during post-rename/pre-check window must fail across processes."""
    import os

    import pytest

    if os.name != "posix":
        pytest.skip("POSIX nonblocking shared locks required")
    root = tmp_path / "study"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    writer = context.Process(
        target=_pause_writer_after_directory_rename,
        args=(str(root), ready, release),
    )
    writer.start()
    try:
        assert ready.wait(10.0), "the writer never reached the uncommitted window"
        from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError

        with pytest.raises(ArtifactIntegrityError, match="publication"):
            StudyStore(root).read_json("evidence/synthetic/genuine.json")
    finally:
        release.set()
        writer.join(timeout=10.0)
    assert writer.exitcode == 0
    assert StudyStore(root).read_json("evidence/synthetic/genuine.json") == {
        "committed": True
    }
