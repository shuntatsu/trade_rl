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
