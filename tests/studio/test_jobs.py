from __future__ import annotations

import os
from pathlib import Path

import pytest

import trade_rl.studio.jobs as studio_jobs
from trade_rl.studio.errors import IdentityConflict, JobOwnershipLost
from trade_rl.studio.jobs import JobSupervisor

from .helpers import write_run
from .support import FakeCatalog, FakeFactory, FakeProcess, request, settings



def test_submit_training_persists_fixed_command_and_reconciles_success(
    tmp_path: Path,
) -> None:
    catalog = FakeCatalog(tmp_path)
    factory = FakeFactory()
    supervisor = JobSupervisor(
        settings(tmp_path), catalog=catalog, process_factory=factory
    )

    job = supervisor.submit_training(request(catalog))

    assert job.status == "running"
    assert job.cancellable is True
    assert job.schema_version == "studio_job_v2"
    command = factory.commands[0]
    assert command[-10:] == (
        "train",
        "run",
        "--config",
        str(catalog.config.path.resolve()),
        "--dataset",
        str(catalog.dataset.path.resolve()),
        "--output",
        str((tmp_path / "research").resolve()),
        "--run-id",
        "run-001",
    )

    write_run(tmp_path / "research", run_id="run-001", dataset_id="d" * 64)
    factory.process.exit_code = 0
    finished = supervisor.get_job(job.id)

    assert finished.status == "succeeded"
    assert finished.exit_code == 0


def test_two_supervisors_cannot_reserve_the_same_run(tmp_path: Path) -> None:
    catalog = FakeCatalog(tmp_path)
    first = JobSupervisor(
        settings(tmp_path), catalog=catalog, process_factory=FakeFactory(pid=1001)
    )
    second = JobSupervisor(
        settings(tmp_path), catalog=catalog, process_factory=FakeFactory(pid=1002)
    )

    first.submit_training(request(catalog))

    with pytest.raises(IdentityConflict, match="reserved"):
        second.submit_training(request(catalog))


def test_restart_does_not_mutate_detached_job_before_rejecting_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = FakeCatalog(tmp_path)
    first = JobSupervisor(
        settings(tmp_path), catalog=catalog, process_factory=FakeFactory()
    )
    job = first.submit_training(request(catalog))
    restarted = JobSupervisor(
        settings(tmp_path), catalog=catalog, process_factory=FakeFactory(pid=9999)
    )
    monkeypatch.setattr("trade_rl.studio.jobs._pid_matches", lambda pid, token: True)

    detached = restarted.get_job(job.id)
    assert detached.status == "running"
    assert detached.cancellable is False

    with pytest.raises(JobOwnershipLost, match="not owned"):
        restarted.cancel(job.id)

    persisted = restarted.get_job(job.id)
    assert persisted.status == "running"


def test_nonzero_worker_exit_is_failed_and_log_tail_is_bounded(tmp_path: Path) -> None:
    catalog = FakeCatalog(tmp_path)
    factory = FakeFactory()
    supervisor = JobSupervisor(
        settings(tmp_path), catalog=catalog, process_factory=factory
    )
    job = supervisor.submit_training(request(catalog))
    factory.logs[0].write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    factory.process.exit_code = 2

    failed = supervisor.get_job(job.id)
    lines, truncated = supervisor.tail_log(job.id, limit=2)

    assert failed.status == "failed"
    assert lines == ("three", "four")
    assert truncated is True


def test_cancel_owned_process_persists_cancelled_state(tmp_path: Path) -> None:
    catalog = FakeCatalog(tmp_path)
    factory = FakeFactory()
    supervisor = JobSupervisor(
        settings(tmp_path), catalog=catalog, process_factory=factory
    )
    job = supervisor.submit_training(request(catalog))

    cancelled = supervisor.cancel(job.id)

    assert factory.process.terminated is True
    assert cancelled.status == "cancelled"
    assert cancelled.completed_at is not None


def test_pid_identity_fails_closed_without_start_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(studio_jobs, "_pid_alive", lambda pid: True)

    assert studio_jobs._pid_matches(1234, None) is False


def test_process_group_options_isolate_workers_on_supported_platforms() -> None:
    assert studio_jobs._process_group_options("posix") == {"start_new_session": True}
    assert studio_jobs._process_group_options("nt") == {
        "creationflags": studio_jobs._WINDOWS_NEW_PROCESS_GROUP
    }


def test_cancel_uses_process_tree_terminator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = FakeCatalog(tmp_path)
    factory = FakeFactory()
    supervisor = JobSupervisor(
        settings(tmp_path), catalog=catalog, process_factory=factory
    )
    job = supervisor.submit_training(request(catalog))
    calls: list[int] = []

    def terminate_tree(process: FakeProcess) -> int:
        calls.append(process.pid)
        process.exit_code = -15
        return -15

    monkeypatch.setattr(
        studio_jobs,
        "_terminate_process_tree",
        terminate_tree,
        raising=False,
    )

    cancelled = supervisor.cancel(job.id)

    assert calls == [factory.process.pid]
    assert factory.process.terminated is False
    assert cancelled.status == "cancelled"
    assert cancelled.exit_code == -15


def test_process_group_options_match_current_platform() -> None:
    options = studio_jobs._process_group_options(os.name)
    if os.name == "nt":
        assert options == {"creationflags": studio_jobs._WINDOWS_NEW_PROCESS_GROUP}
    else:
        assert options == {"start_new_session": True}
