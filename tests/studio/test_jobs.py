from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.studio.contracts import ConfigSummary, DatasetSummary, TrainingJobRequest
from trade_rl.studio.errors import IdentityConflict, JobOwnershipLost, ResourceNotFound
from trade_rl.studio.jobs import JobSupervisor
from trade_rl.studio.resource_ids import resource_id

from .helpers import write_run
from .test_catalog import settings


class FakeProcess:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.exit_code: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True
        self.exit_code = -15

    def kill(self) -> None:
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0 if self.exit_code is None else self.exit_code


class FakeFactory:
    def __init__(self, *, pid: int = 4242) -> None:
        self.process = FakeProcess(pid)
        self.commands: list[tuple[str, ...]] = []
        self.logs: list[Path] = []
        self.cwds: list[Path] = []

    def __call__(
        self, command: tuple[str, ...], *, cwd: Path, log_path: Path
    ) -> FakeProcess:
        self.commands.append(command)
        self.logs.append(log_path)
        self.cwds.append(cwd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("started\n", encoding="utf-8")
        return self.process


class FakeCatalog:
    def __init__(self, root: Path) -> None:
        config_path = root / "configs" / "training.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{}", encoding="utf-8")
        dataset_path = root / "datasets" / "btc"
        dataset_path.mkdir(parents=True, exist_ok=True)
        self.config = SimpleNamespace(
            path=config_path,
            summary=ConfigSummary(
                id=resource_id("config", "configs/training.json", "c" * 64),
                config_digest="c" * 64,
                name="training",
                relative_path="configs/training.json",
                algorithm="ppo",
                status="VALID",
            ),
        )
        self.dataset = SimpleNamespace(
            path=dataset_path,
            summary=DatasetSummary(
                id=resource_id("dataset", "datasets/btc", "d" * 64),
                dataset_id="d" * 64,
                name="btc",
                relative_path="datasets/btc",
                market="continuous",
                symbols=("BTCUSDT",),
                timeframes=("1h",),
                range="2026-01-01 — 2026-01-02",
                status="VALID",
                feature_count=1,
                bar_count=12,
                symbol_count=1,
                updated="2026-01-01T00:00:00+00:00",
            ),
        )

    def resolve_config(self, value: str):
        if value != self.config.summary.id:
            raise ResourceNotFound(value)
        return self.config

    def resolve_dataset(self, value: str):
        if value != self.dataset.summary.id:
            raise ResourceNotFound(value)
        return self.dataset


def request(catalog: FakeCatalog, *, run_id: str = "run-001") -> TrainingJobRequest:
    return TrainingJobRequest(
        config_resource_id=catalog.config.summary.id,
        dataset_resource_id=catalog.dataset.summary.id,
        run_id=run_id,
    )


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
