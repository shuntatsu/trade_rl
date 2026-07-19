from __future__ import annotations

from pathlib import Path

import pytest

from tests.studio.test_catalog import settings
from trade_rl.studio.contracts import TrainingJobRequest
from trade_rl.studio.jobs import JobSupervisor


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
    def __init__(self) -> None:
        self.process = FakeProcess()
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


def prepare_inputs(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "training.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"training":{"algorithm":"ppo"}}', encoding="utf-8")
    dataset = tmp_path / "datasets" / "btc"
    dataset.mkdir(parents=True, exist_ok=True)


def request() -> TrainingJobRequest:
    return TrainingJobRequest(
        config_path="configs/training.json",
        dataset_path="datasets/btc",
        artifact_root="research",
        run_id="run-001",
    )


def test_submit_training_persists_fixed_command_and_reconciles_success(
    tmp_path: Path,
) -> None:
    prepare_inputs(tmp_path)
    factory = FakeFactory()
    supervisor = JobSupervisor(settings(tmp_path), process_factory=factory)

    job = supervisor.submit_training(request())

    assert job.status == "running"
    assert job.pid == 4242
    command = factory.commands[0]
    assert command[-10:] == (
        "train",
        "run",
        "--config",
        str((tmp_path / "configs" / "training.json").resolve()),
        "--dataset",
        str((tmp_path / "datasets" / "btc").resolve()),
        "--output",
        str((tmp_path / "research").resolve()),
        "--run-id",
        "run-001",
    )
    assert (tmp_path / "jobs" / f"{job.id}.json").is_file()

    (tmp_path / "research" / "runs" / "run-001").mkdir(parents=True)
    factory.process.exit_code = 0
    finished = supervisor.get_job(job.id)

    assert finished.status == "succeeded"
    assert finished.exit_code == 0
    assert finished.completed_at is not None


def test_nonzero_worker_exit_is_failed_and_log_tail_is_bounded(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    factory = FakeFactory()
    supervisor = JobSupervisor(settings(tmp_path), process_factory=factory)
    job = supervisor.submit_training(request())
    factory.logs[0].write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    factory.process.exit_code = 2

    failed = supervisor.get_job(job.id)
    lines, truncated = supervisor.tail_log(job.id, limit=2)

    assert failed.status == "failed"
    assert failed.exit_code == 2
    assert lines == ("three", "four")
    assert truncated is True


def test_duplicate_run_is_rejected(tmp_path: Path) -> None:
    prepare_inputs(tmp_path)
    supervisor = JobSupervisor(settings(tmp_path), process_factory=FakeFactory())
    supervisor.submit_training(request())

    with pytest.raises(FileExistsError, match="run-001"):
        supervisor.submit_training(request())


def test_cancel_terminates_running_process_and_persists_cancelled_state(
    tmp_path: Path,
) -> None:
    prepare_inputs(tmp_path)
    factory = FakeFactory()
    supervisor = JobSupervisor(settings(tmp_path), process_factory=factory)
    job = supervisor.submit_training(request())

    cancelled = supervisor.cancel(job.id)

    assert factory.process.terminated is True
    assert cancelled.status == "cancelled"
    assert cancelled.completed_at is not None


def test_unknown_job_is_rejected(tmp_path: Path) -> None:
    supervisor = JobSupervisor(settings(tmp_path), process_factory=FakeFactory())

    with pytest.raises(KeyError, match="missing"):
        supervisor.get_job("missing")
