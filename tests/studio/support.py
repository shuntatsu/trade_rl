from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from trade_rl.studio.api import create_app
from trade_rl.studio.catalog import StudioCatalog
from trade_rl.studio.contracts import ConfigSummary, DatasetSummary, TrainingJobRequest
from trade_rl.studio.errors import ResourceNotFound
from trade_rl.studio.jobs import JobSupervisor
from trade_rl.studio.resource_ids import resource_id
from trade_rl.studio.settings import StudioSettings

from .helpers import write_dataset, write_run


def settings(
    tmp_path: Path,
    *,
    dataset_roots: tuple[Path, ...] | None = None,
    run_roots: tuple[Path, ...] | None = None,
) -> StudioSettings:
    return StudioSettings(
        project_root=tmp_path,
        dataset_roots=dataset_roots or (tmp_path / "datasets",),
        run_roots=run_roots or (tmp_path / "research",),
        config_roots=(tmp_path / "configs",),
        job_root=tmp_path / "jobs",
    )


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

    def resolve_config(self, value: str) -> SimpleNamespace:
        if value != self.config.summary.id:
            raise ResourceNotFound(value)
        return self.config

    def resolve_dataset(self, value: str) -> SimpleNamespace:
        if value != self.dataset.summary.id:
            raise ResourceNotFound(value)
        return self.dataset


def request(catalog: FakeCatalog, *, run_id: str = "run-001") -> TrainingJobRequest:
    return TrainingJobRequest(
        config_resource_id=catalog.config.summary.id,
        dataset_resource_id=catalog.dataset.summary.id,
        run_id=run_id,
    )


def studio_client(
    tmp_path: Path,
) -> tuple[TestClient, FakeFactory, FakeCatalog, StudioCatalog]:
    write_dataset(tmp_path / "datasets" / "btc")
    write_run(tmp_path / "research")
    real_catalog = StudioCatalog(settings(tmp_path))
    job_catalog = FakeCatalog(tmp_path)
    factory = FakeFactory()
    supervisor = JobSupervisor(
        settings(tmp_path),
        catalog=job_catalog,
        process_factory=factory,
    )
    return (
        TestClient(
            create_app(
                settings(tmp_path),
                catalog=real_catalog,
                supervisor=supervisor,
            )
        ),
        factory,
        job_catalog,
        real_catalog,
    )
