from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

from trade_rl.studio.contracts import (
    DatasetSummary,
    OverviewEvidenceSummary,
    RunSummary,
    SystemSummary,
)
from trade_rl.studio.overview import OverviewService


class Catalog:
    def __init__(self, items: tuple[Any, ...], root: Path) -> None:
        self.items = items
        self.root = root
        self.settings = SimpleNamespace(run_roots=())

    def list(self) -> tuple[Any, ...]:
        return self.items

    def resolve(self, value: str) -> SimpleNamespace:
        item = next(item for item in self.items if item.id == value)
        return SimpleNamespace(path=self.root / item.id, summary=item)

    def resolve_for_evidence(self, value: str) -> Path:
        return self.root / value


class System:
    def snapshot(self) -> SystemSummary:
        return SystemSummary(
            gpu_name="GPU",
            cuda_ready=True,
            python_version="3.12",
            metrics=(),
        )


def dataset(
    status: Literal["VALID", "INVALID"],
    *,
    updated: str = "2026-08-06T00:00:00+00:00",
) -> DatasetSummary:
    return DatasetSummary(
        id=f"dataset-{status.lower()}",
        dataset_id="a" * 64,
        name=status.lower(),
        relative_path=f"datasets/{status.lower()}",
        market="spot",
        symbols=("BTCUSDT",),
        timeframes=("1h",),
        range="2026",
        status=status,
        feature_count=1,
        bar_count=2,
        symbol_count=1,
        updated=updated,
        validation_error="broken" if status == "INVALID" else None,
    )


def run(status: Literal["VALID", "INVALID"]) -> RunSummary:
    return RunSummary(
        id=f"run-{status.lower()}",
        run_id=f"run-{status.lower()}",
        relative_path=f"research/{status.lower()}",
        run_kind="research_exploratory",
        algorithm="ppo",
        dataset_id="a" * 64,
        period="2026",
        created_at="2026-08-06T00:00:00+00:00",
        completed_at="2026-08-06T01:00:00+00:00",
        file_count=1,
        status=status,
        validation_error="tampered" if status == "INVALID" else None,
    )


def test_overview_exposes_latest_invalid_resources_and_stable_alerts(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    invalid_dataset = dataset("INVALID")
    valid_dataset = dataset("VALID")
    invalid_run = run("INVALID")
    valid_run = run("VALID")
    monkeypatch.setattr(
        "trade_rl.studio.overview.summarize_overview_evidence",
        lambda *_args, **_kwargs: OverviewEvidenceSummary(
            run_resource_id=invalid_run.id,
            status="INVALID",
            required_count=1,
            verified_count=0,
            blocker_count=1,
        ),
    )
    monkeypatch.setattr("trade_rl.studio.overview.read_json", lambda _path: None)
    service = OverviewService(
        cast(Any, Catalog((invalid_dataset, valid_dataset), tmp_path)),
        cast(Any, Catalog((invalid_run, valid_run), tmp_path)),
        cast(Any, System()),
    )

    first = service.build(())
    second = service.build(())

    assert first.latest_dataset == invalid_dataset
    assert first.runs[0] == invalid_run
    assert first.evidence.status == "INVALID"
    assert [item.id for item in first.alerts] == [item.id for item in second.alerts]
    assert any(
        item.id == f"dataset:{invalid_dataset.id}:invalid" for item in first.alerts
    )
    run_alert = next(
        item
        for item in first.alerts
        if item.id == f"run:{invalid_run.id}:invalid"
    )
    assert run_alert.occurred_at == invalid_run.completed_at


def test_overview_reports_unavailable_evidence_without_runs(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "trade_rl.studio.overview.summarize_overview_evidence",
        lambda *_args, **_kwargs: OverviewEvidenceSummary(
            run_resource_id=None,
            status="UNAVAILABLE",
            required_count=0,
            verified_count=0,
            blocker_count=0,
        ),
    )
    service = OverviewService(
        cast(Any, Catalog((), tmp_path)),
        cast(Any, Catalog((), tmp_path)),
        cast(Any, System()),
    )
    result = service.build(())

    assert result.latest_dataset is None
    assert result.runs == ()
    assert result.evidence.status == "UNAVAILABLE"
    assert {item.id for item in result.alerts} == {
        "dataset:no-valid",
        "run:no-valid",
    }
