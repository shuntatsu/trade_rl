from __future__ import annotations

from pathlib import Path

import pytest

from tests.studio.helpers import write_dataset, write_run
from trade_rl.studio.catalog import StudioCatalog
from trade_rl.studio.settings import StudioSettings


def settings(tmp_path: Path) -> StudioSettings:
    return StudioSettings(
        project_root=tmp_path,
        dataset_roots=(tmp_path / "datasets",),
        run_roots=(tmp_path / "research",),
        config_roots=(tmp_path / "configs",),
        job_root=tmp_path / "jobs",
    )


def test_catalog_lists_validated_datasets_and_reports_invalid_artifacts(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "datasets" / "btc-hourly"
    dataset = write_dataset(valid)
    invalid = tmp_path / "datasets" / "broken"
    invalid.mkdir(parents=True)
    (invalid / "manifest.json").write_text("{}", encoding="utf-8")
    (invalid / "arrays.npz").write_bytes(b"broken")

    records = StudioCatalog(settings(tmp_path)).list_datasets()

    assert len(records) == 2
    valid_record = next(item for item in records if item.status == "VALID")
    assert valid_record.id == dataset.dataset_id
    assert valid_record.relative_path == "datasets/btc-hourly"
    assert valid_record.symbols == ("BTCUSDT",)
    assert valid_record.bar_count == 12
    assert valid_record.feature_count == 1
    assert valid_record.validation_error is None
    invalid_record = next(item for item in records if item.status == "INVALID")
    assert invalid_record.relative_path == "datasets/broken"
    assert invalid_record.validation_error


def test_catalog_lists_validated_runs_and_extracts_walk_forward_metrics(
    tmp_path: Path,
) -> None:
    write_run(tmp_path / "research")
    broken = tmp_path / "research" / "runs" / "broken"
    broken.mkdir(parents=True)
    (broken / "run.json").write_text("{}", encoding="utf-8")

    records = StudioCatalog(settings(tmp_path)).list_runs()

    assert len(records) == 2
    run = next(item for item in records if item.status == "VALID")
    assert run.id == "run-001"
    assert run.algorithm == "ppo"
    assert run.sharpe == pytest.approx(1.25)
    assert run.max_drawdown == pytest.approx(0.12)
    assert run.total_return == pytest.approx(0.34)
    assert run.production_status == "NO-GO"
    assert run.relative_path == "research/runs/run-001"
    invalid = next(item for item in records if item.status == "INVALID")
    assert invalid.validation_error


def test_settings_rejects_paths_that_escape_configured_roots(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    (tmp_path / "datasets").mkdir()

    with pytest.raises(ValueError, match="configured roots"):
        configured.resolve_dataset_path("../outside")
    with pytest.raises(ValueError, match="relative"):
        configured.resolve_dataset_path(str((tmp_path / "datasets").resolve()))


def test_overview_uses_real_catalog_and_remains_no_go(tmp_path: Path) -> None:
    write_dataset(tmp_path / "datasets" / "btc-hourly")
    write_run(tmp_path / "research")

    overview = StudioCatalog(settings(tmp_path)).overview(jobs=())

    assert overview.latest_dataset is not None
    assert overview.latest_dataset.name == "btc-hourly"
    assert overview.runs[0].id == "run-001"
    assert overview.equity[-1].rl > overview.equity[-1].baseline
    assert overview.assessment.status == "NO-GO"
    assert overview.assessment.reasons
