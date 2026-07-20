from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from trade_rl.studio.catalog import StudioCatalog
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


def write_config(root: Path, name: str = "training.json") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[2] / "examples/quickstart/training.json"
    target = root / name
    shutil.copyfile(source, target)
    return target


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
    assert valid_record.id.startswith("dataset-")
    assert valid_record.dataset_id == dataset.dataset_id
    assert valid_record.relative_path == "datasets/btc-hourly"
    assert valid_record.symbols == ("BTCUSDT",)
    assert valid_record.bar_count == 12
    invalid_record = next(item for item in records if item.status == "INVALID")
    assert invalid_record.validation_error


def test_catalog_validates_complete_training_configs(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    write_config(tmp_path / "configs")
    (tmp_path / "configs" / "broken.json").write_text(
        '{"training":{"algorithm":"ppo"}}', encoding="utf-8"
    )

    catalog = StudioCatalog(settings(tmp_path))
    records = catalog.list_configs()

    valid = next(item for item in records if item.status == "VALID")
    invalid = next(item for item in records if item.status == "INVALID")
    assert valid.id.startswith("config-")
    assert valid.config_digest is not None
    assert catalog.resolve_config(valid.id).config.training.algorithm == "ppo"
    assert invalid.validation_error


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
    assert run.id.startswith("run-")
    assert run.run_id == "run-001"
    assert run.algorithm == "ppo"
    assert run.sharpe == pytest.approx(1.25)
    assert run.relative_path == "research/runs/run-001"
    invalid = next(item for item in records if item.status == "INVALID")
    assert invalid.validation_error


def test_duplicate_human_run_ids_resolve_by_unique_resource_id(tmp_path: Path) -> None:
    first_root = tmp_path / "research-a"
    second_root = tmp_path / "research-b"
    first = write_run(first_root, run_id="same-run")
    second = write_run(second_root, run_id="same-run")
    catalog = StudioCatalog(settings(tmp_path, run_roots=(first_root, second_root)))

    records = [item for item in catalog.list_runs() if item.status == "VALID"]

    assert len(records) == 2
    assert records[0].run_id == records[1].run_id == "same-run"
    assert records[0].id != records[1].id
    assert {catalog.resolve_run(item.id).path for item in records} == {first, second}


def test_dataset_validation_cache_reuses_and_invalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "datasets" / "btc-hourly"
    write_dataset(root)
    import trade_rl.studio.dataset_catalog as module

    original = module.load_market_dataset_artifact
    calls = 0

    def counted(path: Path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(module, "load_market_dataset_artifact", counted)
    catalog = StudioCatalog(settings(tmp_path))

    assert catalog.list_datasets()[0].status == "VALID"
    assert catalog.list_datasets()[0].status == "VALID"
    assert calls == 1

    (root / "arrays.npz").write_bytes(b"tampered")
    assert catalog.list_datasets()[0].status == "INVALID"
    assert calls == 2


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
    assert overview.runs[0].run_id == "run-001"
    assert overview.equity[-1].rl > overview.equity[-1].baseline
    assert overview.assessment.status == "NO-GO"
