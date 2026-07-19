from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.studio.helpers import write_run
from trade_rl.studio.catalog import StudioCatalog
from trade_rl.studio.comparison import compare_runs
from trade_rl.studio.settings import StudioSettings


def settings_for(root: Path) -> StudioSettings:
    return StudioSettings(
        project_root=root,
        dataset_roots=(root / "datasets",),
        run_roots=(root / "research",),
        config_roots=(root / "configs",),
        job_root=root / "jobs",
        serving_root=root / "serving",
        paper_snapshot_path=root / "paper-inference.json",
    )


def rewrite_run_payload(root: Path, *, total_return: float, sharpe: float, fee: float) -> None:
    walk_forward = json.loads((root / "walk-forward.json").read_text(encoding="utf-8"))
    walk_forward["selected_metrics"]["total_return"] = total_return
    walk_forward["selected_metrics"]["sharpe"] = sharpe
    walk_forward["selected_metrics"]["turnover_total"] = 2.5
    walk_forward["selected_metrics"]["total_cost"] = fee
    walk_forward["folds"] = [
        {
            "fold_index": 0,
            "selected_returns": [total_return / 2.0, total_return / 2.0],
            "baseline_returns": [0.01, 0.01],
        }
    ]
    (root / "walk-forward.json").write_text(json.dumps(walk_forward), encoding="utf-8")
    config = json.loads((root / "training-config.json").read_text(encoding="utf-8"))
    config["execution"] = {"fee_rate": fee}
    (root / "training-config.json").write_text(json.dumps(config), encoding="utf-8")

    # Refresh the manifest because canonical artifact digests changed.
    from datetime import UTC, datetime

    from trade_rl.artifacts.run_manifest import TrainingRunManifest, write_training_run_manifest

    old = json.loads((root / "run.json").read_text(encoding="utf-8"))
    manifest = TrainingRunManifest.build(
        root=root,
        run_id=old["run_id"],
        dataset_id=old["dataset_id"],
        environment_digest=old["environment_digest"],
        ensemble_digest=old["ensemble_digest"],
        training_config_digest=old["training_config_digest"],
        provenance_digest=old["provenance_digest"],
        artifact_paths=tuple(item["path"] for item in old["files"]),
        created_at=datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 7, 19, 12, 5, tzinfo=UTC),
    )
    write_training_run_manifest(root, manifest)


def test_catalog_resolves_only_exact_valid_run_ids(tmp_path: Path) -> None:
    run = write_run(tmp_path / "research", run_id="run-left")
    catalog = StudioCatalog(settings_for(tmp_path))

    assert catalog.resolve_run("run-left") == run
    with pytest.raises(KeyError, match="unknown Studio run"):
        catalog.resolve_run("../run-left")
    with pytest.raises(KeyError, match="unknown Studio run"):
        catalog.resolve_run("missing")


def test_compare_runs_returns_metric_deltas_config_diffs_and_fold_wealth(tmp_path: Path) -> None:
    left = write_run(tmp_path / "research", run_id="run-left", algorithm="ppo")
    right = write_run(tmp_path / "research", run_id="run-right", algorithm="sac")
    rewrite_run_payload(left, total_return=0.20, sharpe=1.0, fee=0.001)
    rewrite_run_payload(right, total_return=0.35, sharpe=1.4, fee=0.002)

    result = compare_runs(left, right)

    metrics = {item.key: item for item in result.metrics}
    assert metrics["total_return"].left_value == pytest.approx(0.20)
    assert metrics["total_return"].right_value == pytest.approx(0.35)
    assert metrics["total_return"].delta == pytest.approx(0.15)
    assert metrics["max_drawdown"].preference == "lower"
    differences = {item.path: item for item in result.config_differences}
    assert differences["training.algorithm"].left == "ppo"
    assert differences["training.algorithm"].right == "sac"
    assert differences["execution.fee_rate"].left == "0.001"
    assert result.folds[0].left_selected_return == pytest.approx(0.21)
    assert result.folds[0].right_selected_return == pytest.approx(0.380625)
    assert result.wealth[0].left == 1.0
    assert result.wealth[-1].right > result.wealth[-1].left
    assert result.production_status == "NO-GO"


def test_compare_runs_preserves_missing_metrics_as_none(tmp_path: Path) -> None:
    left = write_run(tmp_path / "research", run_id="run-left", with_walk_forward=False)
    right = write_run(tmp_path / "research", run_id="run-right")

    result = compare_runs(left, right)

    total_return = next(item for item in result.metrics if item.key == "total_return")
    assert total_return.left_value is None
    assert total_return.delta is None
