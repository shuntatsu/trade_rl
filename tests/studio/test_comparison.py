from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    write_training_run_manifest,
)
from trade_rl.studio.catalog import StudioCatalog
from trade_rl.studio.comparison import compare_runs
from trade_rl.studio.settings import StudioSettings

from .helpers import write_run


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


def rewrite_run_payload(
    root: Path,
    *,
    total_return: float,
    sharpe: float,
    fee: float,
    test_range: list[int] | None = None,
) -> None:
    walk_forward = json.loads((root / "walk-forward.json").read_text(encoding="utf-8"))
    walk_forward["dataset_id"] = json.loads((root / "run.json").read_text())[
        "dataset_id"
    ]
    walk_forward["stitch_mode"] = "independent"
    walk_forward["selected_metrics"]["total_return"] = total_return
    walk_forward["selected_metrics"]["sharpe"] = sharpe
    walk_forward["selected_metrics"]["turnover_total"] = 2.5
    walk_forward["selected_metrics"]["total_cost"] = fee
    fold = {
        "fold_index": 0,
        "selected_returns": [total_return / 2.0, total_return / 2.0],
        "baseline_returns": [0.01, 0.01],
    }
    if test_range is not None:
        fold["test_range"] = test_range
    walk_forward["folds"] = [fold]
    (root / "walk-forward.json").write_text(json.dumps(walk_forward), encoding="utf-8")
    config = json.loads((root / "training-config.json").read_text(encoding="utf-8"))
    config["execution"] = {"fee_rate": fee}
    (root / "training-config.json").write_text(json.dumps(config), encoding="utf-8")

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


def resolved_pair(tmp_path: Path):
    catalog = StudioCatalog(settings_for(tmp_path))
    records = {
        item.run_id: item for item in catalog.list_runs() if item.status == "VALID"
    }
    return catalog.resolve_run(records["run-left"].id), catalog.resolve_run(
        records["run-right"].id
    )


def test_compare_runs_returns_comparable_aligned_metrics(tmp_path: Path) -> None:
    left = write_run(tmp_path / "research", run_id="run-left", algorithm="ppo")
    right = write_run(tmp_path / "research", run_id="run-right", algorithm="sac")
    rewrite_run_payload(
        left, total_return=0.20, sharpe=1.0, fee=0.001, test_range=[10, 12]
    )
    rewrite_run_payload(
        right, total_return=0.35, sharpe=1.4, fee=0.002, test_range=[10, 12]
    )
    left_resolved, right_resolved = resolved_pair(tmp_path)

    result = compare_runs(left_resolved, right_resolved)

    assert result.eligibility.status == "COMPARABLE"
    metrics = {item.key: item for item in result.metrics}
    assert metrics["total_return"].delta == pytest.approx(0.15)
    assert result.wealth[1].label == "10"
    assert result.wealth[-1].right > result.wealth[-1].left
    assert result.left_resource_id == left_resolved.summary.id


def test_compare_runs_marks_legacy_missing_ranges_partial(tmp_path: Path) -> None:
    left = write_run(tmp_path / "research", run_id="run-left")
    right = write_run(tmp_path / "research", run_id="run-right")
    rewrite_run_payload(left, total_return=0.2, sharpe=1.0, fee=0.001)
    rewrite_run_payload(right, total_return=0.3, sharpe=1.1, fee=0.001)

    result = compare_runs(*resolved_pair(tmp_path))

    assert result.eligibility.status == "PARTIALLY_COMPARABLE"
    assert any("test ranges" in reason for reason in result.eligibility.reasons)
    assert result.metrics


def test_compare_runs_fails_closed_for_different_datasets(tmp_path: Path) -> None:
    left = write_run(tmp_path / "research", run_id="run-left", dataset_id="a" * 64)
    right = write_run(tmp_path / "research", run_id="run-right", dataset_id="f" * 64)
    rewrite_run_payload(
        left, total_return=0.2, sharpe=1.0, fee=0.001, test_range=[10, 12]
    )
    rewrite_run_payload(
        right, total_return=0.3, sharpe=1.1, fee=0.001, test_range=[10, 12]
    )

    result = compare_runs(*resolved_pair(tmp_path))

    assert result.eligibility.status == "NOT_COMPARABLE"
    assert result.metrics == ()
    assert result.folds == ()
    assert result.wealth == ()
