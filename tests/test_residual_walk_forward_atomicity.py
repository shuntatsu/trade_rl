from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_lite.eval.residual_walk_forward import ResidualFoldSpec
from mars_lite.pipeline import residual_walk_forward
from mars_lite.pipeline.residual_walk_forward import run_residual_walk_forward


class _FeatureSet:
    n_bars = 2_000


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        folds=3,
        horizon=12,
        purge_bars=24,
        decision_every=4,
        ensemble=3,
        n_seeds=3,
        run_tier="research",
        min_trade_delta=0.04,
        lambda_turnover=0.04,
        seed=0,
        base_timeframe="1h",
        signal_model="gbm",
        fee_profile="taker",
        scan_horizons=False,
    )


def _spec(fold: int) -> ResidualFoldSpec:
    offset = fold * 100
    return ResidualFoldSpec(
        fold=fold,
        policy_train_start=0,
        policy_train_end=700 + offset,
        checkpoint_validation_start=724 + offset,
        checkpoint_validation_end=824 + offset,
        configuration_selection_start=848 + offset,
        configuration_selection_end=948 + offset,
        outer_test_start=972 + offset,
        outer_test_end=1_072 + offset,
        purge_bars=24,
    )


def _book() -> dict[str, float | int]:
    return {
        "total_return": 0.01,
        "n_trades": 2,
        "turnover_total": 0.2,
        "total_cost": 0.001,
    }


def _fold_report(fold: int, *, include_series: bool = False) -> dict[str, object]:
    relative = {
        "hybrid": _book(),
        "shadow": _book(),
        "paired": {"excess_log_return": 0.0},
    }
    report: dict[str, object] = {
        "fold": fold,
        "selected_configuration": "A",
        "alpha_enabled": False,
        "selected_seed_fallbacks": [],
        "outer_oos": {"relative_1x": relative, "relative_2x": relative},
        "split": {"outer_test_scored_bars": 2},
    }
    if include_series:
        report["_return_series_1x"] = {
            "hybrid": [0.001, -0.0005],
            "shadow": [0.001, -0.0005],
        }
        report["_return_series_2x"] = {
            "hybrid": [0.0008, -0.0006],
            "shadow": [0.0008, -0.0006],
        }
    return report


def _stub_dataset(monkeypatch) -> None:
    monkeypatch.setattr(
        residual_walk_forward,
        "build_feature_set",
        lambda args, output_dir=None: _FeatureSet(),
    )
    monkeypatch.setattr(
        residual_walk_forward,
        "feature_set_identity",
        lambda fs: "dataset-identity",
    )
    monkeypatch.setattr(residual_walk_forward, "_new_run_id", lambda: "run-test")


def test_failed_rerun_preserves_prior_success_and_isolates_partial_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _stub_dataset(monkeypatch)
    authoritative = tmp_path / "residual_walk_forward.json"
    previous = b'{"run_id":"prior-success","status":"completed"}\n'
    authoritative.write_bytes(previous)
    monkeypatch.setattr(
        residual_walk_forward,
        "build_residual_fold_specs",
        lambda **kwargs: ([_spec(0), _spec(1)], []),
    )
    calls = 0

    def run_fold(*, output_dir: Path, spec: ResidualFoldSpec, **kwargs):
        nonlocal calls
        calls += 1
        partial = Path(output_dir) / "residual_wf" / f"fold_{spec.fold}"
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "partial.txt").write_text("partial", encoding="utf-8")
        if calls == 2:
            raise RuntimeError("fold failed")
        return _fold_report(spec.fold)

    monkeypatch.setattr(residual_walk_forward, "run_residual_fold", run_fold)

    with pytest.raises(RuntimeError, match="fold failed"):
        run_residual_walk_forward(_args(), tmp_path)

    assert authoritative.read_bytes() == previous
    assert not (tmp_path / "residual_wf").exists()
    assert (tmp_path / "failed" / "run-test" / "residual_wf" / "fold_0").is_dir()
    assert (tmp_path / "failed" / "run-test" / "residual_wf" / "fold_1").is_dir()


def test_run_fails_closed_when_fewer_than_two_folds_complete(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _stub_dataset(monkeypatch)
    monkeypatch.setattr(
        residual_walk_forward,
        "build_residual_fold_specs",
        lambda **kwargs: ([_spec(0)], [{"fold": 1, "reason": "too_short"}]),
    )
    monkeypatch.setattr(
        residual_walk_forward,
        "run_residual_fold",
        lambda **kwargs: _fold_report(0),
    )

    with pytest.raises(RuntimeError, match="at least two completed folds"):
        run_residual_walk_forward(_args(), tmp_path)

    assert not (tmp_path / "residual_walk_forward.json").exists()
    assert (tmp_path / "failed" / "run-test").is_dir()


def test_pointer_publication_failure_moves_completed_run_to_failed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _stub_dataset(monkeypatch)
    authoritative = tmp_path / "residual_walk_forward.json"
    previous = b'{"run_id":"prior-success","status":"completed"}\n'
    authoritative.write_bytes(previous)
    monkeypatch.setattr(
        residual_walk_forward,
        "build_residual_fold_specs",
        lambda **kwargs: ([_spec(0), _spec(1)], []),
    )

    def run_fold(*, output_dir: Path, spec: ResidualFoldSpec, **kwargs):
        payload = _fold_report(spec.fold, include_series=True)
        fold_dir = Path(output_dir) / "residual_wf" / f"fold_{spec.fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        public = {
            key: value for key, value in payload.items() if not key.startswith("_")
        }
        (fold_dir / "fold_report.json").write_text(
            json.dumps(public),
            encoding="utf-8",
        )
        return payload

    monkeypatch.setattr(residual_walk_forward, "run_residual_fold", run_fold)
    monkeypatch.setattr(
        residual_walk_forward,
        "_publish_authoritative_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pointer failed")),
    )

    with pytest.raises(RuntimeError, match="pointer failed"):
        run_residual_walk_forward(_args(), tmp_path)

    assert authoritative.read_bytes() == previous
    assert not (tmp_path / "residual_wf_runs" / "run-test").exists()
    assert (tmp_path / "failed" / "run-test" / "residual_wf").is_dir()
