from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def _book(total_return: float) -> dict[str, float | int]:
    return {
        "total_return": total_return,
        "n_trades": 3,
        "turnover_total": 0.5,
        "total_cost": 0.001,
    }


def _fold_payload(fold: int) -> dict[str, object]:
    return {
        "fold": fold,
        "selected_configuration": "A",
        "alpha_enabled": False,
        "selected_seed_fallbacks": [],
        "outer_oos": {
            "relative_1x": {
                "hybrid": _book(0.1),
                "shadow": _book(0.1),
                "paired": {"excess_log_return": 0.0},
            },
            "relative_2x": {
                "hybrid": _book(0.08),
                "shadow": _book(0.08),
                "paired": {"excess_log_return": 0.0},
            },
        },
        "split": {"outer_test_scored_bars": 2},
        "_return_series_1x": {
            "hybrid": [0.01, -0.005],
            "shadow": [0.01, -0.005],
        },
        "_return_series_2x": {
            "hybrid": [0.008, -0.006],
            "shadow": [0.008, -0.006],
        },
    }


def test_runner_writes_only_authoritative_residual_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _stub_dataset(monkeypatch)

    def fake_fold(*, spec, output_dir, **kwargs):
        payload = _fold_payload(spec.fold)
        destination = Path(output_dir) / "residual_wf" / f"fold_{spec.fold}"
        destination.mkdir(parents=True, exist_ok=True)
        public = {
            key: value for key, value in payload.items() if not key.startswith("_")
        }
        (destination / "fold_report.json").write_text(
            json.dumps(public),
            encoding="utf-8",
        )
        return payload

    monkeypatch.setattr(residual_walk_forward, "run_residual_fold", fake_fold)

    args = _args()
    original = vars(args).copy()
    report = run_residual_walk_forward(args, tmp_path)

    assert vars(args) == original
    assert report["run_id"] == "run-test"
    assert report["status"] == "completed"
    assert report["mode"] == "baseline_residual_walk_forward_v1"
    assert report["action_schema"] == "baseline_residual_v1"
    assert report["release_eligible"] is False
    assert report["summary"]["completed_folds"] == 3
    assert report["summary"]["stitched_oos"]["cost1x"]["n_base_bars"] == 6
    assert report["config"]["effective_decision_every"] == 4
    assert report["config"]["effective_ensemble_size"] == 3
    assert report["config"]["dataset_identity"] == "dataset-identity"
    assert all("_return_series_1x" not in fold for fold in report["folds"])
    assert (tmp_path / "residual_walk_forward.json").is_file()
    assert (tmp_path / "residual_wf_runs" / "run-test").is_dir()
    assert not (tmp_path / "residual_wf").exists()
    assert not (tmp_path / "walk_forward_cost1x.json").exists()
    assert not (tmp_path / "walk_forward_cost2x.json").exists()


def test_runner_does_not_write_success_report_when_fold_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _stub_dataset(monkeypatch)

    def explode(**kwargs):
        raise RuntimeError("fold failed")

    monkeypatch.setattr(residual_walk_forward, "run_residual_fold", explode)

    try:
        run_residual_walk_forward(_args(), tmp_path)
    except RuntimeError as exc:
        assert str(exc) == "fold failed"
    else:
        raise AssertionError("expected fold failure")

    assert not (tmp_path / "residual_walk_forward.json").exists()
    assert (tmp_path / "failed" / "run-test").is_dir()
