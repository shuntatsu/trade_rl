from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mars_lite.eval import residual_walk_forward
from mars_lite.eval.residual_walk_forward import run_residual_walk_forward


class _FeatureSet:
    n_bars = 1_000


def test_runner_writes_only_authoritative_residual_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        residual_walk_forward,
        "build_feature_set",
        lambda args, output_dir=None: _FeatureSet(),
    )
    monkeypatch.setattr(
        residual_walk_forward,
        "run_residual_fold",
        lambda **kwargs: {
            "selected_configuration": "A",
            "alpha_enabled": False,
            "selected_seed_fallbacks": [],
            "outer_oos": {
                "relative_1x": {
                    "hybrid": {"total_return": 0.1, "n_trades": 3},
                    "shadow": {"total_return": 0.1, "n_trades": 3},
                    "paired": {"excess_log_return": 0.0},
                },
                "relative_2x": {
                    "hybrid": {"total_return": 0.08, "n_trades": 3},
                    "shadow": {"total_return": 0.08, "n_trades": 3},
                    "paired": {"excess_log_return": 0.0},
                },
            },
            "split": {"outer_test_scored_bars": 100},
        },
    )

    args = SimpleNamespace(
        folds=3,
        horizon=12,
        purge_bars=24,
        decision_every=4,
        ensemble=3,
        n_seeds=3,
        run_tier="research",
        min_trade_delta=0.04,
        lambda_turnover=0.04,
    )
    report = run_residual_walk_forward(args, tmp_path)

    assert report["mode"] == "baseline_residual_walk_forward_v1"
    assert report["action_schema"] == "baseline_residual_v1"
    assert report["release_eligible"] is False
    assert report["summary"]["completed_folds"] == 3
    assert args.action_mode == "baseline-residual"
    assert args.min_trade_delta == 0.0
    assert args.lambda_turnover == 0.0
    assert (tmp_path / "residual_walk_forward.json").is_file()
    assert not (tmp_path / "walk_forward_cost1x.json").exists()
    assert not (tmp_path / "walk_forward_cost2x.json").exists()


def test_runner_does_not_write_success_report_when_fold_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        residual_walk_forward,
        "build_feature_set",
        lambda args, output_dir=None: _FeatureSet(),
    )

    def explode(**kwargs):
        raise RuntimeError("fold failed")

    monkeypatch.setattr(residual_walk_forward, "run_residual_fold", explode)
    args = SimpleNamespace(
        folds=3,
        horizon=12,
        purge_bars=24,
        decision_every=4,
        ensemble=3,
        n_seeds=3,
        run_tier="research",
        min_trade_delta=0.04,
        lambda_turnover=0.04,
    )

    try:
        run_residual_walk_forward(args, tmp_path)
    except RuntimeError as exc:
        assert str(exc) == "fold failed"
    else:
        raise AssertionError("expected fold failure")

    assert not (tmp_path / "residual_walk_forward.json").exists()
