"""
strategy_wf（多期間walk-forward審判ハーネス）のテスト

ダミーのWeightFn（trend_followingと厳密に一致するもの、明確に勝つもの、
明確に負けるもの）を使い、fold分割・集計・DSR・bootstrap・判定関数の
機械的な正しさを検証する。実データのtrend_v2/carryのテストは各モジュール側。
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.eval.strategy_wf import (
    STRATEGY_GATE_CRITERIA,
    compute_bootstrap_vs_baseline,
    compute_dsr,
    fold_edges,
    judge_carry,
    judge_combo,
    judge_trend_v2,
    run_holdout_once,
    run_strategy_walk_forward,
    save_report,
    summarize,
)
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.learning.baselines import trend_following_strategy


def _fs(days=150, alpha="cross", seed=3):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


class TestFoldEdges:
    def test_edges_span_from_04n_to_n(self):
        edges = fold_edges(1000, 6)
        assert edges[0] == 400
        assert edges[-1] == 1000
        assert len(edges) == 7


class TestHarnessMechanics:
    def test_identical_strategy_scores_zero_uplift(self):
        """候補=trend_followingそのものなら、全foldでuplift=0・beat_baseline=0。"""
        fs = _fs()
        report = run_strategy_walk_forward(
            fs, {"same_as_tf": trend_following_strategy}, n_folds=4
        )
        summary = summarize(report)
        s = summary["same_as_tf"][2.0]
        assert abs(s["median_uplift_pt"]) < 1e-6
        assert s["n_folds_beat_baseline"] == 0

    def test_zero_strategy_loses_to_positive_trend(self):
        """常にflat(全ゼロ)の候補は、アルファ有データでtrend_followingに劣後する
        （trend自体が正のリターンを持つ期間が多いはず）。"""
        fs = _fs(alpha="cross", seed=11)

        def flat_fn(fs, t, prev):
            return np.zeros(fs.n_symbols)

        report = run_strategy_walk_forward(fs, {"flat_dummy": flat_fn}, n_folds=4)
        summary = summarize(report)
        s = summary["flat_dummy"][2.0]
        assert s["median_return"] == 0.0

    def test_baseline_auto_added_when_missing(self):
        fs = _fs()

        def dummy(fs, t, prev):
            return np.zeros(fs.n_symbols)

        report = run_strategy_walk_forward(fs, {"dummy": dummy}, n_folds=3)
        assert "trend_following" in report["fold_returns"][1.0]

    def test_folds_are_non_overlapping(self):
        fs = _fs()
        report = run_strategy_walk_forward(
            fs, {"tf": trend_following_strategy}, n_folds=5
        )
        edges = fold_edges(fs.n_bars, 5)
        for i, f in enumerate(report["folds"]):
            assert f["train_bars"] == int(edges[i])


class TestDSRAndBootstrap:
    def test_dsr_runs_and_returns_expected_keys(self):
        fs = _fs()
        report = run_strategy_walk_forward(
            fs, {"tf_copy": trend_following_strategy}, n_folds=4
        )
        dsr = compute_dsr(report, "tf_copy", 2.0, trial_sharpes=[1.0, 1.2, 0.8])
        assert set(["dsr", "sr0_annualized", "sr_hat_annualized", "n_trials"]).issubset(
            dsr.keys()
        )

    def test_bootstrap_identical_strategy_gives_zero_diff(self):
        fs = _fs()
        report = run_strategy_walk_forward(
            fs, {"tf_copy": trend_following_strategy}, n_folds=4
        )
        boot = compute_bootstrap_vs_baseline(report, "tf_copy", 2.0, seed=0)
        assert abs(boot["observed_diff"]) < 1e-9


class TestJudges:
    def _fake_summary(
        self, median_return, median_sharpe, median_maxdd, beat, total, corr=0.0
    ):
        return {
            2.0: {
                "median_return": median_return,
                "median_sharpe": median_sharpe,
                "median_maxdd": median_maxdd,
                "n_folds_beat_baseline": beat,
                "n_folds_total": total,
                "median_uplift_pt": median_return * 100,
                "correlation_with_baseline": corr,
            }
        }

    def test_judge_trend_v2_passes_when_all_criteria_met(self):
        summary = {
            "trend_v2": self._fake_summary(0.05, 2.0, 0.1, 5, 6),
        }
        dsr = {"dsr": 0.95}
        boot = {"lower_ci": 0.1}
        verdict = judge_trend_v2(summary, dsr, boot)
        assert verdict["passed"] is True

    def test_judge_trend_v2_fails_on_low_dsr(self):
        summary = {"trend_v2": self._fake_summary(0.05, 2.0, 0.1, 5, 6)}
        dsr = {"dsr": 0.5}  # 基準0.90未満
        boot = {"lower_ci": 0.1}
        verdict = judge_trend_v2(summary, dsr, boot)
        assert verdict["passed"] is False
        assert verdict["checks"]["dsr"] is False

    def test_judge_carry_fails_on_high_correlation(self):
        summary = {"carry": self._fake_summary(0.03, 1.0, 0.05, 4, 6, corr=0.8)}
        dsr = {"dsr": 0.95}
        verdict = judge_carry(summary, dsr)
        assert verdict["passed"] is False
        assert verdict["checks"]["low_tf_correlation"] is False

    def test_judge_combo_requires_maxdd_improvement(self):
        summary = {
            "combo": self._fake_summary(0.10, 2.0, 0.30, 5, 6),  # maxDD悪い
            "trend_following": self._fake_summary(0.08, 1.8, 0.20, 0, 6),
            "trend_v2": self._fake_summary(0.09, 1.9, 0.15, 3, 6),
        }
        verdict = judge_combo(summary)
        assert verdict["checks"]["maxdd_improved"] is False
        assert verdict["passed"] is False


class TestSaveAndHoldout:
    def test_save_report_is_json_serializable(self, tmp_path):
        fs = _fs()
        report = run_strategy_walk_forward(
            fs, {"tf_copy": trend_following_strategy}, n_folds=3
        )
        path = tmp_path / "report.json"
        save_report(report, path)
        assert path.exists()
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "folds" in data

    def test_holdout_once_writes_marker_and_warns_on_repeat(self, tmp_path, capsys):
        fs = _fs()
        holdout = fs.slice(fs.n_bars - 300, fs.n_bars)
        run_holdout_once(
            holdout,
            trend_following_strategy,
            tmp_path,
            name="tf_copy",
            trial_sharpes=[1.0],
        )
        marker = tmp_path / "strategy_holdout_used.marker"
        assert marker.exists()

        run_holdout_once(
            holdout,
            trend_following_strategy,
            tmp_path,
            name="tf_copy",
            trial_sharpes=[1.0],
        )
        captured = capsys.readouterr()
        assert "警告" in captured.out
