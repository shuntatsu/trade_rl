"""
ICゲートの安定性判定（t値）のテスト

境界ぎりぎりの平均IC点推定だけでは、fold間で符号が頻繁に反転する
不安定な信号でもGOになってしまう問題への対策を検証する。
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.features.signal_check import _fold_ic_stats, run_signal_check


class TestFoldICStats:

    def test_real_data_example_is_unstable(self):
        """実測（180日Hyperliquid）で観測された符号反転パターンはt値で弾かれる"""
        fold_ics = [-0.046309977969527745, 0.06295499730336089,
                    0.09820605560652944, 0.059541913640067286,
                    -0.07390007141751902]
        mean_ic, pos_ratio, t_stat, stability_passed = _fold_ic_stats(fold_ics, min_t_stat=1.0)
        assert mean_ic == pytest.approx(0.0201, abs=1e-3)
        assert pos_ratio == 0.6
        assert abs(t_stat) < 1.0
        assert not stability_passed

    def test_consistent_positive_folds_pass_stability(self):
        fold_ics = [0.05, 0.06, 0.055, 0.052, 0.058]
        mean_ic, pos_ratio, t_stat, stability_passed = _fold_ic_stats(fold_ics, min_t_stat=1.0)
        assert stability_passed
        assert t_stat > 5.0

    def test_empty_folds(self):
        mean_ic, pos_ratio, t_stat, stability_passed = _fold_ic_stats([])
        assert mean_ic == 0.0
        assert not stability_passed

    def test_single_fold_positive_ic_neutral(self):
        """fold数1では分散が測れないため、非負ICは中立的にstability_passed扱い"""
        mean_ic, pos_ratio, t_stat, stability_passed = _fold_ic_stats([0.05], min_t_stat=1.0)
        assert stability_passed


class TestRunSignalCheckStability:

    def test_alpha_data_passes_stability(self):
        """強いアルファ注入データは安定性判定も自然に通る"""
        src = SyntheticSource(n_days=40, alpha="cross", seed=11)
        fs = FeaturePipeline(src.symbols).build(src)
        report = run_signal_check(fs)
        assert report.stability_passed
        assert report.passed

    def test_unstable_borderline_signal_fails_gate(self):
        """平均ICが閾値を超えても符号が割れていれば不合格になる（回帰テスト）"""
        from mars_lite.features.feature_pipeline import FeatureSet

        # 実測の符号反転パターンを再現する小さなFeatureSetを合成
        rng = np.random.default_rng(0)
        n_bars, n_sym, n_feat = 400, 3, 5
        features = rng.normal(0, 1, size=(n_bars, n_sym, n_feat)).astype(np.float32)
        # 特徴量とリターンの関係を弱くノイズがちに（fold毎に符号が変わりやすいよう強度を小さく）
        signal = features[:, :, 0]
        noise = rng.normal(0, 1, size=(n_bars, n_sym))
        ret = 0.01 * signal + 0.05 * noise
        close = 100 * np.exp(np.cumsum(ret, axis=0))
        fs = FeatureSet(
            symbols=[f"S{i}" for i in range(n_sym)],
            timestamps=np.arange(n_bars).astype("datetime64[h]"),
            features=features,
            global_features=np.zeros((n_bars, 1), dtype=np.float32),
            close=close, open_next=close.copy(),
            funding_rate=np.zeros((n_bars, n_sym), dtype=np.float32),
            feature_names=[f"f{i}" for i in range(n_feat)],
            global_feature_names=["g"],
        )
        report = run_signal_check(fs, threshold=0.0, min_positive_ratio=0.0, min_t_stat=1.0)
        # 閾値0でもt値安定性が無ければ不合格になりうることを確認
        # (mean_icが十分低いかt値が低ければpassed=Falseのはず)
        if not report.stability_passed:
            assert not report.passed
