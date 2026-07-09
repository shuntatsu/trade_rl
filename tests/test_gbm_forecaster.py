"""
gbm_forecaster（LightGBM因果アルファ予測器）のテスト

最重要は因果性: 時刻tの信号が未来のデータに依存しないこと（signal_layerと同じく
fold分割前に一括計算する設計のため）。加えてアルファ有データでICが立つこと、
walk_forward_ic で GBM が動くことを確認する。テストは軽量化のため
num_boost_round を小さくする。
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.features.gbm_forecaster import (
    causal_gbm_signal,
    fit_gbm,
    predict_gbm,
)


def _fs(days=70, alpha="cross", seed=3):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


_FAST = dict(num_boost_round=30)


class TestCausality:
    def test_signal_unchanged_when_future_modified(self):
        """時刻t以前の信号は、t以降の価格・特徴を書き換えても変わらない。"""
        fs = _fs()
        sig_full = causal_gbm_signal(
            fs,
            horizon=4,
            min_train_bars=400,
            refit_every=150,
            train_window=800,
            **_FAST,
        )
        cut = fs.n_bars // 2
        fs2 = _fs()  # 同一シードで再生成
        rng = np.random.default_rng(99)
        fs2.features[cut:] = rng.normal(size=fs2.features[cut:].shape)
        fs2.close[cut:] = fs2.close[cut] * np.exp(
            np.cumsum(rng.normal(0, 0.05, size=fs2.close[cut:].shape), axis=0)
        )
        sig_broken = causal_gbm_signal(
            fs2,
            horizon=4,
            min_train_bars=400,
            refit_every=150,
            train_window=800,
            **_FAST,
        )
        # 最後のrefit境界がcut-horizonを跨がない安全マージンまでは一致
        safe = cut - 150 - 4 - 1
        np.testing.assert_allclose(sig_full[:safe], sig_broken[:safe])


class TestFitPredict:
    def test_fit_predict_shapes(self):
        fs = _fs()
        from mars_lite.features.signal_check import _pool

        X, y, _ = _pool(fs, horizon=4)
        booster = fit_gbm(X, y, num_boost_round=30)
        pred = predict_gbm(booster, X[:100])
        assert pred.shape == (100,)
        assert np.all(np.isfinite(pred))


class TestSignalQuality:
    def test_signal_has_ic_on_alpha_data(self):
        """アルファ有データでは信号が次期リターンと正の順位相関を持つ。"""
        from mars_lite.features.signal_check import _forward_returns, _rank_ic

        fs = _fs(alpha="cross", days=90, seed=7)
        sig = causal_gbm_signal(
            fs, horizon=4, min_train_bars=500, refit_every=200, **_FAST
        )
        fwd = _forward_returns(fs, 4)
        # 信号が立っている（非ゼロ）領域だけで評価
        m = np.isfinite(fwd) & (np.abs(sig) > 1e-9)
        ic = _rank_ic(sig[m], fwd[m])
        assert ic > 0.0


class TestWalkForwardIC:
    def test_gbm_ic_diagnostic_runs(self):
        from mars_lite.eval.gate1_diagnostics import walk_forward_ic

        fs = _fs(alpha="cross", days=90, seed=11)
        rep = walk_forward_ic(fs, horizon=4, target="cs_demean", model="gbm", n_folds=3)
        assert rep["model"] == "gbm"
        assert rep["n_folds"] >= 1
        assert np.isfinite(rep["mean_oos_ic"])

    def test_ridge_mode_also_supported(self):
        from mars_lite.eval.gate1_diagnostics import walk_forward_ic

        fs = _fs(alpha="cross", days=90, seed=11)
        rep = walk_forward_ic(fs, horizon=4, model="ridge", n_folds=3)
        assert rep["model"] == "ridge"
        assert np.isfinite(rep["mean_oos_ic"])
