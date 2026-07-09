"""
causal_ridge_signal / augment_with_signals（シグナルレイヤー）のテスト

最重要は因果性: 時刻tの信号が未来のデータに依存しないこと。これが破れると
walk-forward検証全体が無効になる（fold分割前に一括計算する設計のため）。
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.features.signal_layer import (
    augment_with_signals,
    causal_ridge_signal,
)


def _fs(days=60, alpha="cross", seed=3):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


class TestCausality:
    def test_signal_unchanged_when_future_modified(self):
        """時刻t以前の信号は、t以降の価格・特徴を書き換えても変わらない"""
        fs = _fs()
        sig_full = causal_ridge_signal(
            fs, horizon=4, min_train_bars=400, refit_every=100, train_window=800
        )

        # 未来（後半）の価格と特徴を破壊した別のFeatureSetを作る
        cut = fs.n_bars // 2
        fs2 = _fs()  # 同一シードで再生成
        rng = np.random.default_rng(99)
        fs2.features[cut:] = rng.normal(size=fs2.features[cut:].shape)
        fs2.close[cut:] = fs2.close[cut] * np.exp(
            np.cumsum(rng.normal(0, 0.05, size=fs2.close[cut:].shape), axis=0)
        )
        sig_broken = causal_ridge_signal(
            fs2, horizon=4, min_train_bars=400, refit_every=100, train_window=800
        )

        # cutより十分手前（refit境界+horizonのマージンを引いた位置）までは
        # 完全一致するはず。マージン: 最後のrefit時点がcut-horizonを跨がない
        # 安全側の位置まで確認する
        safe = cut - 100 - 4  # refit_every + horizon のマージン
        np.testing.assert_array_equal(sig_full[:safe], sig_broken[:safe])

    def test_warmup_is_zero(self):
        fs = _fs()
        sig = causal_ridge_signal(fs, horizon=4, min_train_bars=600, refit_every=100)
        assert np.all(sig[:600] == 0.0)

    def test_signal_has_predictive_power_on_synthetic_alpha(self):
        """合成crossアルファ（学習可能な信号が存在する）で正のICが出る"""
        fs = _fs(days=60, alpha="cross")
        sig = causal_ridge_signal(
            fs, horizon=4, min_train_bars=400, refit_every=100, train_window=800
        )
        # 前方リターンとのランク相関（有効領域のみ、pandasでSpearman相当）
        import pandas as pd

        fwd = np.log(fs.close[4:]) - np.log(fs.close[:-4])
        s = sig[400 : len(fwd)]
        f = fwd[400:]
        mask = s.ravel() != 0.0
        # Spearman = 順位系列のPearson相関（scipy非依存で計算）
        rs = pd.Series(s.ravel()[mask]).rank()
        rf = pd.Series(f.ravel()[mask]).rank()
        ic = rs.corr(rf)
        assert ic > 0.05, f"IC too low: {ic}"


class TestAugment:
    def test_append_adds_channels(self):
        fs = _fs(days=30)
        sig = np.zeros((fs.n_bars, fs.n_symbols), dtype=np.float32)
        fs2 = augment_with_signals(fs, sig, signal_names=["ridge_alpha_h4"])
        assert fs2.n_features == fs.n_features + 1
        assert fs2.feature_names[-1] == "ridge_alpha_h4"
        # 価格系は不変
        np.testing.assert_array_equal(fs2.close, fs.close)

    def test_only_replaces_features(self):
        fs = _fs(days=30)
        sig = np.ones((fs.n_bars, fs.n_symbols), dtype=np.float32)
        fs2 = augment_with_signals(fs, sig, signal_names=["a"], only=True)
        assert fs2.n_features == 1
        assert fs2.feature_names == ["a"]
        np.testing.assert_array_equal(fs2.features[:, :, 0], sig)

    def test_shape_mismatch_raises(self):
        fs = _fs(days=30)
        bad = np.zeros((10, fs.n_symbols), dtype=np.float32)
        with pytest.raises(ValueError):
            augment_with_signals(fs, bad)


class TestCliWiring:
    def test_cli_exposes_signal_layer(self):
        from mars_lite.pipeline.cli import build_parser

        p = build_parser()
        assert p.parse_args([]).signal_layer == "off"
        a = p.parse_args(
            [
                "--signal-layer",
                "only",
                "--signal-train-window",
                "3000",
                "--signal-refit-every",
                "200",
            ]
        )
        assert a.signal_layer == "only"
        assert a.signal_train_window == 3000
        assert a.signal_refit_every == 200

    def test_apply_signal_layer_only_reduces_dims(self):
        from types import SimpleNamespace

        from mars_lite.features.signal_layer import apply_signal_layer

        fs = _fs(days=40)
        args = SimpleNamespace(
            signal_layer="only",
            target="raw",
            signal_train_window=800,
            signal_refit_every=100,
        )
        fs2 = apply_signal_layer(args, fs, horizon=4)
        assert fs2.n_features == 1
        assert fs2.feature_names == ["ridge_alpha_h4"]
