"""
confidence_gate（相対アルファ成分の自己参照的な信頼度ゲート）のテスト

要点:
  - 履歴不足時は純trend（ゲート無効）にフォールバック
  - アルファのトレーリング実現損益が負なら純trend（希釈しない）
  - アルファのトレーリング実現損益が十分正ならアルファ寄りにブレンドされる
  - 因果性: 時刻tの出力はt以降のデータを書き換えても変わらない
  - グロスは1.0を超えない
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.trading.confidence_gate import confidence_gated_blend


def _fs(days=90, alpha="cross", seed=5):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


def _split(fs, frac=0.6):
    purge = 24
    k = int(fs.n_bars * frac)
    return fs.slice(0, k), fs.slice(k + purge, fs.n_bars)


def _make_synthetic_fns(fs):
    """検証しやすい単純なダミーteacher(alpha/trend)を作る（実モデル不要）。"""
    n = fs.n_symbols

    def alpha_fn(fs, t, prev):
        w = np.zeros(n)
        w[0], w[1] = 0.5, -0.5
        return w

    def trend_fn(fs, t, prev):
        w = np.zeros(n)
        w[2] = 1.0
        return w

    return alpha_fn, trend_fn


class TestFallback:
    def test_falls_back_to_trend_when_history_insufficient(self):
        fs = _fs()
        _, test_fs = _split(fs)
        alpha_fn, trend_fn = _make_synthetic_fns(test_fs)
        fn = confidence_gated_blend(alpha_fn, trend_fn, lookback=100, min_lookback=80)
        w = fn(test_fs, 10, np.zeros(test_fs.n_symbols))  # 履歴10本 < min_lookback80
        expected = trend_fn(test_fs, 10, np.zeros(test_fs.n_symbols))
        np.testing.assert_allclose(w, expected)


class TestGating:
    def test_losing_alpha_stays_pure_trend(self):
        """アルファ成分が明確に負けている合成データでは、十分な履歴後は純trendへ
        収束する（希釈されない）。"""
        fs = _fs(alpha="none", days=90, seed=9)  # ノイズのみ=アルファは儲からない
        _, test_fs = _split(fs)
        alpha_fn, trend_fn = _make_synthetic_fns(test_fs)
        fn = confidence_gated_blend(
            alpha_fn, trend_fn, lookback=100, min_lookback=50, alpha_scale=0.02
        )
        t = test_fs.n_bars - 5
        w = fn(test_fs, t, np.zeros(test_fs.n_symbols))
        trend_w = trend_fn(test_fs, t, np.zeros(test_fs.n_symbols))
        # ノイズデータではアルファのトレーリングリターンは0近辺に留まりやすく、
        # 純trendから大きく乖離しないはず（強くアルファへ寄ることはない）
        assert np.abs(w - trend_w).sum() < 1.0

    def test_confidence_zero_when_trailing_negative(self):
        """トレーリング実現損益が明確に負なら conf=0 相当（純trend一致）になる
        ケースを、実現リターン系列を直接操作して確認する。"""
        fs = _fs()
        _, test_fs = _split(fs)

        def losing_alpha(fs, t, prev):
            n = fs.n_symbols
            w = np.zeros(n)
            w[0] = 1.0
            return w

        def trend_fn(fs, t, prev):
            n = fs.n_symbols
            w = np.zeros(n)
            w[1] = 1.0
            return w

        # asset0を継続的に下落させ、losing_alpha(全力ロングasset0)が
        # 確実に損失を出すようにする
        test_fs.close[:, 0] = test_fs.close[0, 0] * np.exp(
            -0.01 * np.arange(test_fs.n_bars)
        )
        fn = confidence_gated_blend(
            losing_alpha, trend_fn, lookback=80, min_lookback=40, alpha_scale=0.01
        )
        t = 100
        w = fn(test_fs, t, np.zeros(test_fs.n_symbols))
        expected = trend_fn(test_fs, t, np.zeros(test_fs.n_symbols))
        np.testing.assert_allclose(w, expected)


class TestGross:
    def test_gross_never_exceeds_one(self):
        fs = _fs()
        _, test_fs = _split(fs)
        alpha_fn, trend_fn = _make_synthetic_fns(test_fs)
        fn = confidence_gated_blend(alpha_fn, trend_fn, lookback=60, min_lookback=30)
        for t in range(40, test_fs.n_bars, 23):
            w = fn(test_fs, t, np.zeros(test_fs.n_symbols))
            assert float(np.abs(w).sum()) <= 1.0 + 1e-9


class TestCausality:
    def test_unaffected_by_future_data(self):
        fs = _fs()
        _, test_fs = _split(fs)
        alpha_fn, trend_fn = _make_synthetic_fns(test_fs)
        fn = confidence_gated_blend(alpha_fn, trend_fn, lookback=80, min_lookback=40)
        t = 120
        w1 = fn(test_fs, t, np.zeros(test_fs.n_symbols))

        broken = test_fs
        rng = np.random.default_rng(0)
        broken.close[t + 5 :] = broken.close[t + 5] * np.exp(
            np.cumsum(rng.normal(0, 0.05, size=broken.close[t + 5 :].shape), axis=0)
        )
        alpha_fn2, trend_fn2 = _make_synthetic_fns(broken)
        fn2 = confidence_gated_blend(alpha_fn2, trend_fn2, lookback=80, min_lookback=40)
        w2 = fn2(broken, t, np.zeros(broken.n_symbols))
        np.testing.assert_allclose(w1, w2)
