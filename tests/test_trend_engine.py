"""
trend_engine（TrendEngine v2）のテスト

最重要は回帰アンカー: 単一lookback=48・ボラスケーリング無効・ボラ目標無効に
設定すると、既存の trend_following_strategy と全時刻で厳密に一致すること。
これが崩れると「拡張が既存の実証済み挙動を包含している」という前提が壊れる。
"""

import numpy as np

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.learning.baselines import BASELINES, trend_following_strategy
from mars_lite.learning.trend_engine import TREND_V2_GRID, make_trend_engine_v2


def _fs(days=90, alpha="cross", seed=3):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


class TestRegressionAnchor:
    def test_matches_trend_following_exactly_when_reduced_to_v1_config(self):
        fs = _fs()
        anchor = make_trend_engine_v2(
            lookbacks=(48,),
            vol_scale_symbols=False,
            target_vol=None,
            rebalance_every=24,
        )
        w_a = np.zeros(fs.n_symbols)
        w_b = np.zeros(fs.n_symbols)
        for t in range(fs.n_bars - 2):
            wa = anchor(fs, t, w_a)
            wb = trend_following_strategy(fs, t, w_b)
            np.testing.assert_allclose(wa, wb, atol=1e-12)
            w_a, w_b = wa, wb


class TestRegistration:
    def test_registered_in_baselines(self):
        assert "trend_v2" in BASELINES


class TestMultiHorizon:
    def test_multiple_lookbacks_change_signal_vs_single(self):
        """複数ホライズン合成は単一ホライズンと異なる出力を持ちうる
        （合成が実際に効いていることの確認）。"""
        fs = _fs()
        single = make_trend_engine_v2(
            lookbacks=(48,), vol_scale_symbols=False, target_vol=None
        )
        multi = make_trend_engine_v2(
            lookbacks=(24, 72, 168), vol_scale_symbols=False, target_vol=None
        )
        diffs = []
        w1, w2 = np.zeros(fs.n_symbols), np.zeros(fs.n_symbols)
        for t in range(0, fs.n_bars - 2, 24):
            w1 = single(fs, t, w1)
            w2 = multi(fs, t, w2)
            diffs.append(np.abs(w1 - w2).sum())
        assert max(diffs) > 1e-6


class TestVolScaling:
    def test_gross_never_exceeds_one(self):
        fs = _fs()
        fn = make_trend_engine_v2()
        w = np.zeros(fs.n_symbols)
        for t in range(0, fs.n_bars - 2, 24):
            w = fn(fs, t, w)
            assert float(np.abs(w).sum()) <= 1.0 + 1e-9

    def test_high_volatility_shrinks_gross_via_vol_target(self):
        """合成データの後半を高ボラに書き換えると、ボラ目標がグロスを
        縮小させる（down-only、実現ボラ超過時のみ）。"""
        fs = _fs(days=120, seed=9)
        rng = np.random.default_rng(0)
        cut = fs.n_bars // 2
        # 後半を極端な高ボラのランダムウォークに置き換える
        fs.close[cut:] = fs.close[cut] * np.exp(
            np.cumsum(rng.normal(0, 0.08, size=fs.close[cut:].shape), axis=0)
        )
        fn_targeted = make_trend_engine_v2(target_vol=0.30)
        fn_untargeted = make_trend_engine_v2(target_vol=None)

        gross_targeted, gross_untargeted = [], []
        wt, wu = np.zeros(fs.n_symbols), np.zeros(fs.n_symbols)
        for t in range(cut + 200, fs.n_bars - 2, 24):
            wt = fn_targeted(fs, t, wt)
            wu = fn_untargeted(fs, t, wu)
            gross_targeted.append(np.abs(wt).sum())
            gross_untargeted.append(np.abs(wu).sum())
        assert np.mean(gross_targeted) < np.mean(gross_untargeted)


class TestCausality:
    def test_unaffected_by_future_data(self):
        fs = _fs()
        fn = make_trend_engine_v2()
        t = 200
        w1 = fn(fs, t, np.zeros(fs.n_symbols))

        broken = _fs()
        rng = np.random.default_rng(0)
        broken.close[t + 5 :] = broken.close[t + 5] * np.exp(
            np.cumsum(rng.normal(0, 0.05, size=broken.close[t + 5 :].shape), axis=0)
        )
        w2 = fn(broken, t, np.zeros(broken.n_symbols))
        np.testing.assert_allclose(w1, w2)


class TestGrid:
    def test_grid_entries_are_constructible(self):
        fs = _fs()
        for cfg in TREND_V2_GRID:
            fn = make_trend_engine_v2(**cfg)
            w = fn(fs, 100, np.zeros(fs.n_symbols))
            assert w.shape == (fs.n_symbols,)
            assert np.all(np.isfinite(w))
