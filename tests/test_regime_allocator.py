"""
regime_allocator（因果的な適応スリーブ配分）のテスト

最重要は因果性: 時刻tの配分は過去のスリーブ実績のみに依存し、未来を見ない。
加えて precompute の因果性、勝者への配分傾斜、グロス上限を検証する。
"""

import numpy as np

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.learning.regime_allocator import (
    _precompute_sleeve_bar_returns,
    make_adaptive_allocator,
)


def _fs(days=90, alpha="cross", seed=3):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


def _const_sleeve(sym_idx, gross=0.5):
    """常に指定銘柄へ gross を張るダミースリーブ。"""

    def fn(fs, t, prev):
        w = np.zeros(fs.n_symbols)
        w[sym_idx] = gross
        return w

    return fn


class TestPrecompute:
    def test_bar_returns_are_causal(self):
        """事前計算したバーリターンは、未来の価格を書き換えても t までは不変。"""
        fs = _fs()
        sleeve = _const_sleeve(0)
        r1 = _precompute_sleeve_bar_returns(fs, sleeve)

        broken = _fs()
        cut = fs.n_bars // 2
        rng = np.random.default_rng(0)
        broken.close[cut:] = broken.close[cut] * np.exp(
            np.cumsum(rng.normal(0, 0.05, size=broken.close[cut:].shape), axis=0)
        )
        r2 = _precompute_sleeve_bar_returns(broken, sleeve)
        # cutより前のバーリターンは一致（バーリターン[s]はclose[s-1..s]依存）
        np.testing.assert_allclose(r1[:cut], r2[:cut])


class TestAllocation:
    def test_allocates_toward_better_sleeve(self):
        """常に勝つスリーブと常に負けるスリーブを与えると、十分な履歴後は
        勝者へより多く配分する（follow-the-winner）。"""
        fs = _fs()
        n = fs.n_symbols
        i_win = int(np.argmax(fs.close[-1] / fs.close[0]))  # 最も上昇した銘柄
        i_lose = int(np.argmin(fs.close[-1] / fs.close[0]))

        winner = _const_sleeve(i_win, gross=0.5)
        loser = _const_sleeve(i_lose, gross=0.5)
        alloc_fn = make_adaptive_allocator(
            [("win", winner), ("lose", loser)],
            lookback=336,
            min_lookback=100,
            temperature=0.02,
        )
        t = fs.n_bars - 5
        w = alloc_fn(fs, t, np.zeros(n))
        # 勝者銘柄への配分が敗者銘柄より大きい（符号込み）
        assert w[i_win] > w[i_lose]


class TestGross:
    def test_gross_never_exceeds_one(self):
        fs = _fs()
        sleeves = [("a", _const_sleeve(0, 0.8)), ("b", _const_sleeve(1, 0.8))]
        fn = make_adaptive_allocator(sleeves, lookback=200, min_lookback=80)
        for t in range(100, fs.n_bars - 2, 24):
            w = fn(fs, t, np.zeros(fs.n_symbols))
            assert float(np.abs(w).sum()) <= 1.0 + 1e-9


class TestCausality:
    def test_allocation_unaffected_by_future(self):
        fs = _fs()
        sleeves = [("a", _const_sleeve(0)), ("b", _const_sleeve(1))]
        fn = make_adaptive_allocator(sleeves, lookback=200, min_lookback=80)
        t = 300
        w1 = fn(fs, t, np.zeros(fs.n_symbols))

        broken = _fs()
        rng = np.random.default_rng(1)
        broken.close[t + 5 :] = broken.close[t + 5] * np.exp(
            np.cumsum(rng.normal(0, 0.05, size=broken.close[t + 5 :].shape), axis=0)
        )
        fn2 = make_adaptive_allocator(sleeves, lookback=200, min_lookback=80)
        w2 = fn2(broken, t, np.zeros(broken.n_symbols))
        np.testing.assert_allclose(w1, w2)


class TestWarmup:
    def test_equal_weight_before_min_lookback(self):
        """min_lookback未満の履歴では等ウェイト配分（両スリーブを均等に混ぜる）。"""
        fs = _fs()
        s_a = _const_sleeve(0, gross=1.0)
        s_b = _const_sleeve(1, gross=1.0)
        fn = make_adaptive_allocator(
            [("a", s_a), ("b", s_b)], lookback=200, min_lookback=150, rebalance_every=24
        )
        w = fn(fs, 48, np.zeros(fs.n_symbols))  # 履歴48 < 150
        # 等配分 0.5/0.5 → 銘柄0と1にそれぞれ0.5
        assert abs(w[0] - 0.5) < 1e-9
        assert abs(w[1] - 0.5) < 1e-9
