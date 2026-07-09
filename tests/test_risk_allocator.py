"""
risk_allocator（skfolio HRPによるリスクベース配分、Phase2）のテスト

要点:
  - hrp_weights: 正でsum=1、履歴不足では None（呼び出し側が中立フォールバック）
  - risk_parity_scaled: 元のグロスを保持したまま銘柄間配分だけを組み替える
  - 因果性: 適合は fs.close[:t] のみに依存し未来を見ない
  - 相関の高い資産ペアはHRPで一方が縮小される（分散効果の確認）
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.trading.risk_allocator import hrp_weights, risk_parity_scaled


def _fs(days=90, alpha="cross", seed=5):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


def _split(fs, frac=0.6):
    purge = 24
    k = int(fs.n_bars * frac)
    return fs.slice(0, k), fs.slice(k + purge, fs.n_bars)


class TestHRPWeights:
    def test_weights_positive_sum_to_one(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, size=(200, 6))
        w = hrp_weights(rets)
        assert w is not None
        assert np.all(w >= 0)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_insufficient_history_returns_none(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, size=(5, 6))  # 明らかに不足
        assert hrp_weights(rets) is None

    def test_correlated_pair_gets_diversified(self):
        """強く相関する2資産のうち一方に極端な配分集中が起きない
        （HRPが冗長なリスクを圧縮する挙動の確認）。"""
        rng = np.random.default_rng(3)
        n, t = 5, 300
        common = rng.normal(0, 0.01, size=t)
        rets = rng.normal(0, 0.01, size=(t, n))
        rets[:, 0] = common + rng.normal(0, 0.001, size=t)  # asset0とasset1は
        rets[:, 1] = common + rng.normal(0, 0.001, size=t)  # ほぼ同じ動き（強相関）
        w = hrp_weights(rets)
        assert w is not None
        # 強相関ペアの合計配分が、無相関資産1本あたりの配分の2倍を大きく超えない
        # （＝両方に満額を張らず、実質的に1本分程度に圧縮される）
        avg_other = w[2:].mean()
        assert (w[0] + w[1]) < 2.5 * avg_other


class TestRiskParityScaled:
    def test_preserves_gross(self):
        fs = _fs()
        train_fs, test_fs = _split(fs)

        def dummy_teacher(fs, t, prev):
            n = fs.n_symbols
            w = np.zeros(n)
            w[0], w[1] = 0.5, -0.5
            return w

        fn = risk_parity_scaled(dummy_teacher, lookback=96, min_lookback=60)
        for t in range(100, test_fs.n_bars, 40):
            w = fn(test_fs, t, np.zeros(test_fs.n_symbols))
            assert abs(float(np.abs(w).sum()) - 1.0) < 1e-6

    def test_falls_back_when_history_insufficient(self):
        fs = _fs()
        train_fs, test_fs = _split(fs)

        def dummy_teacher(fs, t, prev):
            n = fs.n_symbols
            w = np.zeros(n)
            w[0], w[1] = 0.3, -0.3
            return w

        fn = risk_parity_scaled(dummy_teacher, lookback=96, min_lookback=60)
        w_early = fn(test_fs, 5, np.zeros(test_fs.n_symbols))  # 履歴不足
        w_raw = dummy_teacher(test_fs, 5, np.zeros(test_fs.n_symbols))
        np.testing.assert_allclose(w_early, w_raw)

    def test_zero_gross_passthrough(self):
        fs = _fs()
        _, test_fs = _split(fs)
        zero_teacher = lambda fs, t, prev: np.zeros(fs.n_symbols)  # noqa: E731
        fn = risk_parity_scaled(zero_teacher)
        w = fn(test_fs, 100, np.zeros(test_fs.n_symbols))
        assert np.allclose(w, 0.0)


class TestCausality:
    def test_unaffected_by_future_data(self):
        """時刻tの出力は、t以降の価格を書き換えても変わらない。"""
        fs = _fs()
        _, test_fs = _split(fs)

        def dummy_teacher(fs, t, prev):
            n = fs.n_symbols
            w = np.zeros(n)
            w[0], w[1] = 0.4, -0.4
            return w

        fn = risk_parity_scaled(dummy_teacher, lookback=96, min_lookback=60)
        t = 150
        w1 = fn(test_fs, t, np.zeros(test_fs.n_symbols))

        broken = test_fs
        rng = np.random.default_rng(0)
        broken.close[t + 5 :] = broken.close[t + 5] * np.exp(
            np.cumsum(rng.normal(0, 0.05, size=broken.close[t + 5 :].shape), axis=0)
        )
        w2 = fn(broken, t, np.zeros(broken.n_symbols))
        np.testing.assert_allclose(w1, w2)
