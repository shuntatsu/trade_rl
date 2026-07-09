"""
carry（fundingキャリー戦略）のテスト

要点:
  - dollar-neutral: |Σw|≈0（ランクのデミーンにより厳密に保証される）
  - グロス上限: Σ|w| <= gross
  - 符号: 高funding銘柄が負(ショート)、低/負funding銘柄が正(ロング)
  - 因果性: fs.funding_rate[:t+1] のみに依存し未来を見ない
  - BASELINESに登録済み
"""

import numpy as np

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.learning.baselines import BASELINES
from mars_lite.learning.carry import CARRY_GRID, make_carry_strategy


def _fs(days=90, alpha="cross", seed=3):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


class TestRegistration:
    def test_registered_in_baselines(self):
        assert "carry" in BASELINES


class TestDollarNeutral:
    def test_net_exposure_near_zero(self):
        fs = _fs()
        fn = make_carry_strategy(lookback=72)
        w = np.zeros(fs.n_symbols)
        for t in range(0, fs.n_bars - 2, 24):
            w = fn(fs, t, w)
            assert abs(float(w.sum())) < 1e-9

    def test_gross_matches_target(self):
        fs = _fs()
        for gross in (0.3, 0.5, 0.8):
            fn = make_carry_strategy(lookback=72, gross=gross)
            w = fn(fs, 200, np.zeros(fs.n_symbols))
            if np.abs(w).sum() > 0:
                assert abs(float(np.abs(w).sum()) - gross) < 1e-9


class TestSignConvention:
    def test_highest_funding_symbol_gets_negative_weight(self):
        """funding_rateを人為的に単調増加させ、最高fundingの銘柄が最も
        負のウェイト（ショート）になることを確認する。"""
        fs = _fs()
        n = fs.n_symbols
        # 銘柄0が最低、銘柄n-1が最高fundingになるよう固定値で上書き
        fixed = np.tile(np.arange(n, dtype=np.float64), (fs.n_bars, 1))
        fs.funding_rate[:] = fixed
        fn = make_carry_strategy(lookback=72)
        w = fn(fs, 200, np.zeros(n))
        assert w[-1] == w.min()  # 最高funding銘柄が最も負
        assert w[0] == w.max()  # 最低funding銘柄が最も正


class TestInsufficientHistory:
    def test_no_position_when_history_too_short(self):
        fs = _fs()
        fn = make_carry_strategy(lookback=72)
        w = fn(fs, 10, np.zeros(fs.n_symbols))
        assert np.allclose(w, 0.0)


class TestCausality:
    def test_unaffected_by_future_funding(self):
        fs = _fs()
        fn = make_carry_strategy(lookback=72)
        t = 200
        w1 = fn(fs, t, np.zeros(fs.n_symbols))

        broken = _fs()
        rng = np.random.default_rng(0)
        broken.funding_rate[t + 5 :] = rng.normal(
            0, 0.01, size=broken.funding_rate[t + 5 :].shape
        )
        w2 = fn(broken, t, np.zeros(broken.n_symbols))
        np.testing.assert_allclose(w1, w2)


class TestGrid:
    def test_grid_entries_are_constructible(self):
        fs = _fs()
        for cfg in CARRY_GRID:
            fn = make_carry_strategy(**cfg)
            w = fn(fs, 150, np.zeros(fs.n_symbols))
            assert w.shape == (fs.n_symbols,)
            assert np.all(np.isfinite(w))
