"""
crowding（建玉クラウディング戦略）のテスト

要点:
  - dollar-neutral: |Σw|≈0（CSデミーンで厳密保証）
  - グロス上限、符号（高クラウディング=ショート）
  - 因果性: fs.features[t] のみに依存し未来を見ない
  - use_oi/use_funding どちらも無効はエラー
  - BASELINESに登録済み
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.learning.baselines import BASELINES
from mars_lite.learning.crowding import make_crowding_strategy


def _fs(days=90, alpha="cross", seed=3):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


class TestRegistration:
    def test_registered_in_baselines(self):
        assert "crowding" in BASELINES


class TestConfig:
    def test_requires_at_least_one_signal(self):
        with pytest.raises(ValueError):
            make_crowding_strategy(use_oi=False, use_funding=False)


class TestDollarNeutral:
    def test_net_exposure_near_zero(self):
        fs = _fs()
        fn = make_crowding_strategy(use_oi=True)
        w = np.zeros(fs.n_symbols)
        for t in range(0, fs.n_bars - 2, 24):
            w = fn(fs, t, w)
            assert abs(float(w.sum())) < 1e-9

    def test_gross_matches_target(self):
        fs = _fs()
        for gross in (0.5, 1.0):
            fn = make_crowding_strategy(use_oi=True, gross=gross)
            w = fn(fs, 240, np.zeros(fs.n_symbols))
            if np.abs(w).sum() > 0:
                assert abs(float(np.abs(w).sum()) - gross) < 1e-9


class TestSignConvention:
    def test_highest_crowding_symbol_gets_negative_weight(self):
        """csz_oiを人為的に単調増加させ、最高クラウディング銘柄が最も負の
        ウェイト（ショート）になることを確認する。"""
        fs = _fs()
        i_oi = fs.feature_names.index("csz_oi")
        n = fs.n_symbols
        fs.features[:, :, i_oi] = np.tile(
            np.arange(n, dtype=np.float64), (fs.n_bars, 1)
        )
        fn = make_crowding_strategy(use_oi=True, use_funding=False)
        w = fn(fs, 240, np.zeros(n))
        assert w[-1] == w.min()  # 最高クラウディング=最も負
        assert w[0] == w.max()  # 最低クラウディング=最も正


class TestCausality:
    def test_unaffected_by_future_features(self):
        fs = _fs()
        fn = make_crowding_strategy(use_oi=True)
        t = 240
        w1 = fn(fs, t, np.zeros(fs.n_symbols))

        broken = _fs()
        i_oi = broken.feature_names.index("csz_oi")
        rng = np.random.default_rng(0)
        broken.features[t + 5 :, :, i_oi] = rng.normal(
            size=broken.features[t + 5 :, :, i_oi].shape
        )
        w2 = fn(broken, t, np.zeros(broken.n_symbols))
        np.testing.assert_allclose(w1, w2)
