"""
レジーム特化アンサンブル（項目3）のテスト
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.learning.regime_ensemble import (
    classify_trend, regime_labels, regime_start_pools, RegimeEnsemble, REGIMES,
)


@pytest.fixture(scope="module")
def fs_bull():
    src = SyntheticSource(n_days=90, alpha="bull", alpha_strength=0.001, seed=1)
    return FeaturePipeline(src.symbols).build(src)


class TestRegimeLabels:

    def test_classify_thresholds(self):
        assert classify_trend(1.0) == "bull"
        assert classify_trend(-1.0) == "bear"
        assert classify_trend(0.0) == "range"

    def test_labels_cover_all_bars(self, fs_bull):
        labels = regime_labels(fs_bull)
        assert labels.shape[0] == fs_bull.n_bars
        assert set(np.unique(labels)).issubset(set(REGIMES))

    def test_bull_market_mostly_bull(self, fs_bull):
        """強気合成データでは bull ラベルが range/bear より多い"""
        labels = regime_labels(fs_bull)
        counts = {r: int((labels == r).sum()) for r in REGIMES}
        assert counts["bull"] >= counts["bear"]

    def test_start_pools_partition(self, fs_bull):
        pools = regime_start_pools(fs_bull, horizon=240)
        # 各プールは有効な開始位置
        max_start = fs_bull.n_bars - 240 - 2
        for r in REGIMES:
            assert (pools[r] <= max_start).all()
        total = sum(len(pools[r]) for r in REGIMES)
        assert total > 0


class _StubAgent:
    """predictが自分のタグを返すだけのスタブ（ルーティング検証用）"""
    def __init__(self, tag, n):
        self.tag = tag
        self.n = n
        self.device = "cpu"

    def predict(self, obs, deterministic=True):
        return np.full(self.n, self.tag, dtype=np.float32), None


class TestRegimeRouting:

    def test_routes_to_matching_specialist(self, fs_bull):
        env = PortfolioTradingEnv(fs_bull)
        n = fs_bull.n_symbols
        specialists = {"bull": _StubAgent(1.0, n), "bear": _StubAgent(2.0, n),
                       "range": _StubAgent(3.0, n)}
        ens = RegimeEnsemble(specialists, generalist=None,
                             obs_layout=env.obs_layout,
                             n_raw_globals=fs_bull.global_features.shape[1])

        # btc_trend を強制的に強気/弱気/レンジへ設定した観測を作る
        obs, _ = env.reset(options={"start_idx": 100})
        pos = ens._trend_pos
        for trend_z, expect in [(2.0, 1.0), (-2.0, 2.0), (0.0, 3.0)]:
            o = obs.copy()
            o[pos] = trend_z
            action, _ = ens.predict(o)
            assert action[0] == expect

    def test_generalist_fallback_when_missing(self, fs_bull):
        env = PortfolioTradingEnv(fs_bull)
        n = fs_bull.n_symbols
        ens = RegimeEnsemble({"bull": _StubAgent(1.0, n)},
                             generalist=_StubAgent(9.0, n),
                             obs_layout=env.obs_layout,
                             n_raw_globals=fs_bull.global_features.shape[1])
        obs, _ = env.reset(options={"start_idx": 100})
        o = obs.copy()
        o[ens._trend_pos] = -2.0  # bear specialist無し → generalist
        action, _ = ens.predict(o)
        assert action[0] == 9.0


class TestRegimeStartPoolEnv:

    def test_env_samples_from_pool(self, fs_bull):
        pool = np.array([50, 51, 52], dtype=np.int64)
        env = PortfolioTradingEnv(fs_bull, episode_bars=100,
                                  regime_start_pool=pool)
        for seed in range(10):
            env.reset(seed=seed)
            assert env.start_idx in set(pool.tolist())
