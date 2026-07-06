"""執行コストモデルのテスト"""

import numpy as np

from mars_lite.trading.execution import ExecutionModel, make_execution_model


class TestExecutionModel:
    def test_sqrt_impact_superlinear(self):
        """大口のインパクトは線形を超える（sqrt則: cost ∝ size^1.5）"""
        em = ExecutionModel(fee_rate=0, spread_rate=0, impact_coef=0.001)
        c_small = em.cost_fraction(np.array([0.1, 0, 0]))
        c_big = em.cost_fraction(np.array([0.4, 0, 0]))  # 4倍のサイズ
        # 線形なら4倍、sqrt則なら4^1.5=8倍
        assert c_big / c_small > 4.0
        assert abs(c_big / c_small - 8.0) < 0.5

    def test_twap_splitting_reduces_impact(self):
        """TWAP分割でインパクトが1/√Kに減る"""
        d = np.array([0.4, 0, 0])
        c1 = ExecutionModel(
            fee_rate=0, spread_rate=0, impact_coef=0.001, n_slices=1
        ).cost_fraction(d)
        c4 = ExecutionModel(
            fee_rate=0, spread_rate=0, impact_coef=0.001, n_slices=4
        ).cost_fraction(d)
        assert abs(c4 - c1 / 2.0) < 1e-9  # 1/sqrt(4) = 1/2

    def test_fee_spread_linear(self):
        """手数料・スプレッドは回転率に線形"""
        em = ExecutionModel(fee_rate=0.0005, spread_rate=0.0002, impact_coef=0)
        c = em.cost_fraction(np.array([0.1, 0.1, 0, 0]))  # turnover 0.2
        assert abs(c - 0.2 * 0.0007) < 1e-12

    def test_cost_multiplier(self):
        em = make_execution_model(impact_rate=0.0001)
        d = np.array([0.2, 0, 0])
        assert (
            abs(em.with_multiplier(2.0).cost_fraction(d) - 2 * em.cost_fraction(d))
            < 1e-12
        )

    def test_env_uses_execution_model(self):
        """環境が執行モデルを使い、コスト2倍で資産が減る"""
        from mars_lite.data.sources import SyntheticSource
        from mars_lite.env.portfolio_env import PortfolioTradingEnv
        from mars_lite.features.feature_pipeline import FeaturePipeline

        src = SyntheticSource(n_days=15, alpha="cross", seed=1)
        fs = FeaturePipeline(src.symbols).build(src)
        vals = []
        for mult in [1.0, 2.0]:
            env = PortfolioTradingEnv(fs, episode_bars=30, cost_multiplier=mult)
            env.reset(seed=2, options={"start_idx": 10})
            for i in range(30):
                a = np.full(env.n_symbols, 0.5 if i % 2 == 0 else -0.5)
                _, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
            vals.append(env.portfolio_value)
        assert vals[1] < vals[0]
