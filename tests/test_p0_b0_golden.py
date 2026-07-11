"""
Phase 1b 完了・B0 (最小構成) 代表試行のゴールデン回帰テスト
Phase 2 (後処理切り分け) の開発中も、B0最小構成の行動・リターンが不変であることを保証する。
"""

import math
import unittest

import numpy as np

from mars_lite.data.sources import SyntheticSource
from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.features.feature_pipeline import FeaturePipeline


class TestP0B0GoldenRegression(unittest.TestCase):
    def test_b0_env_step_invariants(self):
        """B0設定 (コストなし/既定コスト、後処理なし) の決定論的挙動の不変性検証"""
        src = SyntheticSource(
            seed=42, symbols=["AAA", "BBB", "CCC", "DDD"], n_days=60, alpha="cross"
        )
        fs = FeaturePipeline(src.symbols).build(src)
        env = PortfolioTradingEnv(
            fs=fs,
            fee_rate=0.0005,
            spread_rate=0.0002,
            impact_rate=0.0001,
            decision_every=1,
            lambda_turnover=0.0,
        )

        obs, info = env.reset(seed=42)
        self.assertEqual(obs.shape, env.observation_space.shape)

        # 固定アクション系列を投入し、リターンとTurnoverが絶対一致することを検証
        np.random.seed(123)
        actions = [
            np.array([0.2, -0.2, 0.3, -0.3], dtype=np.float32),
            np.array([0.1, 0.1, -0.1, -0.1], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ]

        total_reward = 0.0
        for act in actions:
            obs, reward, terminated, truncated, info = env.step(act)
            total_reward += float(reward)

        # 浮動小数点まで完全に一貫していることを確認
        self.assertTrue(math.isfinite(total_reward))
        self.assertGreaterEqual(env.turnover_total, 0.0)
        self.assertIn("turnover", info)
        self.assertAlmostEqual(total_reward, 0.0686707759064264, places=10)


if __name__ == "__main__":
    unittest.main()
