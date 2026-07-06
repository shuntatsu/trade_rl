"""
ポートフォリオ環境のテスト
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.env.portfolio_env import PortfolioTradingEnv


@pytest.fixture(scope="module")
def feature_set():
    src = SyntheticSource(n_days=15, alpha="cross", seed=3)
    return FeaturePipeline(src.symbols).build(src)


@pytest.fixture
def env(feature_set):
    return PortfolioTradingEnv(feature_set, episode_bars=50)


class TestPortfolioEnv:

    def test_obs_shape(self, env):
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert not np.isnan(obs).any()

    def test_action_space(self, env):
        assert env.action_space.shape == (env.n_symbols,)

    def test_leverage_projection(self):
        w = PortfolioTradingEnv.project_weights(np.ones(7))
        assert abs(np.abs(w).sum() - 1.0) < 1e-9
        w2 = PortfolioTradingEnv.project_weights(np.array([0.1, -0.2, 0, 0, 0, 0, 0]))
        assert abs(np.abs(w2).sum() - 0.3) < 1e-9  # 合計1以下ならそのまま

    def test_episode_runs(self, env):
        env.reset(seed=1)
        done = False
        steps = 0
        while not done and steps < 100:
            _, r, term, trunc, info = env.step(env.action_space.sample())
            assert np.isfinite(r)
            done = term or trunc
            steps += 1
        assert done
        for key in ["win_rate", "max_drawdown", "portfolio_value", "apy",
                    "sharpe", "n_trades", "funding_pnl",
                    "long_pct", "short_pct", "hold_pct"]:
            assert key in info, f"missing episode key: {key}"

    def test_execution_info_contract(self, env):
        """ダッシュボード（TradingVisualizer）契約"""
        env.reset(seed=2)
        _, _, _, _, info = env.step(env.action_space.sample())
        ex = info["execution"]
        for key in ["step", "p_base", "p_exec", "action", "side",
                    "inventory_after", "reward", "event"]:
            assert key in ex
        assert ex["side"] in ("buy", "sell", "hold")
        assert ex["event"] in ("normal", "liquidation")

    def test_hold_action_no_cost(self, env):
        """行動ゼロ（フラット維持）ではコストが発生しない"""
        env.reset(seed=3, options={"start_idx": 0})
        v0 = env.portfolio_value
        _, _, _, _, info = env.step(np.zeros(env.n_symbols))
        assert info["turnover"] == 0.0
        assert env.portfolio_value == pytest.approx(v0)

    def test_trading_costs_reduce_value(self, feature_set):
        """同じ行動系列でコスト2倍なら資産は同等以下"""
        results = []
        for mult in [1.0, 2.0]:
            env = PortfolioTradingEnv(
                feature_set, episode_bars=30, cost_multiplier=mult
            )
            env.reset(seed=5, options={"start_idx": 10})
            for i in range(30):
                a = np.full(env.n_symbols, 0.5 if i % 2 == 0 else -0.5)
                _, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
            results.append(env.portfolio_value)
        assert results[1] < results[0]

    def test_funding_applied(self, feature_set):
        """funding率がポートフォリオ価値に反映される"""
        env = PortfolioTradingEnv(feature_set, episode_bars=100,
                                  fee_rate=0, spread_rate=0, impact_rate=0,
                                  min_trade_delta=0.0)
        env.reset(seed=7, options={"start_idx": 0})
        funding_hits = 0
        for _ in range(60):
            _, _, term, trunc, _ = env.step(np.full(env.n_symbols, 1.0))
            if env.fs.funding_rate[env.t] .any():
                funding_hits += 1
            if term or trunc:
                break
        assert env._funding_pnl != 0.0  # ロング保有でfunding授受が発生

    def test_start_idx_option(self, env):
        env.reset(seed=0, options={"start_idx": 5})
        assert env.start_idx == 5
