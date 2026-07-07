"""
リスクオーバーレイRL学習(mars_lite.learning.overlay_trainer)のテスト

学習の仕組みが最後まで壊れずに動くこと（スモーク）と、RiskOverlayEnvが
グロスを増やさない不変条件を守ることを確認する。P0/汎用性ベンチマークでの
ルール比の優劣評価はここでは行わない（docs/ARCHITECTURE.md §2.8参照）。
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.learning.overlay_trainer import RiskOverlayEnv, train_risk_overlay
from mars_lite.trading.risk_overlay import RLRiskOverlay


@pytest.fixture(scope="module")
def feature_set():
    src = SyntheticSource(n_days=15, alpha="cross", seed=3)
    return FeaturePipeline(src.symbols).build(src)


class _RandomAllocationAgent:
    """配分エージェントのダミー実装（決定論的な固定提案）"""

    def __init__(self, n_symbols, seed=0):
        rng = np.random.default_rng(seed)
        self._action = rng.uniform(-0.3, 0.3, n_symbols)

    def predict(self, obs, deterministic=True):
        return self._action, None


class TestRiskOverlayEnv:
    def test_reset_and_step_run(self, feature_set):
        agent = _RandomAllocationAgent(feature_set.n_symbols)
        env = RiskOverlayEnv(feature_set, agent, episode_bars=30)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (6,)
        assert np.isfinite(obs).all()

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert np.isfinite(reward)
            assert obs.shape == (6,)
            if term or trunc:
                break

    def test_gross_multiplier_never_increases_exposure(self, feature_set):
        agent = _RandomAllocationAgent(feature_set.n_symbols)
        env = RiskOverlayEnv(feature_set, agent, episode_bars=30)
        env.reset(seed=1)

        w_proposed = env.inner_env.project_weights(
            np.asarray(env._cached_alloc_action, dtype=np.float64).flatten()
        )
        gross_proposed = float(np.abs(w_proposed).sum())

        # gross_mult=0.0 -> ポジションは持たない方向に縮小されるはず
        env.step(np.array([0.0]))
        assert float(np.abs(env.inner_env.weights).sum()) <= gross_proposed + 1e-9

    def test_action_space_is_gross_multiplier(self, feature_set):
        agent = _RandomAllocationAgent(feature_set.n_symbols)
        env = RiskOverlayEnv(feature_set, agent, episode_bars=30)
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == 0.0
        assert env.action_space.high[0] == 1.0


class TestTrainRiskOverlay:
    def test_train_runs_end_to_end(self, feature_set):
        """短ステップでの学習が最後まで完走し、有効なエージェントを返すことを確認（スモーク）"""
        agent = _RandomAllocationAgent(feature_set.n_symbols)
        overlay_agent = train_risk_overlay(
            feature_set,
            agent,
            timesteps=512,
            episode_bars=30,
            verbose=0,
        )
        obs = np.zeros(6, dtype=np.float32)
        action, _ = overlay_agent.predict(obs, deterministic=True)
        assert 0.0 <= float(np.asarray(action).flatten()[0]) <= 1.0

    def test_rl_risk_overlay_wraps_trained_agent(self, feature_set):
        agent = _RandomAllocationAgent(feature_set.n_symbols)
        overlay_agent = train_risk_overlay(
            feature_set,
            agent,
            timesteps=512,
            episode_bars=30,
            verbose=0,
        )
        rl_overlay = RLRiskOverlay(overlay_agent, target_vol=0.5)

        w = np.array([0.3, -0.2, 0.1, 0.0, -0.1, 0.2, -0.05])[: feature_set.n_symbols]
        scaled, info = rl_overlay.scale(
            w, drawdown=0.1, disagreement=0.0, recent_returns=None
        )
        assert np.abs(scaled).sum() <= np.abs(w).sum() + 1e-9
        assert 0.0 <= info["dd_scale"] <= 1.0
