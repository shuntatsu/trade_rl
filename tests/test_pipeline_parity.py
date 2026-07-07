"""
DecisionPipeline の train/serve一致テスト

env.step の内部計算と、/api/signal/latest が使う外部呼び出しが
同一のパイプライン実装を通る（つまり結果が一致する）ことを保証する。
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.trading.pipeline import DecisionPipeline, MarketView, PortfolioState
from mars_lite.trading.post_processor import make_default_processor


@pytest.fixture(scope="module")
def feature_set():
    src = SyntheticSource(n_days=15, alpha="cross", seed=3)
    return FeaturePipeline(src.symbols).build(src)


class TestPipelineParity:
    def test_env_step_matches_external_pipeline_call(self, feature_set):
        """env.step が内部でコミットするウェイトと、同じ状態を渡して外部から
        DecisionPipeline.target_weights を呼んだ結果が一致することを確認する。
        """
        pp = make_default_processor()
        env = PortfolioTradingEnv(feature_set, episode_bars=50, post_processor=pp)
        env.reset(seed=7)

        rng = np.random.default_rng(0)
        for _ in range(10):
            action = rng.uniform(-1, 1, env.n_symbols)

            # env.stepが使う内部状態のスナップショット（step呼び出し前）
            prev = env.weights.copy()
            portfolio_value = env.portfolio_value
            peak_value = env.peak_value
            disagreement = env.disagreement
            t = env.t
            proj = env.project_weights(action.astype(np.float64))

            _, _, terminated, truncated, _ = env.step(action)
            committed = env.weights.copy()

            # 外部からの同一呼び出し
            pipeline = DecisionPipeline(post_processor=pp)
            state = PortfolioState(
                weights=prev, portfolio_value=portfolio_value,
                peak_value=peak_value, disagreement=disagreement,
            )
            market = MarketView.from_feature_set(feature_set, t, vol_lookback=pp.cfg.vol_lookback)
            target, _ = pipeline.target_weights(proj, state, market)

            np.testing.assert_allclose(target, committed, atol=1e-12)

            if terminated or truncated:
                break

    def test_serve_style_call_matches_env_pipeline(self, feature_set):
        """/api/signal/latest 相当の呼び出し（stateless: prev=0, drawdown=0）が
        env内部の同条件呼び出しと一致することを確認する。
        """
        pp = make_default_processor()
        env = PortfolioTradingEnv(feature_set, episode_bars=10, post_processor=pp)
        env.reset(options={"start_idx": max(feature_set.n_bars - 3, 0)})

        raw_action = np.array([0.3, -0.2, 0.1, 0.0, -0.1, 0.2, -0.05])[: env.n_symbols]
        raw_weights = PortfolioTradingEnv.project_weights(raw_action)

        # serve側の呼び出し（mars_lite/server/signal_server.py get_latest_signalと同一形）
        serve_pipeline = DecisionPipeline(post_processor=pp)
        serve_state = PortfolioState(weights=np.zeros(env.n_symbols))
        serve_market = MarketView.from_feature_set(feature_set, env.t, vol_lookback=pp.cfg.vol_lookback)
        serve_target, _ = serve_pipeline.target_weights(raw_weights, serve_state, serve_market)

        # env.step相当の呼び出し（同じ状態: prev=0, drawdown=0, disagreement=0）
        env_pipeline = DecisionPipeline(post_processor=pp)
        env_state = PortfolioState(
            weights=env.weights.copy(), portfolio_value=env.portfolio_value,
            peak_value=env.peak_value, disagreement=env.disagreement,
        )
        env_market = MarketView.from_feature_set(feature_set, env.t, vol_lookback=pp.cfg.vol_lookback)
        env_target, _ = env_pipeline.target_weights(raw_weights, env_state, env_market)

        np.testing.assert_allclose(serve_target, env_target, atol=1e-12)

    def test_htf_gate_shared_implementation(self, feature_set):
        """env.apply_htf_gate が DecisionPipeline.apply_htf_gate に委譲していることを確認"""
        pp = make_default_processor()
        env = PortfolioTradingEnv(feature_set, episode_bars=50, post_processor=pp, htf_gate=True)
        env.reset(seed=0)
        w = np.array([0.3, -0.2, 0.1, 0.0, -0.1, 0.2, -0.05])[: env.n_symbols]

        gated_via_env = env.apply_htf_gate(w.copy())

        htf = feature_set.features[env.t][:, env._htf_idx]
        gated_via_pipeline = env._pipeline.apply_htf_gate(w.copy(), htf)

        np.testing.assert_allclose(gated_via_env, gated_via_pipeline)
