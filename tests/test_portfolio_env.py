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


class TestObsRiskState:
    """opt-in観測強化(obs_risk_state)のテスト。既定offでは無効。"""

    def test_default_off_does_not_change_obs_shape(self, feature_set):
        base = PortfolioTradingEnv(feature_set, episode_bars=50)
        assert base.n_global == feature_set.global_features.shape[1] + 3

    def test_opt_in_adds_four_globals(self, feature_set):
        env = PortfolioTradingEnv(feature_set, episode_bars=50, obs_risk_state=True)
        assert env.n_global == feature_set.global_features.shape[1] + 7
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape

    def test_risk_state_reflects_post_processor_output(self, feature_set):
        from mars_lite.trading.post_processor import make_default_processor

        pp = make_default_processor()
        env = PortfolioTradingEnv(
            feature_set, episode_bars=50, post_processor=pp, obs_risk_state=True,
        )
        env.reset(seed=0)
        # リセット直後は中立値（後処理未適用）
        obs0 = env._obs()
        vol_scale, dd_scale, disagreement_scale, est_vol = obs0[-4:]
        assert vol_scale == pytest.approx(1.0)
        assert dd_scale == pytest.approx(1.0)
        assert disagreement_scale == pytest.approx(1.0)
        assert est_vol == pytest.approx(0.0)

        # 大きな行動を繰り返しボラを高めればvol_scaleが1未満に落ちるはず
        for _ in range(10):
            env.step(np.full(env.n_symbols, 1.0))
        obs1 = env._obs()
        assert obs1.shape == env.observation_space.shape
        assert np.isfinite(obs1).all()

    def test_disagreement_dr_disabled_by_default(self, feature_set):
        env = PortfolioTradingEnv(feature_set, episode_bars=50)
        env.reset(seed=0)
        assert env.disagreement == 0.0

    def test_disagreement_dr_samples_when_enabled(self, feature_set):
        env = PortfolioTradingEnv(feature_set, episode_bars=50, disagreement_dr_max=0.3)
        seen = set()
        for seed in range(5):
            env.reset(seed=seed)
            assert 0.0 <= env.disagreement <= 0.3
            seen.add(env.disagreement)
        assert len(seen) > 1  # 複数エピソードで異なる値がサンプルされる

    def test_disagreement_dr_constant_within_episode(self, feature_set):
        env = PortfolioTradingEnv(feature_set, episode_bars=50, disagreement_dr_max=0.3)
        env.reset(seed=1)
        d0 = env.disagreement
        for _ in range(5):
            env.step(np.zeros(env.n_symbols))
            assert env.disagreement == d0

    def test_extractor_adapts_to_enriched_obs(self, feature_set):
        """TFGatedPortfolioExtractorがn_global変化に自動適応することを確認"""
        from mars_lite.models.portfolio_extractor import TFGatedPortfolioExtractor
        from mars_lite.features.feature_pipeline import TF_BLOCK_FEATURES

        env = PortfolioTradingEnv(feature_set, episode_bars=50, obs_risk_state=True)
        tf_prefixes = []
        for name in feature_set.feature_names:
            p = name.split("_")[0]
            if p in ("15m", "30m", "1h", "4h", "1d") and p not in tf_prefixes:
                tf_prefixes.append(p)

        extractor = TFGatedPortfolioExtractor(
            env.observation_space, **env.obs_layout,
            n_tf_blocks=len(tf_prefixes), tf_block_size=len(TF_BLOCK_FEATURES),
            size="small",
        )
        import torch
        obs, _ = env.reset(seed=0)
        out = extractor(torch.as_tensor(obs).unsqueeze(0))
        assert out.shape == (1, 128)  # small preset features_dim
