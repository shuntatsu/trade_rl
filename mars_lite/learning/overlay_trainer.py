"""
リスクオーバーレイRL学習（opt-in、docs/ARCHITECTURE.md §2.8参照）

配分（銘柄間の相対ウェイト）は凍結済みの配分エージェントに任せ、
グロスのスケール（どれだけリスクを取るか）だけを別のRLに学習させる。
現行のRuleRiskOverlay（ボラ目標・DDデリスク・不一致縮小）を置き換える
候補で、P0＋汎用性スイートでルール比の優位性を示すまでは既定にしない。
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.trading.pipeline import MarketView
from mars_lite.trading.post_processor import (
    PortfolioPostProcessor, PostProcessConfig, BARS_PER_YEAR_1H,
)


class _GrossMultAdapter:
    """RiskOverlayEnv専用アダプタ: 外から渡されたgross_multをそのまま適用する

    RiskOverlay プロトコルを満たすが、ボラ/DD/不一致には反応しない
    （それらの判断はRiskOverlayEnvの外側の観測経由でオーバーレイ方策自身に
    委ねられているため、ここは「今回のアクションを適用するだけ」の薄い橋渡し）。
    """

    def __init__(self):
        self.pending_gross_mult = 1.0

    def reset(self) -> None:
        self.pending_gross_mult = 1.0

    def scale(self, w, drawdown, disagreement, recent_returns):
        m = self.pending_gross_mult
        scaled = np.asarray(w, dtype=np.float64) * m
        # 3項目とも同じ値にする理由はRLRiskOverlay.scaleと同じ:
        # 単一のグロス乗数が④⑤⑥全てを代替するため、どれか1つ(dd_scale)だけに
        # 詰め込むと配分エージェントがobs_risk_state付きで学習されていた場合、
        # 次のinner_env観測に「ドローダウン応答だけが動いた」という偽の
        # シグナルが混入する。
        return scaled, {
            "vol_scale": m, "dd_scale": m,
            "disagreement_scale": m, "est_port_vol": 0.0,
        }


class RiskOverlayEnv(gym.Env):
    """凍結した配分エージェントの提案を、学習中のグロス乗数でスケールする環境

    配分エージェントは①EMA平滑②集中上限③レバレッジ射影のみを通した
    ウェイトを提案する（④⑤⑥のリスク調整は行わない）。オーバーレイ方策の
    行動 = グロス乗数[0,1] がその提案をスケールし、⑦no-tradeバンドを経て
    執行される。報酬・コスト・PnLはPortfolioTradingEnvの実装をそのまま使う
    （train/serveで実証済みの経済モデルを再利用し、独自実装のバグを避ける）。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        fs: FeatureSet,
        allocation_agent,
        episode_bars: int = 200,
        target_vol: float = 0.5,
        ema_alpha: float = 0.5,
        max_weight: float = 0.4,
        no_trade_band: float = 0.04,
        lambda_turnover: float = 0.04,
        reward_scale: float = 100.0,
        vol_lookback: int = 48,
        bars_per_year: int = BARS_PER_YEAR_1H,
        **env_kwargs,
    ):
        super().__init__()
        self.allocation_agent = allocation_agent
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.bars_per_year = bars_per_year

        self._adapter = _GrossMultAdapter()
        # ④⑤⑥は無効化（dd_derisk_start=1.0は事実上発火せず、disagreement_penalty=0で不一致は無視）。
        # risk_overlayが④⑤⑥の代わりにgross_multを直接適用する。
        pp_cfg = PostProcessConfig(
            ema_alpha=ema_alpha, max_weight=max_weight, no_trade_band=no_trade_band,
            target_vol=None, dd_derisk_start=1.0, disagreement_penalty=0.0,
        )
        pp = PortfolioPostProcessor(pp_cfg, risk_overlay=self._adapter)
        self.inner_env = PortfolioTradingEnv(
            fs, post_processor=pp, episode_bars=episode_bars,
            lambda_turnover=lambda_turnover, reward_scale=reward_scale, **env_kwargs,
        )

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32,
        )
        self._alloc_obs = None
        self._cached_alloc_action = None

    def _refresh_obs_and_cache_action(self) -> np.ndarray:
        alloc_action, _ = self.allocation_agent.predict(self._alloc_obs, deterministic=True)
        self._cached_alloc_action = alloc_action
        w_proposed = self.inner_env.project_weights(
            np.asarray(alloc_action, dtype=np.float64).flatten()
        )
        gross = float(np.abs(w_proposed).sum())
        drawdown = 1.0 - self.inner_env.portfolio_value / max(self.inner_env.peak_value, 1e-9)
        market = MarketView.from_feature_set(
            self.inner_env.fs, self.inner_env.t, vol_lookback=self.vol_lookback,
        )
        est_vol = 0.0
        ret_mean = ret_std = 0.0
        if market.recent_returns is not None and len(market.recent_returns) >= 5:
            port_ret = market.recent_returns @ w_proposed
            est_vol = float(np.std(port_ret) * np.sqrt(self.bars_per_year))
            ret_mean = float(market.recent_returns.mean())
            ret_std = float(market.recent_returns.std())
        vol_ratio = est_vol / self.target_vol if self.target_vol else 0.0
        # disagreement=0固定: 単独配分エージェントを前提とした簡略化
        # （docs/ARCHITECTURE.md §2.8のdisagreement_dr同様、単独方策には
        # アンサンブル不一致という概念自体が無い）
        return np.array([gross, drawdown, 0.0, vol_ratio, ret_mean, ret_std], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._alloc_obs, info = self.inner_env.reset(seed=seed, options=options)
        obs = self._refresh_obs_and_cache_action()
        return obs, info

    def step(self, action):
        gross_mult = float(np.clip(np.asarray(action, dtype=np.float64).flatten()[0], 0.0, 1.0))
        self._adapter.pending_gross_mult = gross_mult

        obs, reward, terminated, truncated, info = self.inner_env.step(self._cached_alloc_action)
        self._alloc_obs = obs
        next_obs = self._refresh_obs_and_cache_action()
        return next_obs, reward, terminated, truncated, info


def train_risk_overlay(
    fs: FeatureSet,
    allocation_agent,
    timesteps: int = 50_000,
    seed: int = 0,
    target_vol: float = 0.5,
    episode_bars: int = 200,
    learning_rate: float = 3e-4,
    verbose: int = 0,
    **env_kwargs,
):
    """RiskOverlayEnv上でグロス乗数方策をPPOで学習して返す（opt-in）"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def make_env():
        env = RiskOverlayEnv(
            fs, allocation_agent, episode_bars=episode_bars,
            target_vol=target_vol, **env_kwargs,
        )
        env.reset(seed=seed)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])
    agent = PPO(
        "MlpPolicy", vec_env,
        policy_kwargs={"net_arch": [32, 32]},
        learning_rate=learning_rate,
        n_steps=256, batch_size=256, n_epochs=6,
        gamma=0.5, gae_lambda=0.9,
        seed=seed, device="cpu", verbose=verbose,
    )
    agent.learn(total_timesteps=timesteps, progress_bar=False)
    return agent
