"""
ポートフォリオ取引環境（gymnasium.Env）

意思決定: 各バーで全銘柄のターゲットウェイトを同時に決定する連続行動空間。
生行動 → レバレッジ1射影 → （任意）後処理（EMA平滑・集中上限・ボラ目標・
DDデリスク・不一致縮小・no-tradeバンド）→ （任意）階層MTFゲート → 執行。

コストモデルは mars_lite.trading.execution.ExecutionModel を使い、
baselines.simulate_strategy / oracle_dp_strategy と厳密に同一の
sqrt-impact + TWAP分割則を適用する（train/serve/backtest一致）。
"""

from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.trading.execution import make_execution_model

BARS_PER_YEAR_1H = 24 * 365


def _project_leverage(w: np.ndarray, max_leverage: float = 1.0) -> np.ndarray:
    """Sigma|w| <= max_leverage への射影（超過分のみ縮小）"""
    w = np.asarray(w, dtype=np.float64).copy()
    gross = np.abs(w).sum()
    if gross > max_leverage:
        return w * (max_leverage / gross)
    return w


class PortfolioTradingEnv(gym.Env):
    """複数銘柄の同時ポジショニングを行うポートフォリオ取引環境"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        fs: FeatureSet,
        episode_bars: int = 200,
        fee_rate: float = 0.0005,
        spread_rate: float = 0.0002,
        impact_rate: float = 0.0001,
        min_trade_delta: float = 0.02,
        cost_multiplier: float = 1.0,
        initial_capital: float = 1.0,
        post_processor=None,
        htf_gate: bool = False,
        htf_threshold: float = 0.3,
        htf_neutral_scale: float = 0.5,
        regime_start_pool: Optional[np.ndarray] = None,
        lambda_turnover: float = 0.02,
        reward_scale: float = 100.0,
        use_dsr: bool = False,
        dsr_eta: float = 0.01,
        decision_every: int = 1,
    ):
        super().__init__()
        self.fs = fs
        self.n_symbols = fs.n_symbols
        self.episode_bars = episode_bars
        self.fee_rate = fee_rate
        self.spread_rate = spread_rate
        self.impact_rate = impact_rate
        self.min_trade_delta = min_trade_delta
        self.cost_multiplier = cost_multiplier
        self.initial_capital = initial_capital
        self.post_processor = post_processor
        self.htf_gate = htf_gate
        self.htf_threshold = htf_threshold
        self.htf_neutral_scale = htf_neutral_scale
        self.regime_start_pool = (
            np.asarray(regime_start_pool) if regime_start_pool is not None else None
        )
        self.lambda_turnover = lambda_turnover
        self.reward_scale = reward_scale
        self.use_dsr = use_dsr
        self.dsr_eta = dsr_eta
        self.decision_every = max(1, decision_every)

        self._exec_model = make_execution_model(
            fee_rate=fee_rate, spread_rate=spread_rate,
            impact_rate=impact_rate, cost_multiplier=cost_multiplier,
        )

        self._htf_idx: Optional[int] = None
        if htf_gate:
            self._htf_idx = fs.feature_names.index("4h_ret_z20")

        self.n_per_symbol = fs.n_features + 1  # 特徴 + 現ウェイト
        self.n_global = fs.global_features.shape[1] + 3  # 生グローバル + ポート状態3

        obs_dim = self.n_symbols * self.n_per_symbol + self.n_global
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_symbols,), dtype=np.float32,
        )

        # 属性の初期化（reset前にアクセスされても壊れないように）
        self.t = 0
        self.start_idx = 0
        self.weights = np.zeros(self.n_symbols)
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.disagreement = 0.0
        self._funding_pnl = 0.0
        self.n_trades = 0
        self.turnover_total = 0.0
        self.max_dd = 0.0
        self._dsr_A = 0.0
        self._dsr_B = 0.0

    # ---- ユーティリティ ----

    @staticmethod
    def project_weights(w: np.ndarray, max_leverage: float = 1.0) -> np.ndarray:
        return _project_leverage(w, max_leverage)

    @property
    def obs_layout(self) -> Dict[str, int]:
        return {
            "n_symbols": self.n_symbols,
            "n_per_symbol": self.n_per_symbol,
            "n_global": self.n_global,
        }

    def apply_htf_gate(self, w: np.ndarray) -> np.ndarray:
        """
        階層MTFゲート: 上位足(4h)トレンドと逆方向のポジションを禁止し、
        トレンド無し(neutral)では縮小する。グロスは増加させない。
        """
        if self._htf_idx is None:
            return w
        htf = self.fs.features[self.t][:, self._htf_idx]
        gated = np.asarray(w, dtype=np.float64).copy()
        for i in range(len(gated)):
            h = float(htf[i])
            if h > self.htf_threshold:
                if gated[i] < 0:
                    gated[i] = 0.0
            elif h < -self.htf_threshold:
                if gated[i] > 0:
                    gated[i] = 0.0
            else:
                gated[i] = gated[i] * self.htf_neutral_scale
        return gated

    def set_reward_mode(self, use_dsr: bool) -> None:
        """CurriculumCallback契約: 報酬モードをPnL/DSR間で切替"""
        self.use_dsr = use_dsr
        self._dsr_A = 0.0
        self._dsr_B = 0.0

    def _obs(self) -> np.ndarray:
        feats = self.fs.features[self.t]  # (n_sym, n_feat)
        per_sym = np.concatenate(
            [feats, self.weights.reshape(-1, 1).astype(np.float32)], axis=1
        ).flatten()
        raw_globals = self.fs.global_features[self.t]
        drawdown = 1.0 - self.portfolio_value / max(self.peak_value, 1e-9)
        gross = float(np.abs(self.weights).sum())
        progress = (self.t - self.start_idx) / max(self.episode_bars, 1)
        port_globals = np.array([drawdown, gross, progress], dtype=np.float32)
        return np.concatenate([per_sym, raw_globals, port_globals]).astype(np.float32)

    def _dsr_reward(self, r: float) -> float:
        eta = self.dsr_eta
        dA = r - self._dsr_A
        dB = r * r - self._dsr_B
        denom = (self._dsr_B - self._dsr_A ** 2)
        if denom > 1e-12:
            dsr = (self._dsr_B * dA - 0.5 * self._dsr_A * dB) / (denom ** 1.5)
        else:
            dsr = 0.0
        self._dsr_A += eta * dA
        self._dsr_B += eta * dB
        return float(dsr) * self.reward_scale

    # ---- gym.Env API ----

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        if "start_idx" in options:
            self.start_idx = int(options["start_idx"])
        elif self.regime_start_pool is not None and len(self.regime_start_pool):
            self.start_idx = int(self.np_random.choice(self.regime_start_pool))
        else:
            max_start = max(0, self.fs.n_bars - self.episode_bars - 2)
            self.start_idx = int(self.np_random.integers(0, max_start + 1))

        self.t = self.start_idx
        self.weights = np.zeros(self.n_symbols)
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.max_dd = 0.0
        self._funding_pnl = 0.0
        self.n_trades = 0
        self.turnover_total = 0.0
        self._returns_history = []
        self._n_hold_steps = 0
        self._long_symbol_steps = 0
        self._short_symbol_steps = 0
        self._total_symbol_steps = 0
        self.disagreement = 0.0
        self._dsr_A = 0.0
        self._dsr_B = 0.0

        return self._obs(), {}

    def step(self, action):
        raw = np.asarray(action, dtype=np.float64).flatten()
        prev = self.weights.copy()

        is_decision_bar = (self.t - self.start_idx) % self.decision_every == 0
        if not is_decision_bar:
            # 非意思決定バー: 前回のウェイトを保持（低頻度アルファ向け）
            proj = prev.copy()
        else:
            proj = self.project_weights(raw)

        if self.post_processor is not None:
            cfg = self.post_processor.cfg
            lb = cfg.vol_lookback
            start = max(0, self.t - lb)
            recent_returns = None
            if self.t > start:
                recent_returns = (
                    np.diff(self.fs.close[start:self.t + 1], axis=0)
                    / self.fs.close[start:self.t, :]
                )
            drawdown = 1.0 - self.portfolio_value / max(self.peak_value, 1e-9)
            target, _pp_info = self.post_processor.process(
                proj, prev, recent_returns=recent_returns,
                drawdown=drawdown, disagreement=self.disagreement,
            )
        else:
            delta0 = proj - prev
            delta0[np.abs(delta0) < self.min_trade_delta] = 0.0
            target = prev + delta0

        if self._htf_idx is not None:
            target = self.apply_htf_gate(target)

        delta = target - prev
        turnover = float(np.abs(delta).sum())
        cost = self._exec_model.cost_fraction(delta)

        r_vec = self.fs.close[self.t + 1] / self.fs.close[self.t] - 1.0
        funding = float(np.sum(target * self.fs.funding_rate[self.t + 1]))
        gross_pnl = float(np.dot(target, r_vec))
        net = gross_pnl - cost - funding

        p_base = self.portfolio_value
        self.portfolio_value *= (1.0 + net)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.max_dd = max(self.max_dd, 1.0 - self.portfolio_value / max(self.peak_value, 1e-9))
        self._funding_pnl += funding

        self.weights = target
        self.turnover_total += turnover
        if turnover > 0:
            self.n_trades += 1
        else:
            self._n_hold_steps += 1
        self._returns_history.append(net)
        self._long_symbol_steps += int((target > 1e-9).sum())
        self._short_symbol_steps += int((target < -1e-9).sum())
        self._total_symbol_steps += self.n_symbols

        if self.use_dsr:
            reward = self._dsr_reward(net)
        else:
            reward = net * self.reward_scale - self.lambda_turnover * turnover

        self.t += 1
        terminated = bool(self.portfolio_value <= 1e-6 * self.initial_capital)
        truncated = bool(
            (self.t - self.start_idx) >= self.episode_bars
            or self.t >= self.fs.n_bars - 2
        )

        if delta.sum() > 1e-9:
            side = "buy"
        elif delta.sum() < -1e-9:
            side = "sell"
        else:
            side = "hold"
        event = "liquidation" if terminated else "normal"

        info = {
            "turnover": turnover,
            "execution": {
                "step": self.t, "p_base": p_base, "p_exec": self.portfolio_value,
                "action": raw.tolist(), "side": side,
                "inventory_after": self.weights.tolist(),
                "reward": float(reward), "event": event,
            },
        }

        if terminated or truncated:
            rets = np.array(self._returns_history) if self._returns_history else np.zeros(1)
            n_steps = len(self._returns_history)
            sharpe = float(rets.mean() / rets.std() * np.sqrt(BARS_PER_YEAR_1H)) \
                if rets.std() > 0 else 0.0
            apy = float((self.portfolio_value / self.initial_capital) **
                        (BARS_PER_YEAR_1H / max(n_steps, 1)) - 1.0) if n_steps > 0 else 0.0
            info.update({
                "win_rate": float((rets > 0).mean()),
                "max_drawdown": self.max_dd,
                "portfolio_value": self.portfolio_value,
                "apy": apy,
                "sharpe": sharpe,
                "n_trades": self.n_trades,
                "turnover_total": self.turnover_total,
                "funding_pnl": self._funding_pnl,
                "long_pct": float(self._long_symbol_steps / max(self._total_symbol_steps, 1)),
                "short_pct": float(self._short_symbol_steps / max(self._total_symbol_steps, 1)),
                "hold_pct": float(self._n_hold_steps / max(n_steps, 1)),
            })

        obs = self._obs()
        return obs, float(reward), terminated, truncated, info
