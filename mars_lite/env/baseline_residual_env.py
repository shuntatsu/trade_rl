from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mars_lite.env.market_execution_core import BookState, MarketExecutionCore
from mars_lite.env.observation import (
    ObservationSchema,
    ObservationState,
    build_observation,
)
from mars_lite.trading.baseline_residual import BaselineResidualComposer
from mars_lite.trading.execution import make_execution_model
from mars_lite.trading.pipeline import DecisionPipeline, MarketView, PortfolioState
from mars_lite.trading.post_processor import BARS_PER_YEAR_1H, PostProcessInfo
from mars_lite.trading.pre_trade_risk import PreTradeRiskVerifier
from mars_lite.trading.trend_family import TrendFamily


class BaselineResidualTradingEnv(gym.Env):
    """Two-action environment rewarded relative to an independent base-trend book."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        fs,
        *,
        trend_family: Optional[TrendFamily] = None,
        alpha_provider=None,
        alpha_enabled: bool = True,
        composer: Optional[BaselineResidualComposer] = None,
        episode_bars: int = 200,
        decision_every: int = 4,
        fee_rate: float = 0.0005,
        spread_rate: float = 0.0002,
        impact_rate: float = 0.0001,
        cost_multiplier: float = 1.0,
        initial_capital: float = 1.0,
        post_processor=None,
        min_trade_delta: float = 0.0,
        htf_gate: bool = False,
        htf_threshold: float = 0.3,
        htf_neutral_scale: float = 0.5,
        reward_scale: float = 100.0,
        pre_trade_verifier: Optional[PreTradeRiskVerifier] = None,
        obs_risk_state: bool = False,
    ):
        super().__init__()
        if decision_every <= 0:
            raise ValueError("decision_every must be positive")
        if episode_bars <= 0:
            raise ValueError("episode_bars must be positive")
        self.fs = fs
        self.n_symbols = fs.n_symbols
        self.trend_family = trend_family or TrendFamily()
        self.alpha_provider = alpha_provider
        self.alpha_enabled = bool(alpha_enabled)
        self.composer = composer or BaselineResidualComposer()
        self.episode_bars = int(episode_bars)
        self.decision_every = int(decision_every)
        self.initial_capital = float(initial_capital)
        self.reward_scale = float(reward_scale)
        self.pre_trade_verifier = pre_trade_verifier
        self.obs_risk_state = bool(obs_risk_state)
        self.observation_schema = ObservationSchema(
            include_risk_state=obs_risk_state, progress_mode="zero"
        )

        self._htf_idx: Optional[int] = None
        if htf_gate:
            try:
                self._htf_idx = fs.feature_names.index("4h_ret_z20")
            except ValueError as exc:
                raise ValueError("htf_gate requires feature 4h_ret_z20") from exc
        self._pipeline = DecisionPipeline(
            post_processor=post_processor,
            min_trade_delta=min_trade_delta,
            htf_threshold=htf_threshold,
            htf_neutral_scale=htf_neutral_scale,
        )
        self._vol_lookback = (
            post_processor.cfg.vol_lookback if post_processor is not None else 0
        )
        self.bars_per_year = int(
            post_processor.cfg.bars_per_year
            if post_processor is not None
            else BARS_PER_YEAR_1H
        )
        if self.bars_per_year <= 0:
            raise ValueError("post-processor bars_per_year must be positive")
        execution_model = make_execution_model(
            fee_rate=fee_rate,
            spread_rate=spread_rate,
            impact_rate=impact_rate,
            cost_multiplier=cost_multiplier,
        )
        self._execution = MarketExecutionCore(fs, execution_model)

        # Existing features + fast/base/slow/alpha + current hybrid weight.
        self.n_per_symbol = fs.n_features + 5
        self.n_global = fs.global_features.shape[1] + 3 + (4 if obs_risk_state else 0)
        obs_dim = self.n_symbols * self.n_per_symbol + self.n_global
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.t = 0
        self.start_idx = 0
        self.end_t = 0
        self.hybrid = BookState.zero(self.n_symbols, self.initial_capital)
        self.shadow = BookState.zero(self.n_symbols, self.initial_capital)
        self._last_pp_info = PostProcessInfo()
        self._decision_step_index = 0

    @property
    def obs_layout(self) -> dict[str, int]:
        return {
            "n_symbols": self.n_symbols,
            "n_per_symbol": self.n_per_symbol,
            "n_global": self.n_global,
        }

    def _alpha_at(self, t: int) -> np.ndarray:
        if not self.alpha_enabled or self.alpha_provider is None:
            return np.zeros(self.n_symbols, dtype=np.float64)
        if hasattr(self.alpha_provider, "predict_at"):
            value = self.alpha_provider.predict_at(self.fs, t)
        elif callable(self.alpha_provider):
            value = self.alpha_provider(self.fs, t)
        else:
            raise TypeError("alpha_provider must be callable or implement predict_at")
        alpha = np.asarray(value, dtype=np.float64).reshape(-1)
        if alpha.shape != (self.n_symbols,) or not np.all(np.isfinite(alpha)):
            raise ValueError("alpha provider returned invalid weights")
        gross = float(np.abs(alpha).sum())
        return alpha / gross if gross > 1.0 else alpha

    def _market_inputs(self):
        trends = self.trend_family.targets(self.fs, self.t)
        alpha = self._alpha_at(self.t)
        return trends, alpha

    def _obs(self) -> np.ndarray:
        trends, alpha = self._market_inputs()
        augmented = np.concatenate(
            [
                self.fs.features[self.t],
                trends.fast.reshape(-1, 1),
                trends.base.reshape(-1, 1),
                trends.slow.reshape(-1, 1),
                alpha.reshape(-1, 1),
            ],
            axis=1,
        )
        info = self._last_pp_info
        return build_observation(
            per_symbol_features=augmented,
            global_features=self.fs.global_features[self.t],
            state=ObservationState(
                weights=self.hybrid.weights,
                portfolio_value=self.hybrid.portfolio_value,
                peak_value=max(self.hybrid.peak_value, self.hybrid.portfolio_value),
                progress=(self.t - self.start_idx) / max(self.episode_bars, 1),
                vol_scale=info.vol_scale,
                dd_scale=info.dd_scale,
                disagreement_scale=info.disagreement_scale,
                est_port_vol=info.est_port_vol,
            ),
            schema=self.observation_schema,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "start_idx" in options:
            self.start_idx = int(options["start_idx"])
        else:
            max_start = max(0, self.fs.n_bars - self.episode_bars - 2)
            self.start_idx = int(self.np_random.integers(0, max_start + 1))
        if not 0 <= self.start_idx < max(1, self.fs.n_bars - 2):
            raise ValueError("start_idx is outside the executable range")
        self.end_t = min(self.start_idx + self.episode_bars, self.fs.n_bars - 2)
        self.t = self.start_idx
        self.hybrid = BookState.zero(self.n_symbols, self.initial_capital)
        self.shadow = BookState.zero(self.n_symbols, self.initial_capital)
        self._last_pp_info = PostProcessInfo()
        self._decision_step_index = 0
        return self._obs(), {}

    @staticmethod
    def _portfolio_state(book: BookState) -> PortfolioState:
        return PortfolioState(
            weights=book.weights,
            portfolio_value=book.portfolio_value,
            peak_value=max(book.peak_value, book.portfolio_value),
        )

    def _validate_target(self, target: np.ndarray, book: BookState) -> None:
        if self.pre_trade_verifier is not None:
            self.pre_trade_verifier.validate(
                target,
                book.portfolio_value,
                symbols=self.fs.symbols,
                current_weights=book.weights,
            )

    def step(self, action):
        trends, alpha = self._market_inputs()
        composition = self.composer.compose(
            np.asarray(action, dtype=np.float64),
            trends,
            alpha,
            alpha_enabled=self.alpha_enabled,
        )
        market = MarketView.from_feature_set(
            self.fs,
            self.t,
            vol_lookback=self._vol_lookback,
            htf_idx=self._htf_idx,
        )
        hybrid_target, hybrid_pp = self._pipeline.process_proposal(
            composition.proposal, self._portfolio_state(self.hybrid), market
        )
        shadow_target, shadow_pp = self._pipeline.process_proposal(
            trends.base, self._portfolio_state(self.shadow), market
        )
        self._validate_target(hybrid_target, self.hybrid)
        self._validate_target(shadow_target, self.shadow)

        bars = min(self.decision_every, self.end_t - self.t)
        hybrid_exec = self._execution.execute_interval(
            self.hybrid, hybrid_target, start_t=self.t, bars=bars
        )
        shadow_exec = self._execution.execute_interval(
            self.shadow, shadow_target, start_t=self.t, bars=bars
        )
        if hybrid_exec.bars_advanced != shadow_exec.bars_advanced:
            raise RuntimeError("hybrid and shadow advanced different bar counts")

        self.hybrid = hybrid_exec.book
        self.shadow = shadow_exec.book
        self.t = hybrid_exec.next_t
        self._last_pp_info = hybrid_pp
        self._decision_step_index += 1

        terminated = bool(
            self.hybrid.portfolio_value <= 1e-6 * self.initial_capital
            or self.shadow.portfolio_value <= 1e-6 * self.initial_capital
        )
        if terminated:
            reward = -abs(self.reward_scale)
        else:
            reward = self.reward_scale * (
                hybrid_exec.interval_log_return - shadow_exec.interval_log_return
            )
        if not np.isfinite(reward):
            raise ValueError("relative reward is non-finite")
        truncated = bool(self.t >= self.end_t or self.t >= self.fs.n_bars - 2)

        info = {
            "bars_advanced": hybrid_exec.bars_advanced,
            "decision_step_index": self._decision_step_index,
            "interval_gross_return": hybrid_exec.interval_gross_return,
            "interval_cost": hybrid_exec.interval_cost,
            "interval_funding": hybrid_exec.interval_funding,
            "interval_net_return": hybrid_exec.interval_net_return,
            "shadow_interval_net_return": shadow_exec.interval_net_return,
            "excess_log_return": (
                hybrid_exec.interval_log_return - shadow_exec.interval_log_return
            ),
            "composition": composition,
            "hybrid_pp_info": hybrid_pp,
            "shadow_pp_info": shadow_pp,
        }
        if terminated or truncated:
            info.update(self._terminal_info())
        return self._obs(), float(reward), terminated, truncated, info

    def _terminal_info(self) -> dict[str, object]:
        def metrics(book: BookState) -> dict[str, float | int]:
            returns = np.asarray(book.returns_history, dtype=np.float64)
            sharpe = (
                float(returns.mean() / returns.std() * np.sqrt(self.bars_per_year))
                if returns.size and returns.std() > 0.0
                else 0.0
            )
            return {
                "total_return": book.portfolio_value / self.initial_capital - 1.0,
                "sharpe": sharpe,
                "max_drawdown": book.max_drawdown,
                "turnover_total": book.turnover_total,
                "funding_pnl": book.funding_pnl,
                "total_cost": book.total_cost,
                "n_trades": book.n_trades,
            }

        hybrid = metrics(self.hybrid)
        shadow = metrics(self.shadow)
        return {
            "hybrid": hybrid,
            "shadow": shadow,
            "excess_total_return": float(hybrid["total_return"])
            - float(shadow["total_return"]),
        }
