"""Gymnasium environment for baseline-anchored residual portfolio control."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.metrics import PerformanceMetrics, evaluate_performance
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.rl.actions import (
    BaselineResidualComposer,
    ResidualAction,
    ResidualComposition,
)
from trade_rl.rl.observations import build_observation, observation_layout
from trade_rl.rl.rewards import relative_interval_reward
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.strategies.trend import TrendStrategy, TrendTargets


class AlphaProvider(Protocol):
    def predict_at(self, dataset: MarketDataset, index: int) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ResidualMarketEnvConfig:
    episode_bars: int = 200
    decision_every: int = 4
    reward_scale: float = 100.0
    initial_capital: float = 1.0
    minimum_equity_fraction: float = 1e-6
    execution_cost: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)

    def __post_init__(self) -> None:
        if self.episode_bars <= 0:
            raise ValueError("episode_bars must be positive")
        if self.decision_every <= 0:
            raise ValueError("decision_every must be positive")
        for field_name, value in (
            ("reward_scale", self.reward_scale),
            ("initial_capital", self.initial_capital),
            ("minimum_equity_fraction", self.minimum_equity_fraction),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")


class ResidualMarketEnv(gym.Env):
    """Two-action environment rewarded against an independent baseline book."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        trend_strategy: TrendStrategy | None = None,
        alpha_provider: AlphaProvider | Callable[[MarketDataset, int], np.ndarray]
        | None = None,
        alpha_enabled: bool = False,
        composer: BaselineResidualComposer | None = None,
        config: ResidualMarketEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.trend_strategy = trend_strategy or TrendStrategy()
        self.alpha_provider = alpha_provider
        self.alpha_enabled = bool(alpha_enabled)
        self.composer = composer or BaselineResidualComposer()
        self.config = config or ResidualMarketEnvConfig()
        self.executor = MarketExecutor(dataset, self.config.execution_cost)

        layout = observation_layout(dataset)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(layout.size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.start_index = self.trend_strategy.minimum_history
        self.end_index = self.start_index
        self.current_index = self.start_index
        self.hybrid = BookState.zero(dataset.n_symbols, self.config.initial_capital)
        self.shadow = BookState.zero(dataset.n_symbols, self.config.initial_capital)
        self._decision_step_index = 0

    @property
    def dataset_id(self) -> str:
        return self.dataset.dataset_id

    def _alpha_at(self, index: int) -> np.ndarray:
        if not self.alpha_enabled or self.alpha_provider is None:
            return np.zeros(self.dataset.n_symbols, dtype=np.float64)
        provider = self.alpha_provider
        if hasattr(provider, "predict_at"):
            value = provider.predict_at(self.dataset, index)
        else:
            value = provider(self.dataset, index)
        alpha = np.asarray(value, dtype=np.float64).reshape(-1)
        if alpha.shape != (self.dataset.n_symbols,) or not np.isfinite(alpha).all():
            raise ValueError("alpha provider returned an invalid vector")
        gross = float(np.abs(alpha).sum())
        return alpha / gross if gross > 1.0 else alpha

    def _market_inputs(self) -> tuple[TrendTargets, np.ndarray]:
        return (
            self.trend_strategy.targets(self.dataset, self.current_index),
            self._alpha_at(self.current_index),
        )

    def _observation(self) -> np.ndarray:
        trends, alpha = self._market_inputs()
        return build_observation(
            dataset=self.dataset,
            index=self.current_index,
            trends=trends,
            alpha=alpha,
            book=self.hybrid,
            start_index=self.start_index,
            end_index=self.end_index,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        resolved_options = options or {}
        minimum_start = self.trend_strategy.minimum_history
        maximum_start = self.dataset.n_bars - 1 - self.config.episode_bars
        if maximum_start < minimum_start:
            raise ValueError("dataset is too short for the configured episode")
        if "start_idx" in resolved_options:
            raw_start = resolved_options["start_idx"]
            if isinstance(raw_start, bool) or not isinstance(raw_start, int):
                raise ValueError("start_idx must be an integer")
            start = raw_start
        else:
            start = int(self.np_random.integers(minimum_start, maximum_start + 1))
        if not minimum_start <= start <= maximum_start:
            raise ValueError("start_idx is outside the executable range")

        self.start_index = start
        self.current_index = start
        self.end_index = start + self.config.episode_bars
        self.hybrid = BookState.zero(
            self.dataset.n_symbols,
            self.config.initial_capital,
        )
        self.shadow = BookState.zero(
            self.dataset.n_symbols,
            self.config.initial_capital,
        )
        self._decision_step_index = 0
        return self._observation(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self.current_index >= self.end_index:
            raise RuntimeError("step called after the episode ended")
        trends, alpha = self._market_inputs()
        composition = self.composer.compose(
            ResidualAction.from_array(action),
            trends,
            alpha,
            alpha_enabled=self.alpha_enabled,
        )
        bars = min(self.config.decision_every, self.end_index - self.current_index)
        hybrid_execution = self.executor.execute_interval(
            self.hybrid,
            composition.proposal,
            start_index=self.current_index,
            bars=bars,
        )
        shadow_execution = self.executor.execute_interval(
            self.shadow,
            trends.base,
            start_index=self.current_index,
            bars=bars,
        )
        if hybrid_execution.bars_advanced != shadow_execution.bars_advanced:
            raise RuntimeError("hybrid and shadow books advanced different bar counts")

        self.hybrid = hybrid_execution.book
        self.shadow = shadow_execution.book
        self.current_index = hybrid_execution.next_index
        self._decision_step_index += 1
        threshold = self.config.initial_capital * self.config.minimum_equity_fraction
        terminated = (
            self.hybrid.portfolio_value <= threshold
            or self.shadow.portfolio_value <= threshold
        )
        truncated = self.current_index >= self.end_index
        excess_log_return = (
            hybrid_execution.interval_log_return
            - shadow_execution.interval_log_return
        )
        reward = relative_interval_reward(
            hybrid_log_return=hybrid_execution.interval_log_return,
            shadow_log_return=shadow_execution.interval_log_return,
            scale=self.config.reward_scale,
            terminated=terminated,
        )
        info: dict[str, object] = {
            "bars_advanced": hybrid_execution.bars_advanced,
            "decision_step_index": self._decision_step_index,
            "interval_gross_return": hybrid_execution.interval_gross_return,
            "interval_cost": hybrid_execution.interval_cost,
            "interval_funding": hybrid_execution.interval_funding,
            "interval_net_return": hybrid_execution.interval_net_return,
            "shadow_interval_net_return": shadow_execution.interval_net_return,
            "excess_log_return": excess_log_return,
            "composition": composition,
        }
        if terminated or truncated:
            info.update(self._terminal_info())
        return self._observation(), reward, terminated, truncated, info

    def _book_metrics(self, book: BookState) -> PerformanceMetrics:
        return evaluate_performance(
            ReturnSeries(
                values=tuple(book.returns_history),
                kind=ReturnKind.BASE_BAR,
                periods_per_year=self.dataset.periods_per_year,
            ),
            turnover_total=book.turnover_total,
            total_cost=book.total_cost,
            funding_pnl=book.funding_pnl,
            n_trades=book.n_trades,
        )

    def _terminal_info(self) -> dict[str, object]:
        hybrid_metrics = self._book_metrics(self.hybrid)
        shadow_metrics = self._book_metrics(self.shadow)
        return {
            "hybrid_metrics": hybrid_metrics,
            "shadow_metrics": shadow_metrics,
            "excess_total_return": (
                hybrid_metrics.total_return - shadow_metrics.total_return
            ),
        }
