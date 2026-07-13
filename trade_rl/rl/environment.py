"""Gymnasium environment for baseline-anchored residual portfolio control."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.metrics import PerformanceMetrics, evaluate_performance
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
from trade_rl.rl.actions import ACTION_SCHEMA, BaselineResidualComposer, ResidualAction
from trade_rl.rl.observations import (
    OBSERVATION_SCHEMA,
    build_observation,
    observation_layout,
)
from trade_rl.rl.rewards import (
    AbsoluteGrowthRewardConfig,
    RewardContext,
    absolute_growth_reward,
    build_reward_context,
)
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import (
    ExecutionCostConfig,
    ExecutionResult,
    MarketExecutor,
)
from trade_rl.strategies.trend import TrendStrategy, TrendTargets

_LIQUIDATION_TOLERANCE = 1e-12


class AlphaProvider(Protocol):
    def predict_at(self, dataset: MarketDataset, index: int) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ResidualMarketEnvConfig:
    episode_hours: float = 200.0
    decision_hours: float = 4.0
    episode_bars: int | None = None
    decision_every: int | None = None
    initial_capital: float = math.nan
    minimum_equity_fraction: float = 1e-6
    liquidate_on_end: bool = False
    reward: AbsoluteGrowthRewardConfig = field(
        default_factory=AbsoluteGrowthRewardConfig
    )
    reward_scale: float | None = None
    execution_cost: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)

    def __post_init__(self) -> None:
        if math.isnan(self.initial_capital):
            raise ValueError(
                "initial_capital must be explicitly configured in quote-currency units"
            )
        for positive_field_name, positive_value in (
            ("episode_hours", self.episode_hours),
            ("decision_hours", self.decision_hours),
            ("initial_capital", self.initial_capital),
            ("minimum_equity_fraction", self.minimum_equity_fraction),
        ):
            if (
                isinstance(positive_value, bool)
                or not math.isfinite(positive_value)
                or positive_value <= 0.0
            ):
                raise ValueError(f"{positive_field_name} must be finite and positive")
        for optional_field_name, optional_value in (
            ("episode_bars", self.episode_bars),
            ("decision_every", self.decision_every),
        ):
            if optional_value is not None and (
                isinstance(optional_value, bool)
                or not isinstance(optional_value, int)
                or optional_value <= 0
            ):
                raise ValueError(f"{optional_field_name} must be a positive integer")
        if self.reward_scale is not None:
            if (
                isinstance(self.reward_scale, bool)
                or not math.isfinite(self.reward_scale)
                or self.reward_scale <= 0.0
            ):
                raise ValueError("reward_scale must be finite and positive")
            object.__setattr__(
                self,
                "reward",
                replace(self.reward, scale=float(self.reward_scale)),
            )

    def resolve_episode_bars(self, dataset: MarketDataset) -> int:
        return (
            self.episode_bars
            if self.episode_bars is not None
            else dataset.bars_for_hours(self.episode_hours)
        )

    def resolve_decision_bars(self, dataset: MarketDataset) -> int:
        return (
            self.decision_every
            if self.decision_every is not None
            else dataset.bars_for_hours(self.decision_hours)
        )

    def resolve_reward_window_bars(self, dataset: MarketDataset) -> int:
        return dataset.bars_for_hours(self.reward.baseline_window_hours)

    def resolve_reward_minimum_history_bars(self, dataset: MarketDataset) -> int:
        return dataset.bars_for_hours(self.reward.baseline_minimum_history_hours)


class ResidualMarketEnv(gym.Env):
    """Two-action environment with absolute-growth hierarchical rewards."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        trend_strategy: TrendStrategy | None = None,
        alpha_provider: AlphaProvider
        | Callable[[MarketDataset, int], np.ndarray]
        | None = None,
        alpha_enabled: bool = False,
        composer: BaselineResidualComposer | None = None,
        pre_trade_risk: PreTradeRisk | None = None,
        config: ResidualMarketEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.trend_strategy = trend_strategy or TrendStrategy()
        self.alpha_provider = alpha_provider
        self.alpha_enabled = bool(alpha_enabled)
        self.composer = composer or BaselineResidualComposer()
        self.pre_trade_risk = pre_trade_risk or PreTradeRisk()
        self.config = config or ResidualMarketEnvConfig()
        self._episode_bars = self.config.resolve_episode_bars(dataset)
        self._decision_bars = self.config.resolve_decision_bars(dataset)
        self._reward_window_bars = self.config.resolve_reward_window_bars(dataset)
        self._reward_minimum_history_bars = (
            self.config.resolve_reward_minimum_history_bars(dataset)
        )
        if self._decision_bars > self._episode_bars:
            raise ValueError("decision interval cannot exceed episode duration")
        self._environment_digest = content_digest(
            {
                "action_schema": ACTION_SCHEMA,
                "alpha_enabled": self.alpha_enabled,
                "dataset_id": dataset.dataset_id,
                "decision_bars": self._decision_bars,
                "environment_config": {
                    "decision_hours": self.config.decision_hours,
                    "episode_hours": self.config.episode_hours,
                    "execution_cost": asdict(self.config.execution_cost),
                    "initial_capital": self.config.initial_capital,
                    "liquidate_on_end": self.config.liquidate_on_end,
                    "minimum_equity_fraction": self.config.minimum_equity_fraction,
                    "reward": asdict(self.config.reward),
                    "reward_minimum_history_bars": self._reward_minimum_history_bars,
                    "reward_window_bars": self._reward_window_bars,
                },
                "episode_bars": self._episode_bars,
                "observation_schema": OBSERVATION_SCHEMA,
                "pre_trade_risk": asdict(self.pre_trade_risk.config),
                "schema_version": "residual_market_environment_v2",
                "trend": asdict(self.trend_strategy.config),
            }
        )
        self.hybrid_executor = MarketExecutor(dataset, self.config.execution_cost)
        self.shadow_executor = MarketExecutor(dataset, self.config.execution_cost)
        self.executor = self.hybrid_executor

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

        self.start_index = self.trend_strategy.minimum_history_for(dataset)
        self.end_index = self.start_index
        self.current_index = self.start_index
        initial_prices = dataset.close[self.start_index]
        self.hybrid = BookState.zero(
            dataset.n_symbols,
            self.config.initial_capital,
            initial_prices,
        )
        self.shadow = BookState.zero(
            dataset.n_symbols,
            self.config.initial_capital,
            initial_prices,
        )
        self._decision_step_index = 0
        self._emergency_deleverage = False

    @property
    def dataset_id(self) -> str:
        return self.dataset.dataset_id

    @property
    def initial_capital(self) -> float:
        return self.config.initial_capital

    @property
    def environment_digest(self) -> str:
        return self._environment_digest

    @property
    def episode_bars(self) -> int:
        return self._episode_bars

    @property
    def decision_bars(self) -> int:
        return self._decision_bars

    @staticmethod
    def _drawdown(book: BookState) -> float:
        return 1.0 - book.portfolio_value / max(book.peak_value, book.portfolio_value)

    def _reward_context(self) -> RewardContext:
        return build_reward_context(
            hybrid_returns=self.hybrid.returns_history,
            shadow_returns=self.shadow.returns_history,
            hybrid_drawdown=self._drawdown(self.hybrid),
            window_bars=self._reward_window_bars,
            minimum_history_bars=self._reward_minimum_history_bars,
            config=self.config.reward,
        )

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
            hybrid=self.hybrid,
            shadow=self.shadow,
            start_index=self.start_index,
            end_index=self.end_index,
            hybrid_risk_scale=self.pre_trade_risk.risk_scale(
                self._drawdown(self.hybrid)
            ),
            shadow_risk_scale=self.pre_trade_risk.risk_scale(
                self._drawdown(self.shadow)
            ),
            reward_context=self._reward_context(),
            emergency_deleverage=self._emergency_deleverage,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        resolved_options = options or {}
        minimum_start = self.trend_strategy.minimum_history_for(self.dataset)
        maximum_start = self.dataset.n_bars - 1 - self.episode_bars
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
        self.end_index = start + self.episode_bars
        initial_prices = self.dataset.close[start]
        self.hybrid = BookState.zero(
            self.dataset.n_symbols,
            self.config.initial_capital,
            initial_prices,
        )
        self.shadow = BookState.zero(
            self.dataset.n_symbols,
            self.config.initial_capital,
            initial_prices,
        )
        execution_seed = int(
            self.np_random.integers(0, np.iinfo(np.int32).max, endpoint=True)
        )
        self.hybrid_executor.reset_random_state(execution_seed)
        self.shadow_executor.reset_random_state(execution_seed)
        self._decision_step_index = 0
        self._emergency_deleverage = False
        return self._observation(), {}

    def _constrain_target(
        self,
        proposal: np.ndarray,
        book: BookState,
    ) -> RiskConstrainedTarget:
        target = np.asarray(proposal, dtype=np.float64).reshape(-1).copy()
        if target.shape != (self.dataset.n_symbols,) or not np.isfinite(target).all():
            raise ValueError("proposal does not match dataset symbols")
        return self.pre_trade_risk.constrain(
            target,
            current=book.weights,
            drawdown=self._drawdown(book),
        )

    @staticmethod
    def _merge_liquidation_return(
        liquidation: ExecutionResult,
    ) -> BookState:
        result = liquidation.book
        if abs(liquidation.interval_net_return) <= 1e-15:
            return result
        if result.returns_history:
            previous = result.returns_history[-1]
            result.returns_history[-1] = (1.0 + previous) * (
                1.0 + liquidation.interval_net_return
            ) - 1.0
        else:
            result.returns_history.append(liquidation.interval_net_return)
        result.peak_value = max(result.peak_value, result.portfolio_value)
        result.max_drawdown = max(
            result.max_drawdown,
            1.0 - result.portfolio_value / result.peak_value,
        )
        return result

    @staticmethod
    def _require_complete_liquidation(
        *,
        name: str,
        liquidation: ExecutionResult,
    ) -> None:
        if liquidation.unfilled_turnover > _LIQUIDATION_TOLERANCE or np.any(
            np.abs(liquidation.book.quantities) > _LIQUIDATION_TOLERANCE
        ):
            raise RuntimeError(f"{name} liquidation left residual positions")

    def _liquidate_pair(self) -> tuple[ExecutionResult, ExecutionResult]:
        hybrid_liquidation = self.hybrid_executor.liquidate_at_close(
            self.hybrid,
            index=self.current_index,
        )
        shadow_liquidation = self.shadow_executor.liquidate_at_close(
            self.shadow,
            index=self.current_index,
        )
        self._require_complete_liquidation(
            name="hybrid",
            liquidation=hybrid_liquidation,
        )
        self._require_complete_liquidation(
            name="shadow",
            liquidation=shadow_liquidation,
        )
        self.hybrid = self._merge_liquidation_return(hybrid_liquidation)
        self.shadow = self._merge_liquidation_return(shadow_liquidation)
        return hybrid_liquidation, shadow_liquidation

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self.current_index >= self.end_index:
            raise RuntimeError("step called after the episode ended")
        reward_context_before = self._reward_context()
        portfolio_value_before = self.hybrid.portfolio_value
        trends, alpha = self._market_inputs()
        composition = self.composer.compose(
            ResidualAction.from_array(action),
            trends,
            alpha,
            alpha_enabled=self.alpha_enabled,
        )
        hybrid_risk = self._constrain_target(composition.proposal, self.hybrid)
        shadow_risk = self._constrain_target(trends.base, self.shadow)
        bars = min(self.decision_bars, self.end_index - self.current_index)
        hybrid_execution = self.hybrid_executor.execute_interval(
            self.hybrid,
            hybrid_risk.weights,
            start_index=self.current_index,
            bars=bars,
        )
        shadow_execution = self.shadow_executor.execute_interval(
            self.shadow,
            shadow_risk.weights,
            start_index=self.current_index,
            bars=bars,
        )
        if hybrid_execution.bars_advanced != shadow_execution.bars_advanced:
            raise RuntimeError("hybrid and shadow books advanced different bar counts")

        self.hybrid = hybrid_execution.book
        self.shadow = shadow_execution.book
        self.current_index = hybrid_execution.next_index
        self._decision_step_index += 1
        time_limit_reached = self.current_index >= self.end_index
        hybrid_log_return = hybrid_execution.interval_log_return
        shadow_log_return = shadow_execution.interval_log_return
        hybrid_liquidation: ExecutionResult | None = None
        shadow_liquidation: ExecutionResult | None = None
        liquidation_terminal = time_limit_reached and self.config.liquidate_on_end
        drawdown_stop_terminal = False
        if liquidation_terminal:
            hybrid_liquidation, shadow_liquidation = self._liquidate_pair()
            hybrid_log_return += hybrid_liquidation.interval_log_return
            shadow_log_return += shadow_liquidation.interval_log_return
        elif self._drawdown(self.hybrid) >= self.config.reward.drawdown_stop:
            self._emergency_deleverage = True
            drawdown_stop_terminal = True
            hybrid_liquidation, shadow_liquidation = self._liquidate_pair()
            hybrid_log_return += hybrid_liquidation.interval_log_return
            shadow_log_return += shadow_liquidation.interval_log_return

        threshold = self.config.initial_capital * self.config.minimum_equity_fraction
        hybrid_terminated = self.hybrid.portfolio_value <= threshold
        shadow_terminated = self.shadow.portfolio_value <= threshold
        terminated = (
            hybrid_terminated
            or shadow_terminated
            or liquidation_terminal
            or drawdown_stop_terminal
        )
        truncated = time_limit_reached and not terminated
        if drawdown_stop_terminal:
            termination_reason: str | None = "drawdown_stop"
        elif hybrid_terminated or shadow_terminated:
            termination_reason = "minimum_equity"
        elif liquidation_terminal:
            termination_reason = "evaluation_liquidation"
        else:
            termination_reason = None

        reward_context_after = self._reward_context()
        reward_breakdown = absolute_growth_reward(
            hybrid_log_return=hybrid_log_return,
            before=reward_context_before,
            after=reward_context_after,
            config=self.config.reward,
        )
        excess_log_return = hybrid_log_return - shadow_log_return
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
            "hybrid_risk": hybrid_risk,
            "shadow_risk": shadow_risk,
            "hybrid_execution": hybrid_execution,
            "shadow_execution": shadow_execution,
            "hybrid_terminated": hybrid_terminated,
            "shadow_terminated": shadow_terminated,
            "liquidation_terminal": liquidation_terminal,
            "emergency_deleverage": self._emergency_deleverage,
            "termination_reason": termination_reason,
            "reward_context_before": reward_context_before,
            "reward_context_after": reward_context_after,
            "reward_growth_raw": reward_breakdown.growth_raw,
            "reward_baseline_penalty_delta": (
                reward_breakdown.baseline_penalty_delta
            ),
            "reward_baseline_penalty_weighted": (
                reward_breakdown.baseline_penalty_weighted
            ),
            "reward_drawdown_penalty_delta": (
                reward_breakdown.drawdown_penalty_delta
            ),
            "reward_drawdown_penalty_weighted": (
                reward_breakdown.drawdown_penalty_weighted
            ),
            "reward_total_raw": reward_breakdown.total_raw,
            "reward_total_scaled": reward_breakdown.total_scaled,
            "rolling_hybrid_log_growth": (
                reward_context_after.rolling_hybrid_log_growth
            ),
            "rolling_baseline_log_growth": (
                reward_context_after.rolling_shadow_log_growth
            ),
            "baseline_shortfall": reward_context_after.baseline_shortfall,
            "baseline_tolerance": reward_context_after.baseline_tolerance,
            "baseline_penalty": reward_context_after.baseline_penalty,
            "drawdown_before": reward_context_before.hybrid_drawdown,
            "drawdown_after": reward_context_after.hybrid_drawdown,
            "drawdown_severity_before": reward_context_before.drawdown_severity,
            "drawdown_severity_after": reward_context_after.drawdown_severity,
            "portfolio_value_before": portfolio_value_before,
            "portfolio_value_after": self.hybrid.portfolio_value,
            "peak_value": self.hybrid.peak_value,
        }
        if hybrid_liquidation is not None and shadow_liquidation is not None:
            info["hybrid_liquidation"] = hybrid_liquidation
            info["shadow_liquidation"] = shadow_liquidation
        if terminated or truncated:
            info.update(self._terminal_info())
        return (
            self._observation(),
            reward_breakdown.total_scaled,
            terminated,
            truncated,
            info,
        )

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
            n_trades=book.fill_count,
        )

    def _terminal_info(self) -> dict[str, object]:
        hybrid_metrics = self._book_metrics(self.hybrid)
        shadow_metrics = self._book_metrics(self.shadow)
        return {
            "hybrid_metrics": hybrid_metrics,
            "shadow_metrics": shadow_metrics,
            "hybrid_rebalance_events": self.hybrid.rebalance_events,
            "shadow_rebalance_events": self.shadow.rebalance_events,
            "excess_total_return": (
                hybrid_metrics.total_return - shadow_metrics.total_return
            ),
        }
