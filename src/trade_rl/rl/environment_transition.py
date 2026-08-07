"""Terminal accounting and transition resolution for the RL environment."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.environment_execution import EnvironmentExecutionCoordinator
from trade_rl.rl.rewards import RewardTracker
from trade_rl.rl.transition import EconomicTransition, classify_economic_transition
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionResult


@dataclass(frozen=True, slots=True)
class EnvironmentTransitionOutcome:
    """Books, returns and Gymnasium flags after terminal accounting."""

    hybrid: BookState
    shadow: BookState
    hybrid_log_return: float
    shadow_log_return: float
    hybrid_liquidation: ExecutionResult | None
    shadow_liquidation: ExecutionResult | None
    liquidation_complete: bool
    liquidation_terminal: bool
    emergency_deleverage: bool
    hybrid_terminated: bool
    shadow_terminated: bool
    economic_transition: EconomicTransition


class EnvironmentTerminationCoordinator:
    """Resolve drawdown, minimum-equity and episode-end terminal accounting."""

    def __init__(
        self,
        *,
        config: ResidualMarketEnvConfig,
        reward_tracker: RewardTracker,
        pre_trade_risk: PreTradeRisk,
        hybrid_executor: MarketExecutor,
        shadow_executor: MarketExecutor,
        execution_coordinator: EnvironmentExecutionCoordinator,
    ) -> None:
        self.config = config
        self.reward_tracker = reward_tracker
        self.pre_trade_risk = pre_trade_risk
        self.hybrid_executor = hybrid_executor
        self.shadow_executor = shadow_executor
        self.execution_coordinator = execution_coordinator

    @staticmethod
    def _drawdown(book: BookState) -> float:
        value = max(book.portfolio_value, 0.0)
        return min(1.0, max(0.0, 1.0 - value / max(book.peak_value, value, 1e-12)))

    def resolve(
        self,
        *,
        hybrid: BookState,
        shadow: BookState,
        current_index: int,
        time_limit_reached: bool,
        hybrid_log_return: float,
        shadow_log_return: float,
    ) -> EnvironmentTransitionOutcome:
        hybrid_liquidation: ExecutionResult | None = None
        shadow_liquidation: ExecutionResult | None = None
        liquidation_complete = True
        emergency_deleverage = False
        drawdown_after_execution = self._drawdown(hybrid)
        drawdown_stop = min(
            self.reward_tracker.config.drawdown_stop,
            self.pre_trade_risk.config.drawdown_stop,
        )
        if not hybrid.insolvent and drawdown_after_execution + 1e-12 >= drawdown_stop:
            emergency_deleverage = True
            hybrid_liquidation = self.hybrid_executor.liquidate_at_close(
                hybrid,
                index=current_index,
            )
            liquidation_complete = self.execution_coordinator.liquidation_complete(
                hybrid_liquidation
            )
            if (
                not liquidation_complete
                and self.config.fail_on_incomplete_emergency_liquidation
            ):
                raise RuntimeError(
                    "hybrid liquidation could not fully exit at drawdown stop"
                )
            hybrid = self.execution_coordinator.merge_liquidation_return(
                hybrid_liquidation
            )
            hybrid_log_return += hybrid_liquidation.interval_log_return
            hybrid.terminate(EconomicTerminationReason.DRAWDOWN_STOP)

        liquidation_terminal = (
            time_limit_reached
            and self.config.liquidate_on_end
            and not emergency_deleverage
        )
        if liquidation_terminal:
            hybrid_liquidation = self.hybrid_executor.liquidate_at_close(
                hybrid,
                index=current_index,
            )
            shadow_liquidation = self.shadow_executor.liquidate_at_close(
                shadow,
                index=current_index,
            )
            liquidation_complete = self.execution_coordinator.liquidation_complete(
                hybrid_liquidation
            ) and self.execution_coordinator.liquidation_complete(shadow_liquidation)
            hybrid = self.execution_coordinator.merge_liquidation_return(
                hybrid_liquidation
            )
            shadow = self.execution_coordinator.merge_liquidation_return(
                shadow_liquidation
            )
            hybrid_log_return += hybrid_liquidation.interval_log_return
            shadow_log_return += shadow_liquidation.interval_log_return

        threshold = self.config.initial_capital * self.config.minimum_equity_fraction
        if hybrid.portfolio_value <= threshold and not hybrid.insolvent:
            hybrid.terminate(EconomicTerminationReason.MINIMUM_EQUITY)
        if shadow.portfolio_value <= threshold and not shadow.insolvent:
            shadow.terminate(EconomicTerminationReason.MINIMUM_EQUITY)

        hybrid_terminated = hybrid.insolvent
        shadow_terminated = shadow.insolvent
        economic_transition = classify_economic_transition(
            hybrid=hybrid,
            shadow=shadow,
            time_limit_reached=time_limit_reached,
            liquidation_terminal=liquidation_terminal,
            liquidation_complete=liquidation_complete,
        )
        return EnvironmentTransitionOutcome(
            hybrid=hybrid,
            shadow=shadow,
            hybrid_log_return=hybrid_log_return,
            shadow_log_return=shadow_log_return,
            hybrid_liquidation=hybrid_liquidation,
            shadow_liquidation=shadow_liquidation,
            liquidation_complete=liquidation_complete,
            liquidation_terminal=liquidation_terminal,
            emergency_deleverage=emergency_deleverage,
            hybrid_terminated=hybrid_terminated,
            shadow_terminated=shadow_terminated,
            economic_transition=economic_transition,
        )
