"""Lean single-symbol strategy replay on the canonical execution ledger."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import (
    BookState,
    EconomicTerminationReason,
    ExecutionCostConfig,
    MarketExecutor,
)
from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.position_intent import (
    PositionIntent,
    target_weight_for_intent,
)


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    index: int
    intent: PositionIntent
    changed_intent: bool
    target_weight: float


@dataclass(frozen=True, slots=True)
class SingleSymbolReplayResult:
    book: BookState
    returns: ReturnSeries
    diagnostics: ExecutionDiagnostics
    decisions: tuple[ReplayDecision, ...]


def _desired_quantity_from_weight(book: BookState, target_weight: float) -> float:
    multipliers = np.asarray(book.contract_multipliers, dtype=np.float64)
    denominator = float(book.mark_prices[0] * multipliers[0])
    return float(target_weight * book.portfolio_value / denominator)


def _weight_for_desired_quantity(book: BookState, desired_quantity: float) -> float:
    if book.portfolio_value <= 0.0:
        return 0.0
    multipliers = np.asarray(book.contract_multipliers, dtype=np.float64)
    return float(
        desired_quantity * book.mark_prices[0] * multipliers[0] / book.portfolio_value
    )


def _default_replay_risk(executor: MarketExecutor) -> PreTradeRisk:
    hard_limit = min(1.0, float(executor.cost.max_leverage))
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=hard_limit,
            max_abs_weight=hard_limit,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )


def _validate_risk_execution_compatibility(
    risk: PreTradeRisk,
    executor: MarketExecutor,
) -> None:
    execution_limit = float(executor.cost.max_leverage)
    if (
        risk.config.max_gross > execution_limit
        or risk.config.max_abs_weight > execution_limit
    ):
        raise ValueError("risk exposure limits must not exceed execution max_leverage")


def _observation(
    dataset: MarketDataset,
    *,
    index: int,
    book: BookState,
    current_intent: PositionIntent,
) -> StrategyObservation:
    return StrategyObservation(
        index=index,
        timestamp=dataset.timestamps[index],
        symbol=dataset.symbols[0],
        features=dataset.features[index, 0],
        feature_available=dataset.feature_available[index, 0],
        global_features=dataset.global_features[index],
        global_feature_available=dataset.resolved_array("global_feature_available")[
            index
        ],
        current_intent=current_intent,
        current_weight=float(book.weights[0]),
    )


def run_single_symbol_replay(
    dataset: MarketDataset,
    strategy: SingleSymbolStrategy,
    *,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    initial_capital: float = 100_000.0,
    execution_cost: ExecutionCostConfig | None = None,
    risk: PreTradeRisk | None = None,
) -> SingleSymbolReplayResult:
    """Replay one symbol with quantity-preserving holds and hard risk limits.

    ``stop_index`` is exclusive. Every decision uses row ``t`` and executes over
    the following bar through the canonical ``MarketExecutor``. When no explicit
    risk controller is supplied, replay installs only an execution-aligned hard
    exposure guard; stricter drawdown or turnover constraints must be explicit.
    """

    if dataset.n_symbols != 1:
        raise ValueError("single-symbol replay requires exactly one symbol")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or isinstance(stop_index, bool)
        or not isinstance(stop_index, int)
        or not 0 <= start_index < stop_index < dataset.n_bars
    ):
        raise ValueError("replay range must satisfy 0 <= start < stop < n_bars")
    if not math.isfinite(initial_capital) or initial_capital <= 0.0:
        raise ValueError("initial_capital must be finite and positive")
    target_weight_for_intent(PositionIntent.LONG, gross_budget=gross_budget)

    initial_prices = dataset.resolved_array("mark_price")[start_index]
    book = BookState.zero(
        1,
        initial_capital,
        initial_prices,
        contract_multipliers=dataset.contract_multipliers,
    )
    executor = MarketExecutor(dataset, execution_cost or ExecutionCostConfig.zero())
    risk_controller = risk or _default_replay_risk(executor)
    _validate_risk_execution_compatibility(risk_controller, executor)
    current_intent = PositionIntent.FLAT
    desired_quantity = 0.0
    decisions: list[ReplayDecision] = []
    returns: list[float] = []
    index = start_index

    while index < stop_index:
        observation = _observation(
            dataset,
            index=index,
            book=book,
            current_intent=current_intent,
        )
        intent = strategy.decide(observation)
        if not isinstance(intent, PositionIntent):
            raise TypeError("strategy.decide must return PositionIntent")
        changed_intent = intent is not current_intent
        if changed_intent:
            proposal_weight = target_weight_for_intent(
                intent,
                gross_budget=gross_budget,
            )
            desired_quantity = _desired_quantity_from_weight(book, proposal_weight)
        proposal_weight = _weight_for_desired_quantity(book, desired_quantity)
        constrained = risk_controller.constrain(
            np.asarray([proposal_weight], dtype=np.float64),
            current=book.weights,
            drawdown=book.max_drawdown,
        )
        target_weight = float(constrained.weights[0])
        if constrained.was_constrained and any(
            reason != "max_turnover" for reason in constrained.reasons
        ):
            desired_quantity = _desired_quantity_from_weight(book, target_weight)
        decisions.append(
            ReplayDecision(
                index=index,
                intent=intent,
                changed_intent=changed_intent,
                target_weight=target_weight,
            )
        )
        execution = executor.execute_interval(
            book,
            np.asarray([target_weight], dtype=np.float64),
            start_index=index,
            bars=1,
        )
        if execution.next_index <= index:
            raise RuntimeError("execution did not advance replay index")
        book = execution.book
        returns.append(execution.interval_net_return)
        current_intent = intent
        index = execution.next_index
        if book.termination_reason is not None:
            break

    termination_reasons: tuple[str, ...] = ()
    reason = book.termination_reason
    if reason is not None:
        reason_value = (
            reason.value if isinstance(reason, EconomicTerminationReason) else reason
        )
        termination_reasons = (reason_value,)
    diagnostics = ExecutionDiagnostics(
        turnover_total=book.turnover_total,
        total_cost=book.total_cost,
        funding_pnl=book.funding_pnl,
        borrow_cost=book.borrow_cost,
        n_trades=book.n_trades,
        rebalance_events=book.rebalance_events,
        termination_reasons=termination_reasons,
    )
    return SingleSymbolReplayResult(
        book=book.clone(),
        returns=ReturnSeries(
            values=tuple(returns),
            kind=ReturnKind.BASE_BAR,
            periods_per_year=dataset.periods_per_year,
        ),
        diagnostics=diagnostics,
        decisions=tuple(decisions),
    )


__all__ = [
    "ReplayDecision",
    "SingleSymbolReplayResult",
    "run_single_symbol_replay",
]
