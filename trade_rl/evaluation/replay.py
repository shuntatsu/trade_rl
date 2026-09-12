"""Lean per-symbol strategy replay on the canonical execution ledger."""

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


def _desired_quantity_from_weight(
    book: BookState,
    target_weight: float,
    *,
    symbol_index: int,
) -> float:
    multipliers = np.asarray(book.contract_multipliers, dtype=np.float64)
    denominator = float(book.mark_prices[symbol_index] * multipliers[symbol_index])
    return float(target_weight * book.portfolio_value / denominator)


def _weight_for_desired_quantity(
    book: BookState,
    desired_quantity: float,
    *,
    symbol_index: int,
) -> float:
    if book.portfolio_value <= 0.0:
        return 0.0
    multipliers = np.asarray(book.contract_multipliers, dtype=np.float64)
    return float(
        desired_quantity
        * book.mark_prices[symbol_index]
        * multipliers[symbol_index]
        / book.portfolio_value
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
    symbol_index: int,
    book: BookState,
    current_intent: PositionIntent,
) -> StrategyObservation:
    return StrategyObservation(
        index=index,
        timestamp=dataset.timestamps[index],
        symbol=dataset.symbols[symbol_index],
        features=dataset.features[index, symbol_index],
        feature_available=dataset.feature_available[index, symbol_index],
        feature_staleness=dataset.resolved_array("feature_staleness")[
            index, symbol_index
        ],
        global_features=dataset.global_features[index],
        global_feature_available=dataset.resolved_array("global_feature_available")[
            index
        ],
        current_intent=current_intent,
        current_weight=float(book.weights[symbol_index]),
    )


def run_single_symbol_replay(
    dataset: MarketDataset,
    strategy: SingleSymbolStrategy,
    *,
    symbol_index: int = 0,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    initial_capital: float = 100_000.0,
    execution_cost: ExecutionCostConfig | None = None,
    risk: PreTradeRisk | None = None,
) -> SingleSymbolReplayResult:
    """Replay one selected symbol while every other symbol remains flat.

    ``stop_index`` is exclusive. The full source dataset and canonical executor
    remain intact, but observation, desired quantity, and target exposure are
    restricted to ``symbol_index``. This allows the same frozen strategy to be
    evaluated independently on every symbol without inventing sliced datasets.
    """

    if (
        isinstance(symbol_index, bool)
        or not isinstance(symbol_index, int)
        or not 0 <= symbol_index < dataset.n_symbols
    ):
        raise ValueError("symbol_index must identify an existing dataset symbol")
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
        dataset.n_symbols,
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
            symbol_index=symbol_index,
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
            desired_quantity = _desired_quantity_from_weight(
                book,
                proposal_weight,
                symbol_index=symbol_index,
            )
        proposal_weight = _weight_for_desired_quantity(
            book,
            desired_quantity,
            symbol_index=symbol_index,
        )
        proposal_weights = np.zeros(dataset.n_symbols, dtype=np.float64)
        proposal_weights[symbol_index] = proposal_weight
        constrained = risk_controller.constrain(
            proposal_weights,
            current=book.weights,
            drawdown=book.max_drawdown,
        )
        target_weight = float(constrained.weights[symbol_index])
        if constrained.was_constrained and any(
            reason != "max_turnover" for reason in constrained.reasons
        ):
            desired_quantity = _desired_quantity_from_weight(
                book,
                target_weight,
                symbol_index=symbol_index,
            )
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
            constrained.weights,
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
