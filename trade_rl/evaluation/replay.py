"""Lean per-symbol strategy replay on the canonical execution ledger."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.data.market_order_rules import MarketOrderProfile
from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import (
    BookState,
    EconomicTerminationReason,
    ExecutionCostConfig,
    MarketExecutor,
)
from trade_rl.simulation.diagnostics.funding import FundingBoundaryEvidence
from trade_rl.simulation.orders.model import OrderEvent
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


@dataclass(frozen=True, slots=True)
class SharedCashReplayDecision:
    """One simultaneous multi-symbol decision before shared execution."""

    index: int
    intents: tuple[PositionIntent, ...]
    changed_intents: tuple[bool, ...]
    proposal_weights: tuple[float, ...]
    target_weights: tuple[float, ...]
    risk_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SharedCashLedgerIntervalEvidence:
    """Observer-only evidence for one completed shared-cash execution interval."""

    start_index: int
    next_index: int
    exact_quantities_before: tuple[str, ...]
    exact_quantities_after: tuple[str, ...]
    cash_before: float
    cash_after: float
    portfolio_value_before: float
    portfolio_value_after: float
    total_cost_before: float
    total_cost_after: float
    funding_pnl_before: float
    funding_pnl_after: float
    borrow_cost_before: float
    borrow_cost_after: float
    turnover_total_before: float
    turnover_total_after: float
    max_drawdown_before: float
    max_drawdown_after: float
    interval_cost: float
    interval_funding: float
    interval_borrow_cost: float
    interval_dividend: float
    interval_cash_interest: float
    interval_net_return: float
    termination_reason: str | None
    order_events: tuple[OrderEvent, ...]
    funding_events: tuple[FundingBoundaryEvidence, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "borrow_cost_after": self.borrow_cost_after,
            "borrow_cost_before": self.borrow_cost_before,
            "cash_after": self.cash_after,
            "cash_before": self.cash_before,
            "exact_quantities_after": self.exact_quantities_after,
            "exact_quantities_before": self.exact_quantities_before,
            "funding_events": tuple(
                event.to_mapping() for event in self.funding_events
            ),
            "funding_pnl_after": self.funding_pnl_after,
            "funding_pnl_before": self.funding_pnl_before,
            "interval_borrow_cost": self.interval_borrow_cost,
            "interval_cash_interest": self.interval_cash_interest,
            "interval_cost": self.interval_cost,
            "interval_dividend": self.interval_dividend,
            "interval_funding": self.interval_funding,
            "interval_net_return": self.interval_net_return,
            "max_drawdown_after": self.max_drawdown_after,
            "max_drawdown_before": self.max_drawdown_before,
            "next_index": self.next_index,
            "order_events": tuple(
                event.canonical_payload() for event in self.order_events
            ),
            "portfolio_value_after": self.portfolio_value_after,
            "portfolio_value_before": self.portfolio_value_before,
            "start_index": self.start_index,
            "termination_reason": self.termination_reason,
            "total_cost_after": self.total_cost_after,
            "total_cost_before": self.total_cost_before,
            "turnover_total_after": self.turnover_total_after,
            "turnover_total_before": self.turnover_total_before,
        }


@dataclass(frozen=True, slots=True)
class SharedCashReplayLedgerEvidence:
    """Canonical observer trace for one shared-cash replay."""

    dataset_id: str
    execution_policy_digest: str
    start_index: int
    stop_index: int
    intervals: tuple[SharedCashLedgerIntervalEvidence, ...]
    terminal_exact_quantities: tuple[str, ...]
    final_cash: float
    final_portfolio_value: float
    final_total_cost: float
    final_funding_pnl: float
    final_borrow_cost: float
    final_turnover_total: float
    final_max_drawdown: float
    termination_reason: str | None
    active_order_remainders: tuple[tuple[str, float], ...]
    terminal_order_reasons: tuple[tuple[str, str], ...]
    schema_version: str = "shared_cash_replay_ledger_v1"

    def to_mapping(self) -> dict[str, object]:
        return {
            "active_order_remainders": self.active_order_remainders,
            "dataset_id": self.dataset_id,
            "execution_policy_digest": self.execution_policy_digest,
            "final_borrow_cost": self.final_borrow_cost,
            "final_cash": self.final_cash,
            "final_funding_pnl": self.final_funding_pnl,
            "final_max_drawdown": self.final_max_drawdown,
            "final_portfolio_value": self.final_portfolio_value,
            "final_total_cost": self.final_total_cost,
            "final_turnover_total": self.final_turnover_total,
            "intervals": tuple(interval.to_mapping() for interval in self.intervals),
            "schema_version": self.schema_version,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "terminal_exact_quantities": self.terminal_exact_quantities,
            "terminal_order_reasons": self.terminal_order_reasons,
            "termination_reason": self.termination_reason,
        }


@dataclass(frozen=True, slots=True)
class SharedCashReplayResult:
    """One shared-account multi-symbol replay result."""

    book: BookState
    returns: ReturnSeries
    diagnostics: ExecutionDiagnostics
    decisions: tuple[SharedCashReplayDecision, ...]
    ledger_evidence: SharedCashReplayLedgerEvidence | None = None


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


def run_shared_cash_replay(
    dataset: MarketDataset,
    strategies: Sequence[SingleSymbolStrategy],
    *,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    initial_capital: float = 100_000.0,
    execution_cost: ExecutionCostConfig | None = None,
    risk: PreTradeRisk | None = None,
    market_order_profile: MarketOrderProfile | None = None,
    capture_ledger_evidence: bool = False,
) -> SharedCashReplayResult:
    """Replay all symbols against one shared cash, risk and execution book.

    Every strategy observes the same pre-execution book snapshot for a decision
    bar. The complete proposal vector is then constrained once and executed once,
    so no symbol can consume cash or risk budget before another symbol decides.
    ``strategies`` must contain one distinct instance per dataset symbol to avoid
    hidden mutable strategy state leaking across symbols.
    """

    strategy_tuple = tuple(strategies)
    if len(strategy_tuple) != dataset.n_symbols:
        raise ValueError("shared-cash replay requires one strategy per dataset symbol")
    if len({id(strategy) for strategy in strategy_tuple}) != len(strategy_tuple):
        raise ValueError(
            "shared-cash replay requires a distinct strategy instance per symbol"
        )
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
    executor = MarketExecutor(
        dataset,
        execution_cost or ExecutionCostConfig.zero(),
        market_order_profile=market_order_profile,
    )
    risk_controller = risk or _default_replay_risk(executor)
    _validate_risk_execution_compatibility(risk_controller, executor)
    current_intents = [PositionIntent.FLAT for _ in range(dataset.n_symbols)]
    desired_quantities = np.zeros(dataset.n_symbols, dtype=np.float64)
    decisions: list[SharedCashReplayDecision] = []
    returns: list[float] = []
    ledger_intervals: list[SharedCashLedgerIntervalEvidence] = []
    last_execution = None
    index = start_index

    while index < stop_index:
        intents: list[PositionIntent] = []
        changed_intents: list[bool] = []
        for symbol_index, strategy in enumerate(strategy_tuple):
            observation = _observation(
                dataset,
                index=index,
                symbol_index=symbol_index,
                book=book,
                current_intent=current_intents[symbol_index],
            )
            intent = strategy.decide(observation)
            if not isinstance(intent, PositionIntent):
                raise TypeError("strategy.decide must return PositionIntent")
            changed_intent = intent is not current_intents[symbol_index]
            if changed_intent:
                proposal_weight = target_weight_for_intent(
                    intent,
                    gross_budget=gross_budget,
                )
                desired_quantities[symbol_index] = _desired_quantity_from_weight(
                    book,
                    proposal_weight,
                    symbol_index=symbol_index,
                )
            intents.append(intent)
            changed_intents.append(changed_intent)

        proposal_weights = np.asarray(
            [
                _weight_for_desired_quantity(
                    book,
                    desired_quantities[symbol_index],
                    symbol_index=symbol_index,
                )
                for symbol_index in range(dataset.n_symbols)
            ],
            dtype=np.float64,
        )
        constrained = risk_controller.constrain(
            proposal_weights,
            current=book.weights,
            drawdown=book.max_drawdown,
        )
        if constrained.was_constrained and any(
            reason != "max_turnover" for reason in constrained.reasons
        ):
            desired_quantities = np.asarray(
                [
                    _desired_quantity_from_weight(
                        book,
                        float(constrained.weights[symbol_index]),
                        symbol_index=symbol_index,
                    )
                    for symbol_index in range(dataset.n_symbols)
                ],
                dtype=np.float64,
            )

        decisions.append(
            SharedCashReplayDecision(
                index=index,
                intents=tuple(intents),
                changed_intents=tuple(changed_intents),
                proposal_weights=tuple(float(value) for value in proposal_weights),
                target_weights=tuple(float(value) for value in constrained.weights),
                risk_reasons=constrained.reasons,
            )
        )
        exact_quantities_before = tuple(str(value) for value in book.exact_quantities)
        cash_before = float(book.cash)
        portfolio_value_before = float(book.portfolio_value)
        total_cost_before = float(book.total_cost)
        funding_pnl_before = float(book.funding_pnl)
        borrow_cost_before = float(book.borrow_cost)
        turnover_total_before = float(book.turnover_total)
        max_drawdown_before = float(book.max_drawdown)
        execution = executor.execute_interval(
            book,
            constrained.weights,
            start_index=index,
            bars=1,
        )
        if execution.next_index <= index:
            raise RuntimeError("execution did not advance replay index")
        if capture_ledger_evidence:
            ledger_intervals.append(
                SharedCashLedgerIntervalEvidence(
                    start_index=index,
                    next_index=execution.next_index,
                    exact_quantities_before=exact_quantities_before,
                    exact_quantities_after=tuple(
                        str(value) for value in execution.book.exact_quantities
                    ),
                    cash_before=cash_before,
                    cash_after=float(execution.book.cash),
                    portfolio_value_before=portfolio_value_before,
                    portfolio_value_after=float(execution.book.portfolio_value),
                    total_cost_before=total_cost_before,
                    total_cost_after=float(execution.book.total_cost),
                    funding_pnl_before=funding_pnl_before,
                    funding_pnl_after=float(execution.book.funding_pnl),
                    borrow_cost_before=borrow_cost_before,
                    borrow_cost_after=float(execution.book.borrow_cost),
                    turnover_total_before=turnover_total_before,
                    turnover_total_after=float(execution.book.turnover_total),
                    max_drawdown_before=max_drawdown_before,
                    max_drawdown_after=float(execution.book.max_drawdown),
                    interval_cost=float(execution.interval_cost),
                    interval_funding=float(execution.interval_funding),
                    interval_borrow_cost=float(execution.interval_borrow_cost),
                    interval_dividend=float(execution.interval_dividend),
                    interval_cash_interest=float(execution.interval_cash_interest),
                    interval_net_return=float(execution.interval_net_return),
                    termination_reason=execution.termination_reason,
                    order_events=execution.order_events,
                    funding_events=execution.funding_evidence,
                )
            )
        last_execution = execution
        book = execution.book
        returns.append(execution.interval_net_return)
        current_intents = intents
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
    ledger_evidence = None
    if capture_ledger_evidence:
        if last_execution is None:
            raise RuntimeError("captured replay produced no execution interval")
        terminal_reason = (
            None
            if book.termination_reason is None
            else (
                book.termination_reason.value
                if isinstance(book.termination_reason, EconomicTerminationReason)
                else str(book.termination_reason)
            )
        )
        ledger_evidence = SharedCashReplayLedgerEvidence(
            dataset_id=dataset.dataset_id,
            execution_policy_digest=executor.execution_policy_digest,
            start_index=start_index,
            stop_index=stop_index,
            intervals=tuple(ledger_intervals),
            terminal_exact_quantities=tuple(
                str(value) for value in book.exact_quantities
            ),
            final_cash=float(book.cash),
            final_portfolio_value=float(book.portfolio_value),
            final_total_cost=float(book.total_cost),
            final_funding_pnl=float(book.funding_pnl),
            final_borrow_cost=float(book.borrow_cost),
            final_turnover_total=float(book.turnover_total),
            final_max_drawdown=float(book.max_drawdown),
            termination_reason=terminal_reason,
            active_order_remainders=tuple(
                (order.order_id, float(order.remaining_quantity))
                for order in last_execution.order_book.active_orders
            ),
            terminal_order_reasons=tuple(
                (order.order_id, str(order.terminal_reason))
                for order in last_execution.order_book.terminal_orders
            ),
        )
    return SharedCashReplayResult(
        book=book.clone(),
        returns=ReturnSeries(
            values=tuple(returns),
            kind=ReturnKind.BASE_BAR,
            periods_per_year=dataset.periods_per_year,
        ),
        diagnostics=diagnostics,
        decisions=tuple(decisions),
        ledger_evidence=ledger_evidence,
    )


__all__ = [
    "ReplayDecision",
    "SharedCashLedgerIntervalEvidence",
    "SharedCashReplayDecision",
    "SharedCashReplayLedgerEvidence",
    "SharedCashReplayResult",
    "SingleSymbolReplayResult",
    "run_shared_cash_replay",
    "run_single_symbol_replay",
]
