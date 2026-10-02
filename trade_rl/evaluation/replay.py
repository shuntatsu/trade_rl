"""Lean per-symbol strategy replay on the canonical execution ledger."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.data.market_order_rules import MarketOrderProfile
from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.risk import PreTradeRisk
from trade_rl.risk.pretrade import should_rebind_strategy_proposal
from trade_rl.simulation import (
    BookState,
    EconomicTerminationReason,
    ExecutionCostConfig,
    ExecutionResult,
    MarketExecutor,
)
from trade_rl.simulation.diagnostics.funding import FundingBoundaryEvidence
from trade_rl.simulation.liquidity import SymbolCapacityEvidence
from trade_rl.simulation.orders.model import OrderEvent
from trade_rl.simulation.stateful.execution import StatefulExecutionObservation
from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.position_duration import (
    constrain_intent_for_minimum_hold,
    next_position_age_bars,
)
from trade_rl.strategies.position_intent import (
    PositionIntent,
    target_weight_for_intent,
)
from trade_rl.strategies.rl.intent import PPO_OBSERVATION_SCHEMA_V3


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    index: int
    intent: PositionIntent
    changed_intent: bool
    target_weight: float
    effective_intent: PositionIntent = PositionIntent.FLAT
    minimum_hold_suppressed: bool = False
    minimum_hold_unlocked: bool = False
    position_age_bars: int = 0
    position_age_bars_after: int = 0
    position_quantity_before: float = 0.0
    position_quantity_after: float = 0.0
    risk_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SingleSymbolReplayResult:
    book: BookState
    returns: ReturnSeries
    diagnostics: ExecutionDiagnostics
    decisions: tuple[ReplayDecision, ...]
    active_order_remainders: tuple[tuple[str, float], ...] = ()
    terminal_order_reasons: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class SharedCashReplayDecision:
    """One simultaneous multi-symbol decision before shared execution."""

    index: int
    intents: tuple[PositionIntent, ...]
    changed_intents: tuple[bool, ...]
    proposal_weights: tuple[float, ...]
    target_weights: tuple[float, ...]
    risk_reasons: tuple[str, ...]
    effective_intents: tuple[PositionIntent, ...] = ()
    minimum_hold_suppressed: tuple[bool, ...] = ()
    minimum_hold_unlocked: tuple[bool, ...] = ()
    position_age_bars_before: tuple[int, ...] = ()
    position_age_bars_after: tuple[int, ...] = ()
    position_quantity_before: tuple[float, ...] = ()
    position_quantity_after: tuple[float, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "changed_intents": self.changed_intents,
            "effective_intents": tuple(
                intent.value for intent in self.effective_intents
            ),
            "index": self.index,
            "intents": tuple(intent.value for intent in self.intents),
            "minimum_hold_suppressed": self.minimum_hold_suppressed,
            "minimum_hold_unlocked": self.minimum_hold_unlocked,
            "position_age_bars_after": self.position_age_bars_after,
            "position_age_bars_before": self.position_age_bars_before,
            "position_quantity_after": self.position_quantity_after,
            "position_quantity_before": self.position_quantity_before,
            "proposal_weights": self.proposal_weights,
            "risk_reasons": self.risk_reasons,
            "target_weights": self.target_weights,
        }


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
    capacity_events: tuple[SymbolCapacityEvidence, ...]
    funding_events: tuple[FundingBoundaryEvidence, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "borrow_cost_after": self.borrow_cost_after,
            "borrow_cost_before": self.borrow_cost_before,
            "cash_after": self.cash_after,
            "cash_before": self.cash_before,
            "capacity_events": tuple(asdict(event) for event in self.capacity_events),
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
    decisions: tuple[SharedCashReplayDecision, ...] = ()
    schema_version: str = "shared_cash_replay_ledger_v1"

    def to_mapping(self) -> dict[str, object]:
        payload: dict[str, object] = {
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
        if self.schema_version == "shared_cash_replay_ledger_v2":
            payload["decisions"] = tuple(
                decision.to_mapping() for decision in self.decisions
            )
        return payload


@dataclass(frozen=True, slots=True)
class SharedCashReplayResult:
    """One shared-account multi-symbol replay result."""

    book: BookState
    returns: ReturnSeries
    diagnostics: ExecutionDiagnostics
    decisions: tuple[SharedCashReplayDecision, ...]
    ledger_evidence: SharedCashReplayLedgerEvidence | None = None


class _ExecutedEntryPrices:
    """Track actual average fill prices for currently open signed positions."""

    def __init__(self, n_symbols: int) -> None:
        self._quantities = np.zeros(n_symbols, dtype=np.float64)
        self._average_prices = np.zeros(n_symbols, dtype=np.float64)

    def ingest(
        self,
        events: Sequence[OrderEvent],
        quantities: np.ndarray,
        *,
        terminated: bool = False,
    ) -> None:
        for event in events:
            filled = float(event.filled_quantity)
            if filled == 0.0:
                continue
            price = event.execution_price
            if price is None:
                raise RuntimeError("filled order event is missing its execution price")
            symbol_index = event.symbol_index
            if not 0 <= symbol_index < self._quantities.size:
                raise RuntimeError("fill event references an unknown symbol")
            previous = float(self._quantities[symbol_index])
            following = previous + filled
            previous_average = float(self._average_prices[symbol_index])
            if following == 0.0:
                self._average_prices[symbol_index] = 0.0
                following = 0.0
            elif previous == 0.0 or (previous > 0.0) != (following > 0.0):
                self._average_prices[symbol_index] = float(price)
            elif (previous > 0.0) == (filled > 0.0):
                self._average_prices[symbol_index] = (
                    abs(previous) * previous_average + abs(filled) * float(price)
                ) / abs(following)
            self._quantities[symbol_index] = following

        actual = np.asarray(quantities, dtype=np.float64)
        if actual.shape != self._quantities.shape:
            raise RuntimeError("execution fill events diverged from book quantities")
        if not np.allclose(actual, self._quantities, rtol=1e-9, atol=1e-12):
            if terminated:
                self._quantities = actual.copy()
                self._average_prices.fill(0.0)
                return
            raise RuntimeError("execution fill events diverged from book quantities")
        self._quantities = actual.copy()
        self._average_prices[actual == 0.0] = 0.0

    def apply_split(self, split_factor: np.ndarray) -> None:
        factors = np.asarray(split_factor, dtype=np.float64)
        if factors.shape != self._quantities.shape or not np.isfinite(factors).all():
            raise RuntimeError("split factors do not match the fill tracker roster")
        if np.any(factors <= 0.0):
            raise RuntimeError("split factors must be positive")
        self._quantities *= factors
        self._average_prices /= factors

    def mark_gross_return(
        self,
        symbol_index: int,
        *,
        quantity: float,
        mark_price: float,
    ) -> float | None:
        tracked_quantity = float(self._quantities[symbol_index])
        if quantity == 0.0:
            if tracked_quantity != 0.0:
                raise RuntimeError("flat book quantity diverged from fill tracker")
            return None
        if not math.isclose(quantity, tracked_quantity, rel_tol=1e-9, abs_tol=0.0):
            raise RuntimeError("book quantity diverged from execution fill tracker")
        entry_price = float(self._average_prices[symbol_index])
        if entry_price <= 0.0:
            raise RuntimeError("open position is missing its executed entry price")
        direction = 1.0 if quantity > 0.0 else -1.0
        return direction * (mark_price - entry_price) / entry_price


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


def _protective_exit_pending(strategy: SingleSymbolStrategy) -> bool:
    pending = getattr(strategy, "protective_exit_pending", False)
    if not isinstance(pending, bool):
        raise TypeError("protective_exit_pending must be boolean when provided")
    return pending


def _observation(
    dataset: MarketDataset,
    *,
    index: int,
    symbol_index: int,
    book: BookState,
    current_intent: PositionIntent,
    position_age_bars: int = 0,
    gross_position_return: float | None = None,
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
        current_position_quantity=float(book.quantities[symbol_index]),
        position_age_bars=position_age_bars,
        gross_position_return=gross_position_return,
    )


def _agent_stop_index(
    *,
    start_index: int,
    stop_index: int,
    execution_cost: ExecutionCostConfig,
    settle_terminal_position: bool,
) -> int:
    if not settle_terminal_position:
        return stop_index
    agent_stop_index = stop_index - execution_cost.order_latency_bars - 1
    if agent_stop_index <= start_index:
        raise ValueError(
            "terminal settlement requires at least one agent interval before the close"
        )
    return agent_stop_index


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
    minimum_hold_bars: int | None = None,
    settle_terminal_position: bool = False,
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
    if not isinstance(settle_terminal_position, bool):
        raise ValueError("settle_terminal_position must be boolean")
    resolved_minimum_hold_bars = (
        getattr(strategy, "minimum_hold_bars", 0)
        if minimum_hold_bars is None
        else minimum_hold_bars
    )
    if (
        isinstance(resolved_minimum_hold_bars, bool)
        or not isinstance(resolved_minimum_hold_bars, int)
        or resolved_minimum_hold_bars < 0
    ):
        raise ValueError("minimum_hold_bars must be a non-negative integer")
    strategy_observation_schema = getattr(strategy, "observation_schema", None)
    if (
        resolved_minimum_hold_bars > 0
        and strategy_observation_schema is not None
        and strategy_observation_schema != PPO_OBSERVATION_SCHEMA_V3
    ):
        raise ValueError("PPO minimum hold requires the age-aware observation")
    target_weight_for_intent(PositionIntent.LONG, gross_budget=gross_budget)

    initial_prices = dataset.resolved_array("mark_price")[start_index]
    book = BookState.zero(
        dataset.n_symbols,
        initial_capital,
        initial_prices,
        contract_multipliers=dataset.contract_multipliers,
    )
    executed_entry_prices = _ExecutedEntryPrices(dataset.n_symbols)
    latest_execution_observation: StatefulExecutionObservation | None = None

    def observe_execution(observation: StatefulExecutionObservation) -> None:
        nonlocal latest_execution_observation
        latest_execution_observation = observation

    resolved_execution_cost = execution_cost or ExecutionCostConfig.zero()
    agent_stop_index = _agent_stop_index(
        start_index=start_index,
        stop_index=stop_index,
        execution_cost=resolved_execution_cost,
        settle_terminal_position=settle_terminal_position,
    )
    executor = MarketExecutor(
        dataset,
        resolved_execution_cost,
        execution_observer=observe_execution,
    )
    risk_controller = risk or PreTradeRisk.default_for_execution(
        max_leverage=executor.cost.max_leverage
    )
    _validate_risk_execution_compatibility(risk_controller, executor)
    current_intent = PositionIntent.FLAT
    position_age_bars = 0
    minimum_hold_locked = False
    desired_quantity = 0.0
    decisions: list[ReplayDecision] = []
    returns: list[float] = []
    index = start_index

    while index < agent_stop_index:
        quantity_before = float(book.quantities[symbol_index])
        position_age_before = position_age_bars
        observation = _observation(
            dataset,
            index=index,
            symbol_index=symbol_index,
            book=book,
            current_intent=current_intent,
            position_age_bars=position_age_bars,
            gross_position_return=executed_entry_prices.mark_gross_return(
                symbol_index,
                quantity=quantity_before,
                mark_price=float(book.mark_prices[symbol_index]),
            ),
        )
        requested_intent = strategy.decide(observation)
        if not isinstance(requested_intent, PositionIntent):
            raise TypeError("strategy.decide must return PositionIntent")
        hold_decision = constrain_intent_for_minimum_hold(
            requested_intent,
            current_quantity=quantity_before,
            position_age_bars=position_age_bars,
            minimum_hold_bars=resolved_minimum_hold_bars,
            allow_protective_exit=_protective_exit_pending(strategy),
        )
        minimum_hold_unlocked = minimum_hold_locked and not hold_decision.suppressed
        intent = hold_decision.effective_intent
        changed_intent = intent is not current_intent
        if hold_decision.target_quantity_override is not None:
            desired_quantity = hold_decision.target_quantity_override
        elif changed_intent or minimum_hold_unlocked:
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
        if should_rebind_strategy_proposal(constrained):
            desired_quantity = _desired_quantity_from_weight(
                book,
                target_weight,
                symbol_index=symbol_index,
            )
        execution = executor.execute_interval(
            book,
            constrained.weights,
            start_index=index,
            bars=1,
        )
        if execution.next_index <= index:
            raise RuntimeError("execution did not advance replay index")
        if latest_execution_observation is None:
            raise RuntimeError("execution observer did not emit interval fills")
        executed_entry_prices.apply_split(
            dataset.resolved_array("split_factor")[execution.next_index]
        )
        executed_entry_prices.ingest(
            latest_execution_observation.order_events,
            execution.book.quantities,
            terminated=execution.termination_reason is not None,
        )
        book = execution.book
        position_age_bars = next_position_age_bars(
            position_age_bars,
            previous_quantity=quantity_before,
            filled_quantity=float(book.quantities[symbol_index]),
        )
        decisions.append(
            ReplayDecision(
                index=index,
                intent=requested_intent,
                changed_intent=changed_intent,
                target_weight=target_weight,
                effective_intent=intent,
                minimum_hold_suppressed=hold_decision.suppressed,
                minimum_hold_unlocked=minimum_hold_unlocked,
                position_age_bars=position_age_before,
                position_age_bars_after=position_age_bars,
                position_quantity_before=quantity_before,
                position_quantity_after=float(book.quantities[symbol_index]),
                risk_reasons=tuple(constrained.reasons),
            )
        )
        returns.append(execution.interval_net_return)
        current_intent = intent
        minimum_hold_locked = (
            hold_decision.suppressed and float(book.quantities[symbol_index]) != 0.0
        )
        index = execution.next_index
        if book.termination_reason is not None:
            break

    if (
        settle_terminal_position
        and book.termination_reason is None
        and index >= agent_stop_index
    ):
        current_intent = PositionIntent.FLAT
        desired_quantity = 0.0
        while index < stop_index and book.termination_reason is None:
            quantity_before = float(book.quantities[symbol_index])
            flat_target = np.zeros(dataset.n_symbols, dtype=np.float64)
            constrained = risk_controller.constrain(
                flat_target,
                current=book.weights,
                drawdown=book.max_drawdown,
            )
            execution = executor.execute_interval(
                book,
                constrained.weights,
                start_index=index,
                bars=1,
            )
            if execution.next_index <= index:
                raise RuntimeError("terminal settlement did not advance replay")
            if latest_execution_observation is None:
                raise RuntimeError("execution observer did not emit interval fills")
            executed_entry_prices.apply_split(
                dataset.resolved_array("split_factor")[execution.next_index]
            )
            executed_entry_prices.ingest(
                latest_execution_observation.order_events,
                execution.book.quantities,
                terminated=execution.termination_reason is not None,
            )
            book = execution.book
            position_age_bars = next_position_age_bars(
                position_age_bars,
                previous_quantity=quantity_before,
                filled_quantity=float(book.quantities[symbol_index]),
            )
            returns.append(execution.interval_net_return)
            index = execution.next_index

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
        active_order_remainders=(
            ()
            if latest_execution_observation is None
            else latest_execution_observation.active_order_remainders
        ),
        terminal_order_reasons=(
            ()
            if latest_execution_observation is None
            else latest_execution_observation.terminal_order_reasons
        ),
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
    minimum_hold_bars: int | Sequence[int] | None = None,
    settle_terminal_position: bool = False,
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
    if not isinstance(settle_terminal_position, bool):
        raise ValueError("settle_terminal_position must be boolean")
    if minimum_hold_bars is None:
        hold_bars_by_symbol = tuple(
            getattr(strategy, "minimum_hold_bars", 0) for strategy in strategy_tuple
        )
    elif isinstance(minimum_hold_bars, bool):
        raise ValueError("minimum_hold_bars must be a non-negative integer")
    elif isinstance(minimum_hold_bars, int):
        hold_bars_by_symbol = (minimum_hold_bars,) * dataset.n_symbols
    elif isinstance(minimum_hold_bars, Sequence):
        if isinstance(minimum_hold_bars, (str, bytes)):
            raise ValueError("minimum_hold_bars must contain non-negative integers")
        hold_bars_by_symbol = tuple(minimum_hold_bars)
        if len(hold_bars_by_symbol) != dataset.n_symbols:
            raise ValueError("minimum_hold_bars must match the dataset symbol roster")
    else:
        raise ValueError("minimum_hold_bars must be an integer or one value per symbol")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in hold_bars_by_symbol
    ):
        raise ValueError("minimum_hold_bars must contain non-negative integers")
    for strategy, hold_bars in zip(strategy_tuple, hold_bars_by_symbol, strict=True):
        observation_schema = getattr(strategy, "observation_schema", None)
        if (
            hold_bars > 0
            and observation_schema is not None
            and observation_schema != PPO_OBSERVATION_SCHEMA_V3
        ):
            raise ValueError("PPO minimum hold requires the age-aware observation")
    target_weight_for_intent(PositionIntent.LONG, gross_budget=gross_budget)

    resolved_execution_cost = execution_cost or ExecutionCostConfig.zero()
    agent_stop_index = _agent_stop_index(
        start_index=start_index,
        stop_index=stop_index,
        execution_cost=resolved_execution_cost,
        settle_terminal_position=settle_terminal_position,
    )
    initial_prices = dataset.resolved_array("mark_price")[start_index]
    book = BookState.zero(
        dataset.n_symbols,
        initial_capital,
        initial_prices,
        contract_multipliers=dataset.contract_multipliers,
    )
    executed_entry_prices = _ExecutedEntryPrices(dataset.n_symbols)
    execution_observation: StatefulExecutionObservation | None = None
    execution_observation_count = 0

    def retain_latest_execution_observation(
        observation: StatefulExecutionObservation,
    ) -> None:
        nonlocal execution_observation, execution_observation_count
        execution_observation = observation
        execution_observation_count += 1

    executor = MarketExecutor(
        dataset,
        resolved_execution_cost,
        market_order_profile=market_order_profile,
        execution_observer=retain_latest_execution_observation,
    )
    risk_controller = risk or PreTradeRisk.default_for_execution(
        max_leverage=executor.cost.max_leverage
    )
    _validate_risk_execution_compatibility(risk_controller, executor)
    current_intents = [PositionIntent.FLAT for _ in range(dataset.n_symbols)]
    position_age_bars = [0 for _ in range(dataset.n_symbols)]
    minimum_hold_locked = [False for _ in range(dataset.n_symbols)]
    desired_quantities = np.zeros(dataset.n_symbols, dtype=np.float64)
    decisions: list[SharedCashReplayDecision] = []
    returns: list[float] = []
    ledger_intervals: list[SharedCashLedgerIntervalEvidence] = []
    active_order_remainders: tuple[tuple[str, float], ...] = ()
    terminal_order_reasons: tuple[tuple[str, str], ...] = ()
    index = start_index

    def execute_and_record(
        target_weights: np.ndarray,
        *,
        interval_index: int,
    ) -> ExecutionResult:
        nonlocal active_order_remainders, book, terminal_order_reasons
        observations_before = execution_observation_count
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
            target_weights,
            start_index=interval_index,
            bars=1,
        )
        if execution.next_index <= interval_index:
            raise RuntimeError("execution did not advance replay index")
        if (
            execution_observation_count != observations_before + 1
            or execution_observation is None
        ):
            raise RuntimeError("execution observer did not emit exactly one interval")
        stateful_evidence = execution_observation
        if stateful_evidence.next_index != execution.next_index:
            raise RuntimeError("execution observer index differs from replay result")
        executed_entry_prices.apply_split(
            dataset.resolved_array("split_factor")[execution.next_index]
        )
        executed_entry_prices.ingest(
            stateful_evidence.order_events,
            execution.book.quantities,
            terminated=execution.termination_reason is not None,
        )
        if capture_ledger_evidence:
            if execution_observation_count != len(ledger_intervals) + 1:
                raise RuntimeError(
                    "execution observer did not emit exactly one interval"
                )
            ledger_intervals.append(
                SharedCashLedgerIntervalEvidence(
                    start_index=interval_index,
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
                    order_events=stateful_evidence.order_events,
                    capacity_events=stateful_evidence.capacity_evidence,
                    funding_events=stateful_evidence.funding_evidence,
                )
            )
            active_order_remainders = stateful_evidence.active_order_remainders
            terminal_order_reasons = stateful_evidence.terminal_order_reasons
        book = execution.book
        returns.append(execution.interval_net_return)
        return execution

    while index < agent_stop_index:
        intents: list[PositionIntent] = []
        effective_intents: list[PositionIntent] = []
        changed_intents: list[bool] = []
        suppressed: list[bool] = []
        unlocked: list[bool] = []
        ages_before = tuple(position_age_bars)
        quantities_before = tuple(float(value) for value in book.quantities)
        for symbol_index, strategy in enumerate(strategy_tuple):
            observation = _observation(
                dataset,
                index=index,
                symbol_index=symbol_index,
                book=book,
                current_intent=current_intents[symbol_index],
                position_age_bars=position_age_bars[symbol_index],
                gross_position_return=executed_entry_prices.mark_gross_return(
                    symbol_index,
                    quantity=quantities_before[symbol_index],
                    mark_price=float(book.mark_prices[symbol_index]),
                ),
            )
            requested_intent = strategy.decide(observation)
            if not isinstance(requested_intent, PositionIntent):
                raise TypeError("strategy.decide must return PositionIntent")
            hold_decision = constrain_intent_for_minimum_hold(
                requested_intent,
                current_quantity=quantities_before[symbol_index],
                position_age_bars=position_age_bars[symbol_index],
                minimum_hold_bars=hold_bars_by_symbol[symbol_index],
                allow_protective_exit=_protective_exit_pending(strategy),
            )
            effective_intent = hold_decision.effective_intent
            changed_intent = effective_intent is not current_intents[symbol_index]
            minimum_hold_unlocked = (
                minimum_hold_locked[symbol_index] and not hold_decision.suppressed
            )
            if hold_decision.target_quantity_override is not None:
                desired_quantities[symbol_index] = (
                    hold_decision.target_quantity_override
                )
            elif changed_intent or minimum_hold_unlocked:
                proposal_weight = target_weight_for_intent(
                    effective_intent,
                    gross_budget=gross_budget,
                )
                desired_quantities[symbol_index] = _desired_quantity_from_weight(
                    book,
                    proposal_weight,
                    symbol_index=symbol_index,
                )
            intents.append(requested_intent)
            effective_intents.append(effective_intent)
            changed_intents.append(changed_intent)
            suppressed.append(hold_decision.suppressed)
            unlocked.append(minimum_hold_unlocked)

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
        if should_rebind_strategy_proposal(constrained):
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

        decision_index = index
        execution = execute_and_record(
            constrained.weights,
            interval_index=decision_index,
        )
        for symbol_index in range(dataset.n_symbols):
            position_age_bars[symbol_index] = next_position_age_bars(
                position_age_bars[symbol_index],
                previous_quantity=quantities_before[symbol_index],
                filled_quantity=float(book.quantities[symbol_index]),
            )
            minimum_hold_locked[symbol_index] = (
                suppressed[symbol_index] and float(book.quantities[symbol_index]) != 0.0
            )
        decisions.append(
            SharedCashReplayDecision(
                index=decision_index,
                intents=tuple(intents),
                changed_intents=tuple(changed_intents),
                proposal_weights=tuple(float(value) for value in proposal_weights),
                target_weights=tuple(float(value) for value in constrained.weights),
                risk_reasons=constrained.reasons,
                effective_intents=tuple(effective_intents),
                minimum_hold_suppressed=tuple(suppressed),
                minimum_hold_unlocked=tuple(unlocked),
                position_age_bars_before=ages_before,
                position_age_bars_after=tuple(position_age_bars),
                position_quantity_before=quantities_before,
                position_quantity_after=tuple(
                    float(value) for value in book.quantities
                ),
            )
        )
        current_intents = effective_intents
        index = execution.next_index
        if book.termination_reason is not None:
            break

    if (
        settle_terminal_position
        and book.termination_reason is None
        and index >= agent_stop_index
    ):
        flat_proposal = np.zeros(dataset.n_symbols, dtype=np.float64)
        while index < stop_index and book.termination_reason is None:
            quantities_before = tuple(float(value) for value in book.quantities)
            constrained = risk_controller.constrain(
                flat_proposal,
                current=book.weights,
                drawdown=book.max_drawdown,
            )
            execution = execute_and_record(
                constrained.weights,
                interval_index=index,
            )
            for symbol_index in range(dataset.n_symbols):
                position_age_bars[symbol_index] = next_position_age_bars(
                    position_age_bars[symbol_index],
                    previous_quantity=quantities_before[symbol_index],
                    filled_quantity=float(book.quantities[symbol_index]),
                )
            index = execution.next_index

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
        if not ledger_intervals:
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
            active_order_remainders=active_order_remainders,
            terminal_order_reasons=terminal_order_reasons,
            decisions=tuple(decisions),
            schema_version=(
                "shared_cash_replay_ledger_v2"
                if settle_terminal_position or any(hold_bars_by_symbol)
                else "shared_cash_replay_ledger_v1"
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
