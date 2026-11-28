"""Incremental state owner for numeric shared-portfolio execution."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from numbers import Real
from typing import Literal

import numpy as np

from trade_rl.data.market import MarketDataset, _require_available_marks_at_row
from trade_rl.data.market_order_rules import MarketOrderProfile
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.diagnostics.fill_cash import NativeFillCashObservation
from trade_rl.simulation.execution import (
    ExecutionCostConfig,
    ExecutionResult,
    MarketExecutor,
)
from trade_rl.simulation.native_transition import (
    PendingContinuationStatus,
    _execute_native_pending_transition,
    _validate_native_target_request,
    _validate_pending_continuation_config,
    execute_native_target_transition,
    validate_risk_execution_compatibility,
)
from trade_rl.simulation.orders.model import PendingOrder
from trade_rl.simulation.stateful.execution import (
    StatefulExecutionObservation,
    _require_native_execution_rows,
)


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric value")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def resolve_agent_stop_index(
    *,
    start_index: int,
    stop_index: int,
    execution_cost: ExecutionCostConfig,
    settle_terminal_position: bool,
    terminal_reserve_bars: int | None = None,
) -> int:
    """Resolve the actor boundary without changing the default latency reserve."""
    if terminal_reserve_bars is not None and (
        type(terminal_reserve_bars) is not int or not settle_terminal_position
    ):
        raise ValueError(
            "explicit terminal reserve requires an integer and terminal settlement"
        )
    if not settle_terminal_position:
        return stop_index
    minimum = execution_cost.order_latency_bars + 1
    reserve = minimum if terminal_reserve_bars is None else terminal_reserve_bars
    if reserve < minimum:
        raise ValueError(
            "terminal reserve must cover order latency and at least one close interval"
        )
    agent_stop_index = stop_index - reserve
    if agent_stop_index <= start_index:
        raise ValueError(
            "terminal settlement requires at least one agent interval before the close"
        )
    return agent_stop_index


class SharedPortfolioSession:
    """Advance one shared account by one native bar per numeric target.

    The session retains the exact returned book and executor between calls. Its
    public book view is a detached clone. Callers build actor-facing decision
    snapshots through the strategy-owned observation builder; the session stays
    independent of evaluation and policy DTOs. Terminal intervals are reserved
    from actor decisions and can only be advanced by the internal
    emergency-flatten path. An explicit terminal reserve can extend, but
    never shorten, the latency-aware default.
    """

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        start_index: int,
        stop_index: int,
        initial_capital: float = 100_000.0,
        execution_cost: ExecutionCostConfig | None = None,
        risk: PreTradeRisk | None = None,
        market_order_profile: MarketOrderProfile | None = None,
        settle_terminal_position: bool = False,
        terminal_reserve_bars: int | None = None,
        insolvency_valuation: Literal["floor_zero", "retain_debt"] = "floor_zero",
        hold_unchanged_risk_target: bool = False,
        strict_spot_cash: bool = False,
        native_calendar_month: bool = False,
        native_fixed_linear_carry: bool = False,
        native_calendar_month_linear_carry: bool = False,
        execution_observer: Callable[[StatefulExecutionObservation], None]
        | None = None,
        execution_trace_observer: Callable[[StatefulExecutionObservation], None]
        | None = None,
        fill_cash_observer: Callable[[NativeFillCashObservation], None] | None = None,
    ) -> None:
        if not isinstance(dataset, MarketDataset):
            raise TypeError("dataset must be a MarketDataset")
        if (
            isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or isinstance(stop_index, bool)
            or not isinstance(stop_index, int)
            or not 0 <= start_index < stop_index < dataset.n_bars
        ):
            raise ValueError("session range must satisfy 0 <= start < stop < n_bars")
        start_index = int(start_index)
        stop_index = int(stop_index)
        initial_capital = _finite_real(initial_capital, name="initial_capital")
        if initial_capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
        if execution_cost is not None and not isinstance(
            execution_cost, ExecutionCostConfig
        ):
            raise TypeError("execution_cost must be an ExecutionCostConfig or None")
        if risk is not None and not isinstance(risk, PreTradeRisk):
            raise TypeError("risk must be a PreTradeRisk or None")
        if not isinstance(settle_terminal_position, bool):
            raise TypeError("settle_terminal_position must be boolean")
        if type(hold_unchanged_risk_target) is not bool:
            raise ValueError("hold_unchanged_risk_target must be boolean")
        if type(native_fixed_linear_carry) is not bool:
            raise ValueError("native_fixed_linear_carry must be boolean")
        if type(native_calendar_month_linear_carry) is not bool:
            raise ValueError("native_calendar_month_linear_carry must be boolean")
        if (
            native_fixed_linear_carry or native_calendar_month_linear_carry
        ) and not settle_terminal_position:
            raise ValueError("native linear carry requires terminal settlement")
        if execution_observer is not None and not callable(execution_observer):
            raise TypeError("execution_observer must be callable")
        if execution_trace_observer is not None and not callable(
            execution_trace_observer
        ):
            raise TypeError("execution_trace_observer must be callable")

        resolved_cost = (
            ExecutionCostConfig.zero() if execution_cost is None else execution_cost
        )
        agent_stop_index = resolve_agent_stop_index(
            start_index=start_index,
            stop_index=stop_index,
            execution_cost=resolved_cost,
            settle_terminal_position=settle_terminal_position,
            terminal_reserve_bars=terminal_reserve_bars,
        )

        _require_available_marks_at_row(
            dataset,
            row_index=start_index,
            selected_symbol_indices=tuple(range(dataset.n_symbols)),
        )
        initial_prices = dataset.resolved_array("mark_price")[start_index]
        book = BookState.zero(
            dataset.n_symbols,
            initial_capital,
            initial_prices,
            contract_multipliers=dataset.contract_multipliers,
        )
        executor = MarketExecutor(
            dataset,
            resolved_cost,
            market_order_profile=market_order_profile,
            insolvency_valuation=insolvency_valuation,
            strict_spot_cash=strict_spot_cash,
            native_calendar_month=native_calendar_month,
            native_fixed_linear_carry=native_fixed_linear_carry,
            native_calendar_month_linear_carry=native_calendar_month_linear_carry,
            execution_observer=execution_observer,
            execution_trace_observer=execution_trace_observer,
            fill_cash_observer=fill_cash_observer,
        )
        if executor._native_linear_carry:
            _require_native_execution_rows(
                dataset,
                start_index=start_index,
                stop_index=stop_index,
                native_linear_carry=True,
            )
        risk_controller = (
            PreTradeRisk.default_for_execution(max_leverage=executor.cost.max_leverage)
            if risk is None
            else risk
        )
        validate_risk_execution_compatibility(risk_controller, executor)

        self._dataset = dataset
        self._executor = executor
        self._risk = risk_controller
        self._book = book
        self._current_index = start_index
        self._stop_index = stop_index
        self._agent_stop_index = agent_stop_index
        self._settle_terminal_position = settle_terminal_position
        self._poisoned = False
        self._hold_unchanged_risk_target = hold_unchanged_risk_target

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def agent_stop_index(self) -> int:
        return self._agent_stop_index

    @property
    def book(self) -> BookState:
        """Return a detached current-account view."""
        return self._book.clone()

    @property
    def active_orders(self) -> tuple[PendingOrder, ...]:
        """Return the immutable active-order tuple for evaluation evidence."""
        return self._executor.compatibility_order_book.active_orders

    @property
    def risk_controller(self) -> PreTradeRisk:
        return self._risk

    @property
    def execution_cost(self) -> ExecutionCostConfig:
        return self._executor.cost

    @property
    def execution_policy_digest(self) -> str:
        return self._executor.execution_policy_digest

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def can_decide(self) -> bool:
        return (
            not self._poisoned
            and self._book.termination_reason is None
            and not self._book.insolvent
            and self._current_index < self._agent_stop_index
        )

    @property
    def can_settle_terminal(self) -> bool:
        return (
            not self._poisoned
            and self._settle_terminal_position
            and self._book.termination_reason is None
            and not self._book.insolvent
            and self._agent_stop_index <= self._current_index < self._stop_index
        )

    @property
    def finished(self) -> bool:
        return (
            self._poisoned
            or self._book.termination_reason is not None
            or self._book.insolvent
            or self._current_index >= self._stop_index
        )

    def advance(
        self,
        target_weights: object,
        *,
        hold_current_quantity_symbols: tuple[int, ...] = (),
    ) -> tuple[RiskConstrainedTarget, ExecutionResult, tuple[int, ...]]:
        """Validate and execute one actor-controlled target interval."""
        if not self.can_decide:
            raise RuntimeError("portfolio session is not accepting actor decisions")
        return self._execute(
            target_weights,
            hold_current_quantity_symbols=hold_current_quantity_symbols,
            emergency_flatten_mask=None,
        )

    def advance_without_new_intent(
        self,
    ) -> tuple[
        RiskConstrainedTarget,
        ExecutionResult,
        tuple[int, ...],
        PendingContinuationStatus,
    ]:
        """Advance without resubmitting an old target.

        Continue compatible outstanding MARKET orders only if current-mark
        exposure remains risk-admissible, including independent partial
        fills. Otherwise use the ordinary risk-constrained exact-HOLD path.
        No new actor decision, future price input or liquidity bypass.
        """
        if not self.can_decide:
            raise RuntimeError("portfolio session is not accepting actor decisions")
        _validate_pending_continuation_config(self._executor.cost)
        index = self._current_index
        for row in (index, index + 1):
            _require_available_marks_at_row(
                self._dataset,
                row_index=row,
                selected_symbol_indices=tuple(range(self._dataset.n_symbols)),
                held_quantities=self._book.quantities,
            )
        try:
            risk_target, execution, effective_holds, status = (
                _execute_native_pending_transition(
                    self._executor,
                    self._book,
                    self._risk,
                    start_index=index,
                    hold_unchanged_risk_target=self._hold_unchanged_risk_target,
                )
            )
        except BaseException:
            self._poisoned = True
            raise
        return (*self._finish_advance(risk_target, execution, effective_holds), status)

    def settle_next_interval(
        self,
    ) -> tuple[RiskConstrainedTarget, ExecutionResult, tuple[int, ...]]:
        """Advance one reserved terminal interval with risk-owned flattening."""
        if not self.can_settle_terminal:
            raise RuntimeError("portfolio session has no reserved terminal interval")
        flatten_mask = np.asarray(self._book.quantities != 0.0, dtype=np.bool_)
        return self._execute(
            np.zeros(self._dataset.n_symbols, dtype=np.float64),
            hold_current_quantity_symbols=(),
            emergency_flatten_mask=flatten_mask,
        )

    def _execute(
        self,
        target_weights: object,
        *,
        hold_current_quantity_symbols: tuple[int, ...],
        emergency_flatten_mask: np.ndarray | None,
    ) -> tuple[RiskConstrainedTarget, ExecutionResult, tuple[int, ...]]:
        index = self._current_index
        proposed_weights, hold_symbols, emergency_mask, _ = (
            _validate_native_target_request(
                self._executor,
                self._book,
                self._risk,
                target_weights,
                start_index=index,
                hold_current_quantity_symbols=hold_current_quantity_symbols,
                emergency_flatten_mask=emergency_flatten_mask,
                reduce_only_mask=None,
            )
        )
        _require_available_marks_at_row(
            self._dataset,
            row_index=index + 1,
            selected_symbol_indices=tuple(range(self._dataset.n_symbols)),
            held_quantities=self._book.quantities,
        )
        try:
            risk_target, execution, effective_holds = execute_native_target_transition(
                self._executor,
                self._book,
                self._risk,
                proposed_weights,
                start_index=index,
                hold_current_quantity_symbols=hold_symbols,
                emergency_flatten_mask=emergency_mask,
                hold_unchanged_risk_target=self._hold_unchanged_risk_target,
            )
        except BaseException:
            # The executor may have advanced its RNG/order book before an
            # observer or downstream invariant raises; never permit a retry.
            self._poisoned = True
            raise
        return self._finish_advance(risk_target, execution, effective_holds)

    def _finish_advance(
        self,
        risk_target: RiskConstrainedTarget,
        execution: ExecutionResult,
        effective_holds: tuple[int, ...],
    ) -> tuple[RiskConstrainedTarget, ExecutionResult, tuple[int, ...]]:
        self._book = execution.book
        self._current_index = execution.next_index
        detached_execution = replace(execution, book=execution.book.clone())
        return risk_target, detached_execution, effective_holds


__all__ = ["SharedPortfolioSession"]
