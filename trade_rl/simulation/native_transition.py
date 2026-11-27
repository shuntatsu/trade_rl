"""Risk-constrained native target transitions over the canonical executor."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Literal

import numpy as np

from trade_rl._array_snapshot import checked_array_input
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import (
    ExecutionCostConfig,
    ExecutionResult,
    MarketExecutor,
)
from trade_rl.simulation.orders.model import OrderType, TimeInForce
from trade_rl.simulation.quantities import exact_quantity, parse_quantity


def _target_weights(value: object, *, n_symbols: int) -> np.ndarray:
    checked = checked_array_input(
        value, field_name="target weights", allow_object_arrays=False
    )
    try:
        raw = np.asarray(checked)
    except (TypeError, ValueError) as error:
        raise ValueError("target weights must be a numeric vector") from error
    if raw.ndim != 1 or raw.shape != (n_symbols,):
        raise ValueError("target weights must match the market symbol vector")
    if raw.dtype.kind not in "iuf":
        raise TypeError("target weights must contain real numeric values")
    weights = np.asarray(raw, dtype=np.float64).copy()
    if not np.isfinite(weights).all():
        raise ValueError("target weights must be finite")
    return weights


def _boolean_mask(
    value: object | None, *, field_name: str, n_symbols: int
) -> np.ndarray | None:
    if value is None:
        return None
    mask = np.asarray(
        checked_array_input(value, field_name=field_name, allow_object_arrays=False)
    )
    if mask.dtype != np.dtype(np.bool_):
        raise TypeError(f"{field_name} must contain booleans")
    if mask.shape != (n_symbols,):
        raise ValueError(f"{field_name} must match the market symbol vector")
    return mask.copy()


def _hold_symbols(
    value: tuple[int, ...], *, n_symbols: int, target: np.ndarray, current: np.ndarray
) -> tuple[int, ...]:
    if type(value) is not tuple or any(type(index) is not int for index in value):
        raise TypeError("hold symbols must be a tuple of native integer indices")
    if len(set(value)) != len(value) or any(
        index < 0 or index >= n_symbols for index in value
    ):
        raise ValueError("hold symbols must be unique valid market indices")
    if any(target[index] != current[index] for index in value):
        raise ValueError("hold target must equal the current book weight")
    return value


def _validate_native_target_request(
    executor: MarketExecutor,
    book: BookState,
    pretrade_risk: PreTradeRisk,
    target_weights: object,
    *,
    start_index: int,
    hold_current_quantity_symbols: tuple[int, ...],
    emergency_flatten_mask: object | None,
    reduce_only_mask: object | None,
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray | None, np.ndarray | None]:
    """Validate one transition request without mutating execution state."""
    if not isinstance(executor, MarketExecutor):
        raise TypeError("executor must be a MarketExecutor")
    if not isinstance(book, BookState):
        raise TypeError("book must be a BookState")
    if not isinstance(pretrade_risk, PreTradeRisk):
        raise TypeError("pretrade_risk must be a PreTradeRisk")
    if type(start_index) is not int:
        raise TypeError("start_index must be a native integer")
    if start_index < 0 or start_index + 1 >= executor.dataset.n_bars:
        raise ValueError("native target interval is outside the dataset")
    if book.termination_reason is not None or book.insolvent:
        raise RuntimeError("cannot execute a target from a terminal book")

    n_symbols = executor.dataset.n_symbols
    if book.quantities.shape != (n_symbols,):
        raise ValueError("book quantities do not match market symbols")
    current_weights = np.asarray(book.weights, dtype=np.float64)
    if current_weights.shape != (n_symbols,):
        raise ValueError("book weights do not match market symbols")
    proposed_weights = _target_weights(target_weights, n_symbols=n_symbols)
    hold_symbols = _hold_symbols(
        hold_current_quantity_symbols,
        n_symbols=n_symbols,
        target=proposed_weights,
        current=current_weights,
    )
    emergency_mask = _boolean_mask(
        emergency_flatten_mask,
        field_name="emergency_flatten_mask",
        n_symbols=n_symbols,
    )
    reduce_mask = _boolean_mask(
        reduce_only_mask,
        field_name="reduce_only_mask",
        n_symbols=n_symbols,
    )
    return proposed_weights, hold_symbols, emergency_mask, reduce_mask


def validate_risk_execution_compatibility(
    pretrade_risk: PreTradeRisk, executor: MarketExecutor
) -> None:
    """Reject risk exposure limits that exceed the executor's leverage bound."""
    if not isinstance(pretrade_risk, PreTradeRisk):
        raise TypeError("pretrade_risk must be a PreTradeRisk")
    if not isinstance(executor, MarketExecutor):
        raise TypeError("executor must be a MarketExecutor")
    execution_limit = float(executor.cost.max_leverage)
    if (
        pretrade_risk.config.max_gross > execution_limit
        or pretrade_risk.config.max_abs_weight > execution_limit
    ):
        raise ValueError("risk exposure limits must not exceed execution max_leverage")


def execute_native_target_transition(
    executor: MarketExecutor,
    book: BookState,
    pretrade_risk: PreTradeRisk,
    target_weights: object,
    *,
    start_index: int,
    hold_current_quantity_symbols: tuple[int, ...] = (),
    emergency_flatten_mask: object | None = None,
    reduce_only_mask: object | None = None,
    hold_unchanged_risk_target: bool = False,
) -> tuple[RiskConstrainedTarget, ExecutionResult, tuple[int, ...]]:
    """Apply hard risk and execute exactly one native interval.

    Strategy and replay callers own action mapping and policy state. This
    transition accepts only numeric portfolio targets, applies risk against the
    pre-execution book, and advances the canonical MarketExecutor once. Exact
    quantity holds survive only when risk leaves their proposed weights intact.
    The opt-in independent-account convention also holds the whole book when
    the final risk-approved vector exactly equals the existing weights.
    """
    if type(hold_unchanged_risk_target) is not bool:
        raise ValueError("hold_unchanged_risk_target must be boolean")
    proposed_weights, hold_symbols, emergency_mask, reduce_mask = (
        _validate_native_target_request(
            executor,
            book,
            pretrade_risk,
            target_weights,
            start_index=start_index,
            hold_current_quantity_symbols=hold_current_quantity_symbols,
            emergency_flatten_mask=emergency_flatten_mask,
            reduce_only_mask=reduce_only_mask,
        )
    )
    current_weights = np.asarray(book.weights, dtype=np.float64)

    risk_target = pretrade_risk.constrain(
        proposed_weights,
        current=current_weights,
        drawdown=book.max_drawdown,
        emergency_flatten_mask=emergency_mask,
        reduce_only_mask=reduce_mask,
    )
    effective_holds = tuple(
        index
        for index in hold_symbols
        if risk_target.weights[index] == proposed_weights[index]
    )
    if hold_unchanged_risk_target and np.array_equal(
        risk_target.weights, current_weights
    ):
        effective_holds = tuple(range(len(current_weights)))
    if effective_holds:
        execution = executor.execute_interval(
            book,
            risk_target.weights,
            start_index=start_index,
            bars=1,
            hold_current_quantity_symbols=effective_holds,
        )
    else:
        execution = executor.execute_interval(
            book,
            risk_target.weights,
            start_index=start_index,
            bars=1,
        )
    if execution.next_index != start_index + 1 or execution.bars_advanced != 1:
        raise RuntimeError("native target transition did not advance exactly one bar")
    return risk_target, execution, effective_holds


def _outward_weight(value: Fraction) -> float:
    # Never erase an exact residual or round its risk exposure toward zero.
    try:
        weight = float(value)
    except OverflowError as error:
        raise ValueError("pending exposure is not representable") from error
    if not math.isfinite(weight):
        raise ValueError("pending exposure is not representable")
    if abs(Fraction.from_float(weight)) < abs(value):
        weight = math.nextafter(weight, math.inf if value > 0 else -math.inf)
    if not math.isfinite(weight):
        raise ValueError("pending exposure is not representable")
    return weight


def _pending_endpoint_weights(
    executor: MarketExecutor, book: BookState, *, start_index: int
) -> np.ndarray:
    quantities = list(book.exact_quantities)
    seen: set[int] = set()
    for order in executor.compatibility_order_book.active_orders:
        intent = order.intent
        symbol = intent.symbol_index
        if (
            symbol in seen
            or not 0 <= symbol < len(quantities)
            or intent.dataset_id != executor.dataset.dataset_id
            or intent.execution_policy_digest != executor.execution_policy_digest
            or intent.order_type is not OrderType.MARKET
            or intent.time_in_force is not TimeInForce.GTC
            or intent.submit_index >= start_index
            # A no-fill observation need not mutate the order state.
            or order.last_processed_index is None
            or not intent.submit_index <= order.last_processed_index <= start_index
        ):
            raise ValueError(
                "pending continuation requires compatible native market orders"
            )
        seen.add(symbol)
        assert order.exact_cumulative_filled_quantity is not None
        residual = exact_quantity(intent.requested_quantity) - parse_quantity(
            order.exact_cumulative_filled_quantity
        )
        quantities[symbol] += residual

    equity = book.portfolio_value
    multipliers = book.contract_multipliers
    if not math.isfinite(equity) or equity <= 0.0 or multipliers is None:
        raise ValueError("pending continuation requires finite positive book equity")
    # Untouched symbols keep the canonical current book weights. The projected
    # symbols are conservatively rounded from exact outstanding quantities.
    projected = book.weights.copy()
    for symbol in seen:
        weight = (
            quantities[symbol]
            * Fraction.from_float(float(book.mark_prices[symbol]))
            * Fraction.from_float(float(multipliers[symbol]))
            / Fraction.from_float(equity)
        )
        projected[symbol] = _outward_weight(weight)
    return projected


def _validate_pending_continuation_config(cost: ExecutionCostConfig) -> None:
    if (
        cost.order_type != "market"
        or cost.processing_bar_volume_capacity
        or cost.max_leverage > 1.0
        or cost.collateral_haircut != 1.0
        or cost.slippage_std != 0.0
        or cost.tail_slippage_probability != 0.0
        or cost.margin_mode != "cross"
    ):
        raise ValueError(
            "pending continuation requires deterministic unlevered MARKET "
            "with previous-bar capacity and cross margin"
        )


PendingContinuationStatus = Literal["continued", "risk_fallback", "hold"]


def _execute_native_pending_transition(
    executor: MarketExecutor,
    book: BookState,
    pretrade_risk: PreTradeRisk,
    *,
    start_index: int,
    hold_unchanged_risk_target: bool,
) -> tuple[
    RiskConstrainedTarget, ExecutionResult, tuple[int, ...], PendingContinuationStatus
]:
    """Recheck residual exposure without resubmitting an actor target."""
    current = book.weights
    active = executor.compatibility_order_book.active_orders
    if active:
        projected = _pending_endpoint_weights(executor, book, start_index=start_index)
        # Independent fills can temporarily hold old and new exposure in
        # different symbols even if the full endpoint is safe. With one
        # order per symbol, the absolute maximum is at one of its endpoints.
        worst = np.where(np.abs(projected) > np.abs(current), projected, current)
        current_risk = pretrade_risk.constrain(
            current, current=current, drawdown=book.max_drawdown
        )
        projected_risk = pretrade_risk.constrain(
            projected, current=current, drawdown=book.max_drawdown
        )
        worst_risk = pretrade_risk.constrain(
            worst, current=current, drawdown=book.max_drawdown
        )
        if (
            np.array_equal(current_risk.weights, current)
            and np.array_equal(projected_risk.weights, projected)
            and np.array_equal(worst_risk.weights, worst)
        ):
            execution = executor._continue_pending_interval(
                book, start_index=start_index
            )
            if execution.next_index != start_index + 1 or execution.bars_advanced != 1:
                raise RuntimeError("native pending transition did not advance one bar")
            return projected_risk, execution, (), "continued"

    # No active orders, or preserving them no longer meets risk. Reuse
    # exact HOLD with ordinary hard-risk correction and cancellation rules.
    result = execute_native_target_transition(
        executor,
        book,
        pretrade_risk,
        current,
        start_index=start_index,
        hold_current_quantity_symbols=tuple(range(len(current))),
        hold_unchanged_risk_target=hold_unchanged_risk_target,
    )
    return (*result, "risk_fallback" if active else "hold")


__all__ = [
    "execute_native_target_transition",
    "validate_risk_execution_compatibility",
]
