"""Risk-constrained native target transitions over the canonical executor."""

from __future__ import annotations

import numpy as np

from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionResult, MarketExecutor


def _target_weights(value: object, *, n_symbols: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
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
    mask = np.asarray(value)
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
) -> tuple[RiskConstrainedTarget, ExecutionResult, tuple[int, ...]]:
    """Apply hard risk and execute exactly one native interval.

    Strategy and replay callers own action mapping and policy state. This
    transition accepts only numeric portfolio targets, applies risk against the
    pre-execution book, and advances the canonical MarketExecutor once. Exact
    quantity holds survive only when risk leaves their proposed weights intact.
    """
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


__all__ = [
    "execute_native_target_transition",
    "validate_risk_execution_compatibility",
]
