"""Pure array and action-code projection for joint portfolio policies."""

from __future__ import annotations

import math
from numbers import Real
from typing import cast

import numpy as np

from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPendingOrderSnapshot,
    PortfolioTargetProposal,
)

ACTION_HOLD = 0
ACTION_SHORT = 1
ACTION_FLAT = 2
ACTION_LONG = 3
ACTION_CODES = (ACTION_HOLD, ACTION_SHORT, ACTION_FLAT, ACTION_LONG)

PORTFOLIO_GYM_OBSERVATION_SCHEMA = "shared_portfolio_gym_observation_v1"
PORTFOLIO_GYM_OBSERVATION_KEYS = (
    "features",
    "feature_available",
    "feature_staleness",
    "global_features",
    "global_feature_available",
    "account",
    "portfolio",
    "risk",
    "active_orders",
    "clock",
    "terminal",
)
PORTFOLIO_STATE_FIELDS = (
    "cash",
    "portfolio_value",
    "peak_value",
    "max_drawdown",
    "turnover_total",
    "total_cost",
    "funding_pnl",
    "borrow_cost",
    "margin_used",
    "maintenance_margin",
    "maintenance_requirement",
    "margin_deficit",
    "insolvent",
    "termination_reason_code",
)
RISK_STATE_FIELDS = (
    "max_gross",
    "max_abs_weight",
    "max_turnover",
    "has_max_turnover",
    "drawdown_start",
    "drawdown_stop",
    "emergency_turnover_override",
    "fail_closed_tolerance",
)
ACTIVE_ORDER_FIELDS = (
    "present",
    "order_type_code",
    "time_in_force_code",
    "status_code",
    "requested_quantity",
    "remaining_quantity",
    "reduce_only",
    "submit_index",
    "eligible_index",
    "has_last_processed_index",
    "last_processed_index",
    "has_trigger_index",
    "trigger_index",
    "has_expiry_index",
    "expiry_index",
    "has_limit_price",
    "limit_price",
    "has_stop_price",
    "stop_price",
    "has_submission_reference_price",
    "submission_reference_price",
)

_ORDER_TYPE_CODES = {
    "market": 1,
    "limit": 2,
    "stop_market": 3,
}
_TIME_IN_FORCE_CODES = {
    "ioc": 1,
    "day": 2,
    "gtc": 3,
}
_ORDER_STATUS_CODES = {
    status: index
    for index, status in enumerate(
        (
            "submitted",
            "latency_wait",
            "eligible",
            "triggered",
            "partially_filled",
            "filled",
            "rejected",
            "expired",
            "cancelled",
        ),
        start=1,
    )
}
_TERMINATION_REASON_CODES = {
    reason: index
    for index, reason in enumerate(
        (
            "minimum_equity",
            "execution_cost_exhaustion",
            "margin_call",
            "liquidation",
            "drawdown_stop",
            "insolvency",
        ),
        start=1,
    )
}


def encode_portfolio_decision_snapshot(
    snapshot: PortfolioDecisionSnapshot,
) -> dict[str, np.ndarray]:
    """Return detached raw-v1 arrays for one nonterminal portfolio snapshot."""

    if not isinstance(snapshot, PortfolioDecisionSnapshot):
        raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
    account = snapshot.account
    reason = account.termination_reason
    if reason is None:
        reason_code = 0
    else:
        try:
            reason_code = _TERMINATION_REASON_CODES[reason]
        except KeyError as error:
            raise ValueError("snapshot has an unknown termination reason") from error
    risk = snapshot.risk
    return {
        "features": np.asarray(snapshot.features, dtype=np.float64).copy(),
        "feature_available": np.asarray(
            snapshot.feature_available, dtype=np.int8
        ).copy(),
        "feature_staleness": np.asarray(
            snapshot.feature_staleness, dtype=np.float64
        ).copy(),
        "global_features": np.asarray(
            snapshot.global_features, dtype=np.float64
        ).copy(),
        "global_feature_available": np.asarray(
            snapshot.global_feature_available, dtype=np.int8
        ).copy(),
        "account": np.column_stack(
            (
                account.quantities,
                account.weights,
                account.mark_prices,
                account.contract_multipliers,
            )
        ).astype(np.float64, copy=False),
        "portfolio": np.asarray(
            (
                account.cash,
                account.portfolio_value,
                account.peak_value,
                account.max_drawdown,
                account.turnover_total,
                account.total_cost,
                account.funding_pnl,
                account.borrow_cost,
                account.margin_used,
                account.maintenance_margin,
                account.maintenance_requirement,
                account.margin_deficit,
                float(account.insolvent),
                float(reason_code),
            ),
            dtype=np.float64,
        ),
        "risk": np.asarray(
            (
                risk.max_gross,
                risk.max_abs_weight,
                0.0 if risk.max_turnover is None else risk.max_turnover,
                float(risk.max_turnover is not None),
                risk.drawdown_start,
                risk.drawdown_stop,
                float(risk.emergency_turnover_override),
                risk.fail_closed_tolerance,
            ),
            dtype=np.float64,
        ),
        "active_orders": encode_portfolio_active_orders(
            len(snapshot.symbols), snapshot.active_orders
        ),
        "clock": np.asarray(
            (
                snapshot.index,
                snapshot.timestamp_ns,
                snapshot.agent_stop_index,
                snapshot.remaining_decisions,
            ),
            dtype=np.int64,
        ),
        "terminal": np.zeros(1, dtype=np.int8),
    }


def encode_portfolio_active_orders(
    n_symbols: int,
    active_orders: tuple[PortfolioPendingOrderSnapshot, ...],
) -> np.ndarray:
    encoded = np.zeros((n_symbols, len(ACTIVE_ORDER_FIELDS)), dtype=np.float64)
    seen: set[int] = set()
    for order in active_orders:
        symbol_index = order.symbol_index
        if symbol_index in seen:
            raise ValueError(
                "the observation supports at most one active order per symbol"
            )
        seen.add(symbol_index)
        if order.submission_reference_price is None:
            raise ValueError("active order price context is incomplete")
        try:
            order_type = _ORDER_TYPE_CODES[order.order_type]
            time_in_force = _TIME_IN_FORCE_CODES[order.time_in_force]
            status = _ORDER_STATUS_CODES[order.status]
        except KeyError as error:
            raise ValueError(
                "active order contains an unsupported enum value"
            ) from error
        values = (
            1.0,
            float(order_type),
            float(time_in_force),
            float(status),
            order.requested_quantity,
            order.remaining_quantity,
            float(order.reduce_only),
            float(order.submit_index),
            float(order.eligible_index),
            float(order.last_processed_index is not None),
            0.0
            if order.last_processed_index is None
            else float(order.last_processed_index),
            float(order.trigger_index is not None),
            0.0 if order.trigger_index is None else float(order.trigger_index),
            float(order.expiry_index is not None),
            0.0 if order.expiry_index is None else float(order.expiry_index),
            float(order.limit_price is not None),
            0.0 if order.limit_price is None else order.limit_price,
            float(order.stop_price is not None),
            0.0 if order.stop_price is None else order.stop_price,
            1.0,
            order.submission_reference_price,
        )
        encoded[symbol_index] = np.asarray(values, dtype=np.float64)
    return encoded


def decode_portfolio_action_codes(action: object, *, n_symbols: int) -> tuple[int, ...]:
    """Validate one native integer action code per symbol without coercion."""

    if type(n_symbols) is not int or n_symbols <= 0:
        raise ValueError("n_symbols must be a positive native integer")
    if isinstance(action, (tuple, list)) and any(
        isinstance(value, (bool, np.bool_)) for value in action
    ):
        raise TypeError("action codes may not contain booleans")
    values = np.asarray(action)
    if values.shape != (n_symbols,):
        raise ValueError("action must contain exactly one code per symbol")
    if values.dtype.kind not in "iu":
        raise TypeError("action codes must be integers, not booleans or floats")
    if np.any(values < ACTION_HOLD) or np.any(values > ACTION_LONG):
        raise ValueError("action code is outside the declared action space")
    return tuple(int(value) for value in values)


def portfolio_action_to_proposal(
    action: object,
    snapshot: PortfolioDecisionSnapshot,
    target_abs_weights: tuple[float, ...],
) -> PortfolioTargetProposal:
    """Map action codes to numeric targets and explicit quantity-preserving HOLDs."""

    if not isinstance(snapshot, PortfolioDecisionSnapshot):
        raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
    if type(target_abs_weights) is not tuple or len(target_abs_weights) != len(
        snapshot.symbols
    ):
        raise TypeError("target_abs_weights must be a symbol-aligned tuple")
    weights: list[float] = []
    for value in cast(tuple[object, ...], target_abs_weights):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("target_abs_weights must contain real non-boolean values")
        try:
            resolved = float(value)
        except OverflowError as error:
            raise ValueError("target_abs_weights must be finite") from error
        if not math.isfinite(resolved) or resolved < 0.0:
            raise ValueError("target_abs_weights must be finite and non-negative")
        weights.append(resolved)

    codes = decode_portfolio_action_codes(action, n_symbols=len(snapshot.symbols))
    target_weights: list[float] = []
    hold_symbols: list[int] = []
    for symbol_index, code in enumerate(codes):
        if code == ACTION_HOLD:
            target_weights.append(snapshot.account.weights[symbol_index])
            hold_symbols.append(symbol_index)
        elif code == ACTION_SHORT:
            target_weights.append(-weights[symbol_index])
        elif code == ACTION_FLAT:
            target_weights.append(0.0)
        elif code == ACTION_LONG:
            target_weights.append(weights[symbol_index])
        else:
            raise ValueError("action code is not supported")
    return PortfolioTargetProposal(
        target_weights=tuple(target_weights),
        hold_current_quantity_symbols=tuple(hold_symbols),
    )


__all__ = [
    "ACTION_CODES",
    "ACTION_FLAT",
    "ACTION_HOLD",
    "ACTION_LONG",
    "ACTION_SHORT",
    "ACTIVE_ORDER_FIELDS",
    "PORTFOLIO_GYM_OBSERVATION_KEYS",
    "PORTFOLIO_GYM_OBSERVATION_SCHEMA",
    "PORTFOLIO_STATE_FIELDS",
    "RISK_STATE_FIELDS",
    "decode_portfolio_action_codes",
    "encode_portfolio_active_orders",
    "encode_portfolio_decision_snapshot",
    "portfolio_action_to_proposal",
]
