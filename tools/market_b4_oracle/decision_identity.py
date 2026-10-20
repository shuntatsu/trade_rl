"""Rebuild allocation identity payloads from explicit caller-owned facts.

This comparison module does not admit source data or authenticate the origin of
any supplied account, forecast, allocation, or risk component.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from fractions import Fraction
from hashlib import sha256
from typing import Any

from tools.market_b4_oracle.expected_intent import ExpectedIntent

BOOK_FIELDS = frozenset(
    {
        "quantities",
        "cash",
        "mark_prices",
        "peak_value",
        "contract_multipliers",
        "max_drawdown",
        "turnover_total",
        "total_cost",
        "funding_pnl",
        "fill_count",
        "rebalance_events",
        "returns_history",
        "borrow_cost",
        "margin_used",
        "maintenance_margin",
        "maintenance_requirement",
        "margin_deficit",
        "insolvent",
        "termination_reason",
        "as_of_index",
        "as_of_dataset_id",
        "exact_quantities",
    }
)
INPUT_FIELDS = frozenset(
    {
        "symbol",
        "decision_time",
        "available_at",
        "horizon_end",
        "source_identity",
        "expected_simple_return",
        "return_variance",
        "buy_cost",
        "sell_cost",
        "exit_cost",
        "funding_return",
        "borrow_return",
        "cash_return",
        "return_unit",
    }
)
ALLOCATOR_FIELDS = frozenset(
    {"lower_weight", "upper_weight", "max_turnover", "risk_aversion"}
)
DECISION_FIELDS = frozenset(
    {
        "feature_names",
        "feature_values",
        "max_drawdown",
        "initial_capital",
        "remaining_steps",
        "pending_gross",
        "pending_count",
    }
)


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical JSON floats must be finite")
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("canonical JSON mapping keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported explicit JSON component: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the native canonical JSON subset without importing production."""
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _closed_mapping(value: Any, expected: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} fields are not closed")
    return {key: value[key] for key in expected}


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite non-boolean number")
    return float(value)


def _identity_float(value: Any, name: str, *, positive: bool = False) -> float:
    """Preserve native pre-normalization identity type; ints are distinct."""
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class OwnedAccountComponents:
    """Explicit account/source facts; no runtime BookState or digest input."""

    account_id: str
    dataset_id: str
    symbols: tuple[str, str]
    execution_policy_digest: str
    risk_config: Mapping[str, Any]
    symbol_index: int
    decision_index: int
    decision_time: str
    book: Mapping[str, Any]
    active_orders: tuple[OwnedPendingOrder, ...] = ()
    terminal_orders: tuple[OwnedPendingOrder, ...] = ()


@dataclass(frozen=True, slots=True)
class OwnedIntentComponents:
    """Intent fields excluding all identity hashes, which are rebuilt here."""

    identity_source: ActorIdentity
    dataset_id: str
    execution_policy_digest: str
    symbol_index: int
    requested_quantity: float
    order_type: str
    time_in_force: str
    limit_price: float | None
    stop_price: float | None
    submit_index: int
    eligible_index: int
    expiry_index: int | None
    submission_reference_price: float
    decision_equity: float
    replaced_order: OwnedPendingOrder | None = None
    reduce_only: bool = False


@dataclass(frozen=True, slots=True)
class OwnedPendingOrder:
    """Explicit order progress plus a builder-produced prior identity source."""

    intent: OwnedIntentComponents
    remaining_quantity: float
    cumulative_filled_quantity: float = 0.0
    cumulative_filled_notional: float = 0.0
    status: str = "submitted"
    trigger_index: int | None = None
    last_processed_index: int | None = None
    terminal_reason: str | None = None
    evidence_version: int = 0
    exact_cumulative_filled_quantity: str = "0"


@dataclass(frozen=True, slots=True, init=False)
class ActorIdentity:
    """Builder-only output; it carries no source authentication."""

    context_state_bytes: bytes
    context_state_digest: str
    context_bytes: bytes
    proposal_bytes: bytes
    proposal_digest: str
    decision_bytes: bytes
    decision_digest: str
    action_bytes: bytes
    action_digest: str
    actor_decision_identity: str
    target_identity: str | None
    current_weights: tuple[float, float]


def _order_mapping(
    order: OwnedPendingOrder, account: OwnedAccountComponents
) -> dict[str, Any]:
    intent = order.intent
    if not isinstance(intent.identity_source, ActorIdentity):
        raise ValueError("order identity source must be a builder result")
    target_identity = intent.identity_source.target_identity
    if target_identity is None:
        raise ValueError("unchanged prior target cannot source a submitted order")
    if (
        intent.dataset_id != account.dataset_id
        or intent.execution_policy_digest != account.execution_policy_digest
    ):
        raise ValueError("owned order source differs from account dataset/policy")
    symbol_index = _integer(intent.symbol_index, "symbol_index")
    if symbol_index >= 2:
        raise ValueError("symbol_index is outside the two-symbol identity scope")
    if type(intent.reduce_only) is not bool:
        raise ValueError("reduce_only must be boolean")
    requested = _identity_float(intent.requested_quantity, "requested_quantity")
    if abs(requested) <= 1e-12:
        raise ValueError("requested_quantity must be finite and non-zero")
    if not isinstance(intent.order_type, str) or not isinstance(
        intent.time_in_force, str
    ):
        raise ValueError("order enum values must be supplied as explicit strings")
    if intent.order_type != "market" or intent.time_in_force != "gtc":
        raise ValueError("order identity prototype supports only MARKET/GTC")
    if intent.limit_price is not None or intent.stop_price is not None:
        raise ValueError("MARKET/GTC identity must not define limit or stop prices")
    submit_index = _integer(intent.submit_index, "submit_index")
    eligible_index = _integer(intent.eligible_index, "eligible_index")
    if eligible_index < submit_index:
        raise ValueError("eligible_index must not precede submit_index")
    if intent.expiry_index is not None:
        expiry = _integer(intent.expiry_index, "expiry_index")
        if expiry < eligible_index:
            raise ValueError("expiry_index must not precede eligible_index")
    replaced_id = (
        None
        if intent.replaced_order is None
        else _order_mapping(intent.replaced_order, account)["intent"]["order_id"]
    )
    rebuilt_intent = ExpectedIntent.create(
        dataset_id=intent.dataset_id,
        target_identity=target_identity,
        execution_policy_digest=intent.execution_policy_digest,
        symbol_index=symbol_index,
        requested_quantity=requested,
        order_type=intent.order_type,
        time_in_force=intent.time_in_force,
        limit_price=intent.limit_price,
        stop_price=intent.stop_price,
        submit_index=submit_index,
        eligible_index=eligible_index,
        expiry_index=intent.expiry_index,
        submission_reference_price=_identity_float(
            intent.submission_reference_price,
            "submission_reference_price",
            positive=True,
        ),
        decision_equity=_identity_float(
            intent.decision_equity, "decision_equity", positive=True
        ),
        replaced_order_id=replaced_id,
        reduce_only=intent.reduce_only,
    )
    # PendingOrder's dataclass mapping includes reduce_only=False; asdict keeps
    # that native-shaped field while ExpectedIntent owns the ID preimage.
    intent_full = asdict(rebuilt_intent)
    status = order.status.value if isinstance(order.status, Enum) else order.status
    if not isinstance(status, str):
        raise ValueError("order status must be an explicit string")
    if status not in {
        "submitted",
        "latency_wait",
        "eligible",
        "triggered",
        "partially_filled",
        "filled",
        "cancelled",
        "expired",
        "rejected",
    }:
        raise ValueError("order status is unsupported")
    if status in {"filled", "cancelled", "expired", "rejected"}:
        if not isinstance(order.terminal_reason, str) or not order.terminal_reason:
            raise ValueError("terminal order requires a terminal reason")
    elif order.terminal_reason is not None:
        raise ValueError("nonterminal order may not have a terminal reason")
    exact = order.exact_cumulative_filled_quantity
    if not isinstance(exact, str) or str(Fraction(exact)) != exact:
        raise ValueError("exact cumulative quantity must be a canonical fraction")
    return {
        "intent": intent_full,
        "remaining_quantity": _number(order.remaining_quantity, "remaining_quantity"),
        "cumulative_filled_quantity": _number(
            order.cumulative_filled_quantity, "cumulative_filled_quantity"
        ),
        "cumulative_filled_notional": _number(
            order.cumulative_filled_notional, "cumulative_filled_notional"
        ),
        "status": status,
        "trigger_index": order.trigger_index,
        "last_processed_index": order.last_processed_index,
        "terminal_reason": order.terminal_reason,
        "evidence_version": _integer(order.evidence_version, "evidence_version"),
        "exact_cumulative_filled_quantity": exact,
    }


def _account_state(
    account: OwnedAccountComponents,
) -> tuple[dict[str, Any], dict[str, Any], tuple[float, float]]:
    if (
        not isinstance(account.account_id, str)
        or not account.account_id.strip()
        or not isinstance(account.dataset_id, str)
        or not account.dataset_id.strip()
        or not isinstance(account.execution_policy_digest, str)
        or len(account.execution_policy_digest) != 64
        or any(
            char not in "0123456789abcdef" for char in account.execution_policy_digest
        )
    ):
        raise ValueError("account/dataset/policy identity is invalid")
    if (
        type(account.symbols) is not tuple
        or len(account.symbols) != 2
        or any(not isinstance(symbol, str) or not symbol for symbol in account.symbols)
    ):
        raise ValueError("identity prototype requires two explicit symbols")
    symbol_index = _integer(account.symbol_index, "symbol_index")
    decision_index = _integer(account.decision_index, "decision_index")
    if symbol_index >= 2 or not isinstance(account.decision_time, str):
        raise ValueError("decision source clock/symbol is invalid")
    book = _closed_mapping(account.book, BOOK_FIELDS, "book")
    vectors = {}
    for name in ("quantities", "mark_prices", "contract_multipliers"):
        value = book[name]
        if type(value) not in (tuple, list) or len(value) != 2:
            raise ValueError(f"{name} must preserve two-symbol shape")
        vectors[name] = tuple(_number(item, name) for item in value)
    exact = book["exact_quantities"]
    if type(exact) not in (tuple, list) or len(exact) != 2:
        raise ValueError("exact_quantities must preserve two-symbol shape")
    exact_text = tuple(str(Fraction(item)) for item in exact)
    if any(
        type(item) is not str or item != normalized
        for item, normalized in zip(exact, exact_text, strict=True)
    ):
        raise ValueError("exact quantities must be canonical rational strings")
    if any(price <= 0.0 for price in vectors["mark_prices"]):
        raise ValueError("mark prices must be positive")
    if any(multiplier <= 0.0 for multiplier in vectors["contract_multipliers"]):
        raise ValueError("contract multipliers must be positive")
    normalized_book = dict(book)
    normalized_book.update(vectors)
    normalized_book["exact_quantities"] = exact_text
    normalized_book["returns_history"] = tuple(
        _number(item, "returns_history") for item in book["returns_history"]
    )
    for name in (
        "cash",
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
    ):
        normalized_book[name] = _number(book[name], name)
    for name in ("fill_count", "rebalance_events"):
        normalized_book[name] = _integer(book[name], name)
    if type(book["insolvent"]) is not bool:
        raise ValueError("insolvent must be boolean")
    if book["termination_reason"] is not None and not isinstance(
        book["termination_reason"], (str, Enum)
    ):
        raise ValueError("termination_reason must be text or null")
    for name in ("as_of_index",):
        if book[name] is not None:
            _integer(book[name], name)
    if book["as_of_dataset_id"] is not None and not isinstance(
        book["as_of_dataset_id"], str
    ):
        raise ValueError("as_of_dataset_id must be text or null")
    if book["as_of_dataset_id"] is not None and (
        len(book["as_of_dataset_id"]) != 64
        or any(char not in "0123456789abcdef" for char in book["as_of_dataset_id"])
    ):
        raise ValueError("as_of_dataset_id must be a lowercase SHA256 identity")

    active = tuple(_order_mapping(order, account) for order in account.active_orders)
    terminal = tuple(
        _order_mapping(order, account) for order in account.terminal_orders
    )
    active_ids = {row["intent"]["order_id"] for row in active}
    terminal_ids = {row["intent"]["order_id"] for row in terminal}
    if len(active_ids) != len(active) or len(terminal_ids) != len(terminal):
        raise ValueError("duplicate owned order identity")
    if active_ids & terminal_ids:
        raise ValueError("order identity may not be both active and terminal")
    terminal_statuses = {"filled", "cancelled", "expired", "rejected"}
    if any(row["status"] in terminal_statuses for row in active):
        raise ValueError("active order list contains a terminal status")
    if any(row["status"] not in terminal_statuses for row in terminal):
        raise ValueError("terminal order list contains a nonterminal status")
    position_values = tuple(
        quantity * price * multiplier
        for quantity, price, multiplier in zip(
            vectors["quantities"],
            vectors["mark_prices"],
            vectors["contract_multipliers"],
            strict=True,
        )
    )
    equity = float(normalized_book["cash"] + sum(position_values))
    if not math.isfinite(equity) or equity <= 0.0:
        raise ValueError("derived equity must be finite and positive")
    weights: tuple[float, float] = (
        float(position_values[0] / equity),
        float(position_values[1] / equity),
    )
    pending = sum(
        (Fraction(str(order.remaining_quantity)) for order in account.active_orders),
        Fraction(0),
    )
    state_payload = {
        "schema": "independent_allocation_context_v1",
        "account_id": account.account_id,
        "dataset_id": account.dataset_id,
        "execution_policy": account.execution_policy_digest,
        "risk_config": _json_value(account.risk_config),
        "symbol_index": symbol_index,
        "decision_index": decision_index,
        "decision_time": account.decision_time,
        "book": normalized_book,
        "active_orders": active,
        "terminal_orders": terminal,
    }
    context_fields = {
        "account_id": account.account_id,
        "symbol": account.symbols[symbol_index],
        "decision_time": account.decision_time,
        "state_digest": _digest(state_payload),
        "current_weight": weights[symbol_index],
        "quantity": exact_text[symbol_index],
        "cash": normalized_book["cash"],
        "equity": equity,
        "pending_remaining": str(pending),
    }
    return state_payload, context_fields, weights


def build_actor_identity(
    account: OwnedAccountComponents,
    allocation_inputs: Mapping[str, Any],
    allocator: Mapping[str, Any],
    *,
    independent_target_weight: float,
    independent_objective_value: float,
    decision_fields: Mapping[str, Any],
    action_contract: Mapping[str, Any],
    action_code: int,
    final_risk_weights: tuple[float, float],
) -> ActorIdentity:
    """Build actor hashes from explicit inputs, never caller-supplied hashes.

    Forecast/source identity, risk result, and optimization result remain
    caller-owned assertions; this function reproduces the payload and hash
    contract but does not validate their source or calculation.
    """
    state_payload, context, current_weights = _account_state(account)
    inputs = _closed_mapping(allocation_inputs, INPUT_FIELDS, "allocation inputs")
    alloc = _closed_mapping(allocator, ALLOCATOR_FIELDS, "allocator")
    decisions = _closed_mapping(decision_fields, DECISION_FIELDS, "decision")
    contract = _closed_mapping(
        action_contract, frozenset({"mode", "scale"}), "action contract"
    )
    for name in ("symbol", "source_identity", "return_unit"):
        if not isinstance(inputs[name], str):
            raise ValueError(f"{name} must be explicit text")
    for name in ("decision_time", "available_at", "horizon_end"):
        if not isinstance(inputs[name], str):
            raise ValueError(f"{name} must be the explicit native time string")
    if not isinstance(contract["mode"], str) or contract["mode"] not in (
        "direct",
        "residual",
    ):
        raise ValueError("action mode must be direct or residual")
    scale = _number(contract["scale"], "action scale")
    if not 0.0 < scale <= 1.0:
        raise ValueError("action scale must be within (0, 1]")
    if type(decisions["feature_names"]) not in (tuple, list) or type(
        decisions["feature_values"]
    ) not in (tuple, list):
        raise ValueError("feature names/values must be explicit ordered sequences")
    if len(decisions["feature_names"]) != len(decisions["feature_values"]):
        raise ValueError("feature names and values must have matching lengths")

    baseline_target = _number(independent_target_weight, "target weight")
    objective = _number(independent_objective_value, "objective value")
    context_fields = dict(context)
    proposal_payload: dict[str, Any] = {
        "inputs": inputs,
        "context": context_fields,
        "allocator": alloc,
        "target": baseline_target,
        "objective_value": objective,
    }
    proposal_digest = _digest(proposal_payload)
    current = _number(context["current_weight"], "current weight")
    lower = _number(alloc["lower_weight"], "lower_weight")
    upper = _number(alloc["upper_weight"], "upper_weight")
    max_turnover = alloc["max_turnover"]
    if max_turnover is not None:
        turnover = _number(max_turnover, "max_turnover")
        lower = max(lower, current - turnover)
        upper = min(upper, current + turnover)
    feasible_bounds = (float(lower), float(upper))
    decision_payload = {
        "schema": "allocation_decision_v1",
        "baseline": proposal_digest,
        "action_contract": {
            "schema": "allocation_action_v1",
            "mode": contract["mode"],
            "scale": scale,
        },
        "feasible_bounds": feasible_bounds,
        "feature_names": decisions["feature_names"],
        "feature_values": decisions["feature_values"],
        "max_drawdown": decisions["max_drawdown"],
        "initial_capital": decisions["initial_capital"],
        "remaining_steps": decisions["remaining_steps"],
        "pending_gross": decisions["pending_gross"],
        "pending_count": decisions["pending_count"],
    }
    decision_digest = _digest(decision_payload)
    code = _integer(action_code, "action_code")
    if code > 3:
        raise ValueError("action_code must be in [0, 3]")
    if contract["mode"] == "direct":
        if code == 0:
            action, raw_target, target, is_hold = (
                "quantity_hold",
                current,
                current,
                True,
            )
        else:
            labels = ("negative", "flat", "positive")
            raw_target = (-scale, 0.0, scale)[code - 1]
            target = min(upper, max(lower, raw_target))
            action, is_hold = labels[code - 1], target == current
    elif code == 0:
        action, raw_target, target, is_hold = (
            "quantity_hold",
            current,
            current,
            True,
        )
    else:
        action = ("negative", "baseline", "positive")[code - 1]
        raw_target = baseline_target
        if code != 2:
            endpoint = lower if code == 1 else upper
            raw_target = (
                endpoint
                if scale == 1.0
                else baseline_target + scale * (endpoint - baseline_target)
            )
        target = raw_target
        is_hold = target == current
    action_payload = {
        "schema": "allocation_action_proposal_v1",
        "decision": decision_digest,
        "raw_action": code,
        "action": action,
        "raw_target_weight": float(raw_target),
        "target_weight": float(target),
        "is_hold": bool(is_hold),
    }
    action_digest = _digest(action_payload)
    actor_decision_identity = (
        proposal_digest if target == baseline_target else action_digest
    )
    if type(final_risk_weights) not in (tuple, list) or len(final_risk_weights) != 2:
        raise ValueError("risk target must preserve the complete two-symbol vector")
    risk_weights = tuple(
        _number(value, "final_risk_weights") for value in final_risk_weights
    )
    target_identity = (
        None
        if risk_weights == current_weights
        else _digest(
            {"decision": actor_decision_identity, "target": list(risk_weights)}
        )
    )
    output = object.__new__(ActorIdentity)
    for name, value in {
        "context_state_bytes": canonical_json_bytes(state_payload),
        "context_state_digest": context["state_digest"],
        "context_bytes": canonical_json_bytes(context_fields),
        "proposal_bytes": canonical_json_bytes(proposal_payload),
        "proposal_digest": proposal_digest,
        "decision_bytes": canonical_json_bytes(decision_payload),
        "decision_digest": decision_digest,
        "action_bytes": canonical_json_bytes(action_payload),
        "action_digest": action_digest,
        "actor_decision_identity": actor_decision_identity,
        "target_identity": target_identity,
        "current_weights": current_weights,
    }.items():
        object.__setattr__(output, name, value)
    return output


def owned_order_mapping(
    order: OwnedPendingOrder, account: OwnedAccountComponents
) -> dict[str, Any]:
    """Expose one rebuilt native-shaped PendingOrder mapping for differential use."""
    return _order_mapping(order, account)


__all__ = [
    "ActorIdentity",
    "OwnedAccountComponents",
    "OwnedIntentComponents",
    "OwnedPendingOrder",
    "build_actor_identity",
    "canonical_json_bytes",
    "owned_order_mapping",
]
