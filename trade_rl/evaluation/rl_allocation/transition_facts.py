"""Detach actual native economics before environment overwrite; no new ledger."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_decision import AllocationActionExecutionResult
from trade_rl.evaluation.objectives import net_equity_increment
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.diagnostics.funding import FundingBoundaryEvidence
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.liquidity import SymbolCapacityEvidence
from trade_rl.simulation.orders.model import (
    _QUANTITY_TOLERANCE,
    OrderBookState,
    OrderEvent,
    OrderStatus,
    OrderType,
    PendingOrder,
    _quantity_tolerance,
)
from trade_rl.simulation.quantities import exact_quantity, parse_quantity
from trade_rl.simulation.stateful.execution import StatefulExecutionResult
from trade_rl.strategies.allocation import (
    AfterCostTargetAllocator,
    AllocationContext,
    AllocationInputs,
)
from trade_rl.strategies.allocation_action import (
    AllocationActionContract,
    AllocationDecision,
)
from trade_rl.strategies.rl.allocation_fee_stress import retained_debt_economics_digest
from trade_rl.strategies.rl.allocation_preprocessing import _native_json


def _json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[ns]").astype("int64"))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {
            field.name: _json(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _pending(order: Any) -> dict[str, Any]:
    result = _json(order)
    result["intent"] = order.intent.canonical_payload()
    return result


def native_allocation_execution_facts(
    execution: StatefulExecutionResult,
) -> dict[str, Any]:
    """Detach canonical returned facts, without inventing an actor proposal."""
    book = deepcopy(execution.book)
    book_facts = {
        field.name: _json(getattr(book, field.name))
        for field in fields(book)
        if field.name not in ("_exact_quantities", "returns_history")
    }
    book_facts.update(
        exact_quantities=[str(q) for q in book.exact_quantities],
        equity=book.portfolio_value,
    )
    terminal_ids = {
        event.order_id for event in execution.order_events if event.new_status.terminal
    }
    return {
        "book": book_facts,
        "execution": {
            field.name: _json(getattr(execution, field.name))
            for field in fields(execution)
            if field.name not in ("book", "order_book", "order_events")
        },
        "order_events": [e.canonical_payload() for e in execution.order_events],
        "active_orders": [_pending(o) for o in execution.order_book.active_orders],
        "terminal_order_reasons": [
            {"order_id": o.order_id, "reason": o.terminal_reason}
            for o in execution.order_book.terminal_orders
            if o.order_id in terminal_ids
        ],
    }


def freeze_allocation_execution(
    env: Any, result: AllocationActionExecutionResult
) -> bytes:
    execution, proposal = result.execution, result.proposal
    if execution.bars_advanced != 1 or execution.next_index != env.index + 1:
        raise ValueError("trace requires the actual pre-overwrite one-bar clock")
    return canonical_json_bytes(
        {
            "decision_index": env.index,
            "decision_time_ns": _json(proposal.decision.baseline.context.decision_time),
            "processing_index": execution.next_index,
            "processing_time_ns": _json(env.dataset.timestamps[execution.next_index]),
            "symbol_index": env.symbol_index,
            "proposal": _json(proposal),
            "proposal_digest": proposal.digest,
            "decision_digest": proposal.decision.decision_digest,
            "risk": _json(result.risk_target),
            "risk_config": _json(env.risk.config),
            "current_weights": deepcopy(env.book).weights.tolist(),
            **native_allocation_execution_facts(execution),
        }
    )


def _closed(value: Any, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("transition facts contain undeclared or missing fields")
    return value


def _construct(cls: Any, raw: Any, *, clocks: tuple[str, ...] = ()) -> Any:
    values = dict(_closed(raw, {field.name for field in fields(cls)}))
    _scalars(cls, values)
    for name in clocks:
        if type(values[name]) is not int:
            raise ValueError("transition clock must be native integer nanoseconds")
        values[name] = np.datetime64(values[name], "ns")
    return cls(**values)


def _scalars(cls: Any, raw: dict[str, Any]) -> None:
    for field in fields(cls):
        if field.name not in raw:
            continue
        value = raw[field.name]
        if field.type == "float" and type(value) not in (float, int):
            raise ValueError("transition financial scalar must be numeric, not bool")
        if field.type == "int" and (type(value) is not int or value < 0):
            raise ValueError("transition counter must be a nonnegative native integer")
        if field.type == "bool" and type(value) is not bool:
            raise ValueError("transition flag must be a native boolean")


def _same(actual: Any, expected: Any) -> None:
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise ValueError("transition facts differ from their native reconstruction")


def _vector(raw: Any, size: int) -> np.ndarray:
    if (
        type(raw) is not list
        or len(raw) != size
        or any(type(x) not in (int, float) for x in raw)
    ):
        raise ValueError("transition vectors must preserve full one-dimensional shape")
    vector = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("transition vectors must be finite")
    return vector


def _active(order: PendingOrder, index: int) -> None:
    """Native active status/progress bounds, including exact underflow evidence."""
    assert order.exact_cumulative_filled_quantity is not None
    cumulative = parse_quantity(order.exact_cumulative_filled_quantity)
    remaining = exact_quantity(order.intent.requested_quantity) - cumulative
    last, trigger = order.last_processed_index, order.trigger_index
    prefill = order.status in (
        OrderStatus.SUBMITTED,
        OrderStatus.LATENCY_WAIT,
        OrderStatus.ELIGIBLE,
    )
    if (
        order.intent.order_type is not OrderType.MARKET
        or order.status is OrderStatus.TRIGGERED
        or trigger is not None
        or remaining == 0
        or cumulative == 0
        and order.cumulative_filled_notional != 0
        or prefill
        and (cumulative != 0 or trigger is not None)
        or order.status is OrderStatus.PARTIALLY_FILLED
        and cumulative == 0
        or order.status is OrderStatus.SUBMITTED
        and last is not None
        or order.status is not OrderStatus.SUBMITTED
        and last is None
        or last is not None
        and not order.intent.submit_index <= last <= index
        or order.status in (OrderStatus.ELIGIBLE, OrderStatus.PARTIALLY_FILLED)
        and (last is None or last < order.intent.eligible_index)
        or cumulative != 0
        and (
            abs(order.cumulative_filled_quantity) <= _QUANTITY_TOLERANCE
            or abs(order.remaining_quantity)
            <= _quantity_tolerance(
                order.intent.requested_quantity,
                order.cumulative_filled_quantity,
                order.remaining_quantity,
            )
        )
    ):
        raise ValueError(
            "transition active order differs from native fill/phase bounds"
        )


def validate_allocation_execution_facts(
    value: object, recipe: dict[str, Any], *, dataset_id: str, action_code: int
) -> float:
    """Check native algebra/shape/clock links; never execute a ledger transition."""
    try:
        return _validate_execution_facts(
            value, recipe, dataset_id=dataset_id, action_code=action_code
        )
    except (KeyError, TypeError, OverflowError, RecursionError, IndexError) as error:
        raise ValueError("invalid closed transition economics") from error


def validate_allocation_execution_cost_payload(
    value: object, *, economics_digest: str
) -> ExecutionCostConfig:
    """Read the complete actual native cost policy; never supply missing defaults."""
    try:
        _native_json(value)
        raw = _closed(
            value,
            {field.name for field in fields(ExecutionCostConfig)} | {"schema_version"},
        )
        if raw["schema_version"] != "execution_policy_v2":
            raise ValueError("transition requires native execution policy v2")
        values = {name: val for name, val in raw.items() if name != "schema_version"}
        _scalars(ExecutionCostConfig, values)
        for name in ("margin_mode", "order_type", "path_mode"):
            if type(values[name]) is not str:
                raise ValueError("transition execution mode must be native text")
        _vector(values["trigger_volume_fractions"], 4)
        values["trigger_volume_fractions"] = tuple(values["trigger_volume_fractions"])
        cost = ExecutionCostConfig(**values)
        _same(raw, cost.execution_policy_payload())
        if retained_debt_economics_digest(raw) != economics_digest:
            raise ValueError("transition actual cost differs from recipe economics")
        return cost
    except (KeyError, TypeError, OverflowError, RecursionError, IndexError) as error:
        raise ValueError("invalid complete native execution cost policy") from error


def validate_allocation_execution_facts_v2(
    value: object,
    recipe: dict[str, Any],
    *,
    dataset_id: str,
    action_code: int,
    actual_cost_payload: object,
) -> float:
    """Verify native static-cap intersection under pinned actual risk and costs."""
    try:
        cost = validate_allocation_execution_cost_payload(
            actual_cost_payload,
            economics_digest=recipe["runtime_profile"]["economics_digest"],
        )
        return _validate_execution_facts(
            value,
            recipe,
            dataset_id=dataset_id,
            action_code=action_code,
            actual_cost=cost,
        )
    except (KeyError, TypeError, OverflowError, RecursionError, IndexError) as error:
        raise ValueError("invalid closed transition economics") from error


def _validate_execution_facts(
    value: object,
    recipe: dict[str, Any],
    *,
    dataset_id: str,
    action_code: int,
    actual_cost: ExecutionCostConfig | None = None,
) -> float:
    _native_json(value)
    raw = _closed(
        value,
        {
            "decision_index",
            "decision_time_ns",
            "processing_index",
            "processing_time_ns",
            "symbol_index",
            "proposal",
            "proposal_digest",
            "decision_digest",
            "risk",
            "risk_config",
            "current_weights",
            "book",
            "execution",
            "order_events",
            "active_orders",
            "terminal_order_reasons",
        },
    )
    for name in (
        "decision_index",
        "processing_index",
        "symbol_index",
        "decision_time_ns",
        "processing_time_ns",
    ):
        if type(raw[name]) is not int or name.endswith("index") and raw[name] < 0:
            raise ValueError("transition indices/clocks must be native integers")
    if (
        raw["processing_index"] != raw["decision_index"] + 1
        or raw["processing_time_ns"] - raw["decision_time_ns"]
        != recipe["runtime_profile"]["decision_interval_seconds"] * 10**9
    ):
        raise ValueError("transition native one-bar clock differs")
    proposal = raw["proposal"]
    decision_raw = proposal["decision"]
    baseline = decision_raw["baseline"]
    inputs = _construct(
        AllocationInputs,
        baseline["inputs"],
        clocks=("decision_time", "available_at", "horizon_end"),
    )
    context = _construct(
        AllocationContext, baseline["context"], clocks=("decision_time",)
    )
    allocator = _construct(AfterCostTargetAllocator, baseline["allocator"])
    action = _construct(AllocationActionContract, decision_raw["action_contract"])
    decision_values: dict[str, Any] = {
        name: tuple(decision_raw[name])
        if name in ("feature_names", "feature_values")
        else decision_raw[name]
        for name in (
            "feature_names",
            "feature_values",
            "max_drawdown",
            "initial_capital",
            "remaining_steps",
            "pending_gross",
            "pending_count",
        )
    }
    decision = AllocationDecision(
        baseline=allocator.propose(inputs, context),
        action_contract=action,
        **decision_values,
    )
    expected = decision.propose(action_code)
    _same(proposal, _json(expected))
    if (
        raw["proposal_digest"] != expected.digest
        or raw["decision_digest"] != decision.decision_digest
        or inputs.decision_time != np.datetime64(raw["decision_time_ns"], "ns")
    ):
        raise ValueError("transition action/decision/clock digest differs")
    if actual_cost is None:
        _same(_json(allocator), recipe["allocator"])
    else:
        cap_risk = _construct(PreTradeRiskConfig, raw["risk_config"])
        if content_digest(cap_risk) != recipe["runtime_profile"]["risk_digest"]:
            raise ValueError("transition risk config differs from recipe")
        original = _construct(AfterCostTargetAllocator, recipe["allocator"])
        cap = min(cap_risk.max_abs_weight, actual_cost.max_leverage)
        resolved = replace(
            original,
            lower_weight=max(original.lower_weight, -cap),
            upper_weight=min(original.upper_weight, cap),
        )
        _same(_json(allocator), _json(resolved))
    _same(action.payload(), recipe["action"])
    if (
        list(decision.feature_names) != recipe["feature_names"]
        or decision.initial_capital != recipe["runtime_profile"]["initial_capital"]
    ):
        raise ValueError("transition decision differs from its recipe")
    book_raw = _closed(
        raw["book"],
        {field.name for field in fields(BookState)}
        - {"_exact_quantities", "returns_history"}
        | {"exact_quantities", "equity"},
    )
    size = len(book_raw["quantities"])
    book_values: dict[str, Any] = {
        name: _vector(book_raw[name], size)
        if name in ("quantities", "mark_prices", "contract_multipliers")
        else val
        for name, val in book_raw.items()
        if name not in ("exact_quantities", "equity")
    }
    book = BookState(
        **book_values, _exact_quantities=tuple(book_raw["exact_quantities"])
    )
    _scalars(BookState, book_raw)
    expected_book = {
        field.name: _json(getattr(book, field.name))
        for field in fields(book)
        if field.name not in ("_exact_quantities", "returns_history")
    }
    expected_book.update(
        exact_quantities=[str(q) for q in book.exact_quantities],
        equity=book.portfolio_value,
    )
    _same(book_raw, expected_book)
    if (
        book.as_of_index != raw["processing_index"]
        or book.as_of_dataset_id != dataset_id
        or not 0 <= raw["symbol_index"] < size
    ):
        raise ValueError("transition book differs from processing source")
    config = _construct(PreTradeRiskConfig, raw["risk_config"])
    if content_digest(config) != recipe["runtime_profile"]["risk_digest"]:
        raise ValueError("transition risk config differs from recipe")
    pretrade = _vector(raw["current_weights"], size)
    requested = np.zeros(size)
    requested[raw["symbol_index"]] = expected.target_weight
    if pretrade[raw["symbol_index"]] != context.current_weight or np.any(
        np.delete(pretrade, raw["symbol_index"])
    ):
        raise ValueError("transition pretrade weight differs from context")
    risk = PreTradeRisk(config).constrain(
        requested, current=pretrade, drawdown=decision.max_drawdown
    )
    for name in ("weights", "proposal_weights", "pretrade_weights"):
        _vector(raw["risk"][name], size)
    _same(raw["risk"], _json(risk))
    execution = _closed(
        raw["execution"],
        {field.name for field in fields(StatefulExecutionResult)}
        - {"book", "order_book", "order_events"},
    )
    _scalars(StatefulExecutionResult, execution)
    if (
        not 0 <= execution["fill_ratio"] <= 1
        or execution["requested_notional"] < 0
        or execution["filled_notional"] < 0
    ):
        raise ValueError(
            "transition unsigned notionals/fill ratio exceed native bounds"
        )
    if (
        type(execution["bars_advanced"]) is not int
        or execution["bars_advanced"] != 1
        or type(execution["next_index"]) is not int
        or execution["next_index"] != raw["processing_index"]
        or execution["termination_reason"] != book_raw["termination_reason"]
    ):
        raise ValueError("transition execution differs from actual book/clock")
    for name in (
        "requested_notional_by_symbol",
        "filled_notional_by_symbol",
        "participation_by_symbol",
        "cost_by_symbol",
    ):
        _vector(execution[name], size)
    for item in execution["capacity_evidence"]:
        _same(item, _json(_construct(SymbolCapacityEvidence, item)))
        if (
            any(
                item[name] < 0
                for name in (
                    "processing_volume",
                    "market_notional",
                    "initial_capacity_notional",
                    "consumed_capacity_notional",
                    "remaining_capacity_notional",
                )
            )
            or item["capacity_reference_price"] <= 0
            or item["contract_multiplier"] <= 0
            or not 0 < item["participation_limit"] <= 1
        ):
            raise ValueError("transition liquidity facts exceed native input bounds")
    for item in execution["funding_evidence"]:
        evidence = FundingBoundaryEvidence.from_mapping(item)
        _same(item, evidence.to_mapping())
        if (
            evidence.processing_index != raw["processing_index"]
            or evidence.timestamp_ns != raw["processing_time_ns"]
            or len(evidence.funding_due) != size
        ):
            raise ValueError(
                "transition funding boundary differs from full account clock"
            )
    orders = tuple(PendingOrder.from_mapping(item) for item in raw["active_orders"])
    OrderBookState(active_orders=orders, terminal_orders=())
    _same(raw["active_orders"], [_pending(o) for o in orders])
    events = tuple(OrderEvent.from_mapping(item) for item in raw["order_events"])
    _same(raw["order_events"], [e.canonical_payload() for e in events])
    for sequence, event in enumerate(events):
        if OrderStatus.TRIGGERED in (event.previous_status, event.new_status):
            raise ValueError("transition MARKET orders have no stop-trigger phase")
        kind = (
            "partially_filled"
            if event.event_type == "partial_fill"
            else event.event_type
        )
        if (
            event.event_type != "no_fill"
            and event.new_status.value != kind
            or event.event_type == "no_fill"
            and event.new_status != event.previous_status
            or event.event_type not in ("filled", "partial_fill")
            and (event.filled_quantity != 0 or event.filled_notional != 0)
        ):
            raise ValueError(
                "transition order event kind differs from native status/fill"
            )
        expected_time = (
            raw["decision_time_ns"]
            if event.processing_index == raw["decision_index"]
            else raw["processing_time_ns"]
        )
        if (
            event.sequence != sequence
            or event.processing_index
            not in (raw["decision_index"], raw["processing_index"])
            or event.timestamp_ns != expected_time
            or event.dataset_id != dataset_id
            or event.execution_policy_digest
            != recipe["runtime_profile"]["economics_digest"]
            or event.symbol_index != raw["symbol_index"]
        ):
            raise ValueError("transition interval-local order clock differs")
    for order in orders:
        _active(order, raw["processing_index"])
        observed = [event for event in events if event.order_id == order.order_id]
        if observed:
            latest = observed[-1]
            changes = [
                event
                for event in observed
                if event.event_type not in ("submitted", "no_fill")
            ]
            last = order.last_processed_index
            if changes:
                clock_matches = changes[-1].processing_index == last
            elif latest.event_type == "no_fill":
                # A new attempt leaves the carried immutable order untouched.
                # Submission needs its own native eligibility/latency mutation.
                clock_matches = (
                    all(event.event_type == "no_fill" for event in observed)
                    and last is not None
                    and last <= raw["decision_index"]
                )
            else:
                clock_matches = latest.event_type == "submitted" and last is None
            if (
                latest.new_status != order.status
                or latest.remaining_quantity != order.remaining_quantity
                or not clock_matches
            ):
                raise ValueError(
                    "transition final active order differs from its last native event"
                )
        if (
            order.intent.dataset_id != dataset_id
            or order.intent.execution_policy_digest
            != recipe["runtime_profile"]["economics_digest"]
            or order.intent.symbol_index != raw["symbol_index"]
        ):
            raise ValueError("transition active order differs from native source")
    terminals = {e.order_id for e in events if e.new_status.terminal}
    if (
        len(raw["terminal_order_reasons"]) != len(terminals)
        or {r["order_id"] for r in raw["terminal_order_reasons"]} != terminals
    ):
        raise ValueError("transition terminal reasons differ from actual events")
    for reason in raw["terminal_order_reasons"]:
        _closed(reason, {"order_id", "reason"})
        if type(reason["reason"]) is not str or not reason["reason"]:
            raise ValueError("transition terminal reason must be nonempty")
        observed = [
            event
            for event in events
            if event.order_id == reason["order_id"] and event.new_status.terminal
        ]
        if observed[-1].reason != reason["reason"]:
            raise ValueError("transition terminal reason differs from its native event")
    return net_equity_increment(
        context.equity, book.portfolio_value, initial_capital=decision.initial_capital
    )
