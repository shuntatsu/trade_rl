"""Canonical deterministic replay and immutable execution-event artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes, to_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.domain.common import require_sha256
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderEvent,
    OrderStatus,
    PendingOrder,
)

REPLAY_EVIDENCE_SCHEMA = "stateful_execution_replay_v1"
EXECUTION_REPLAY_IDENTITY_SCHEMA = "execution_replay_identity_v1"
EXECUTION_EVENT_ARTIFACT_FILE_NAME = "order-events.json"
EXECUTION_EVENT_ARTIFACT_SCHEMA = "execution_order_event_artifact_v2"
_TERMINAL_BOOK_SCHEMA = "execution_terminal_book_v1"
_TERMINAL_ORDER_BOOK_SCHEMA = "execution_terminal_order_book_v1"
_CONTENT_ADDRESSED_SUFFIX = ".execution-replay.json"


_EVENT_TRANSITIONS: dict[str, tuple[frozenset[OrderStatus], OrderStatus | None]] = {
    "submitted": (frozenset({OrderStatus.SUBMITTED}), OrderStatus.SUBMITTED),
    "latency_wait": (
        frozenset({OrderStatus.SUBMITTED, OrderStatus.LATENCY_WAIT}),
        OrderStatus.LATENCY_WAIT,
    ),
    "eligible": (
        frozenset({OrderStatus.SUBMITTED, OrderStatus.LATENCY_WAIT}),
        OrderStatus.ELIGIBLE,
    ),
    "triggered": (
        frozenset(
            {
                OrderStatus.ELIGIBLE,
                OrderStatus.TRIGGERED,
                OrderStatus.PARTIALLY_FILLED,
            }
        ),
        OrderStatus.TRIGGERED,
    ),
    "no_fill": (
        frozenset(
            {
                OrderStatus.ELIGIBLE,
                OrderStatus.TRIGGERED,
                OrderStatus.PARTIALLY_FILLED,
            }
        ),
        None,
    ),
    "partial_fill": (
        frozenset(
            {
                OrderStatus.ELIGIBLE,
                OrderStatus.TRIGGERED,
                OrderStatus.PARTIALLY_FILLED,
            }
        ),
        OrderStatus.PARTIALLY_FILLED,
    ),
    "filled": (
        frozenset(
            {
                OrderStatus.ELIGIBLE,
                OrderStatus.TRIGGERED,
                OrderStatus.PARTIALLY_FILLED,
            }
        ),
        OrderStatus.FILLED,
    ),
    "rejected": (
        frozenset(
            {
                OrderStatus.SUBMITTED,
                OrderStatus.LATENCY_WAIT,
                OrderStatus.ELIGIBLE,
            }
        ),
        OrderStatus.REJECTED,
    ),
    "expired": (
        frozenset(
            {
                OrderStatus.SUBMITTED,
                OrderStatus.LATENCY_WAIT,
                OrderStatus.ELIGIBLE,
                OrderStatus.TRIGGERED,
                OrderStatus.PARTIALLY_FILLED,
            }
        ),
        OrderStatus.EXPIRED,
    ),
    "cancelled": (
        frozenset(
            {
                OrderStatus.SUBMITTED,
                OrderStatus.LATENCY_WAIT,
                OrderStatus.ELIGIBLE,
                OrderStatus.TRIGGERED,
                OrderStatus.PARTIALLY_FILLED,
            }
        ),
        OrderStatus.CANCELLED,
    ),
}


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} keys must be strings")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence")
    return value


def _finite_vector(values: Sequence[float], *, field: str) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{field} must contain only finite values")
    return normalized


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_order_event_stream(
    events: Sequence[OrderEvent],
) -> tuple[OrderEvent, ...]:
    """Validate global order-event sequence and per-order economic identities."""

    normalized = tuple(events)
    if not normalized:
        raise ValueError("order event stream must not be empty")
    if tuple(event.sequence for event in normalized) != tuple(range(len(normalized))):
        raise ValueError("order event sequence must be contiguous from zero")

    histories: dict[str, list[OrderEvent]] = {}
    for event in normalized:
        histories.setdefault(event.order_id, []).append(event)
    for order_id, history in histories.items():
        first = history[0]
        if first.event_type != "submitted":
            raise ValueError(f"order {order_id} must begin with a submitted event")
        if first.previous_status is not OrderStatus.SUBMITTED:
            raise ValueError(f"order {order_id} submitted previous status is invalid")
        requested = first.requested_quantity
        cumulative_fill = 0.0
        previous: OrderEvent | None = None
        for event in history:
            if previous is not None:
                if previous.new_status.terminal:
                    raise ValueError(
                        f"order {order_id} has an event after terminal state"
                    )
                if event.previous_status is not previous.new_status:
                    raise ValueError(f"order {order_id} status chain is discontinuous")
                if event.processing_index < previous.processing_index:
                    raise ValueError(f"order {order_id} processing index regressed")
            for field, actual, expected in (
                ("dataset", event.dataset_id, first.dataset_id),
                (
                    "execution policy",
                    event.execution_policy_digest,
                    first.execution_policy_digest,
                ),
                ("symbol", event.symbol_index, first.symbol_index),
                ("replacement", event.replaced_order_id, first.replaced_order_id),
            ):
                if actual != expected:
                    raise ValueError(f"order {order_id} {field} identity changed")
            if not math.isclose(
                event.requested_quantity,
                requested,
                rel_tol=0.0,
                abs_tol=max(1e-12, 32 * math.ulp(abs(requested))),
            ):
                raise ValueError(f"order {order_id} requested quantity changed")
            try:
                allowed_previous, expected_new = _EVENT_TRANSITIONS[event.event_type]
            except KeyError as error:
                raise ValueError(
                    f"order {order_id} event type is unsupported"
                ) from error
            if event.previous_status not in allowed_previous:
                raise ValueError(f"order {order_id} event transition is invalid")
            if expected_new is None:
                if event.new_status is not event.previous_status:
                    raise ValueError(f"order {order_id} no-fill status changed")
            elif event.new_status is not expected_new:
                raise ValueError(f"order {order_id} event status is invalid")

            fill_event = event.event_type in {"partial_fill", "filled"}
            if fill_event:
                if abs(event.filled_quantity) <= 1e-12:
                    raise ValueError(f"order {order_id} fill quantity is zero")
                if math.copysign(1.0, event.filled_quantity) != math.copysign(
                    1.0, requested
                ):
                    raise ValueError(f"order {order_id} fill direction is invalid")
                if event.execution_price is None or event.filled_notional <= 0.0:
                    raise ValueError(f"order {order_id} fill economics are incomplete")
                expected_notional = abs(event.filled_quantity * event.execution_price)
                tolerance = max(1e-12, expected_notional * 1e-12)
                if not math.isclose(
                    event.filled_notional,
                    expected_notional,
                    rel_tol=0.0,
                    abs_tol=tolerance,
                ):
                    raise ValueError(f"order {order_id} fill notional is inconsistent")
                cumulative_fill += event.filled_quantity
            elif abs(event.filled_quantity) > 1e-12 or event.filled_notional != 0.0:
                raise ValueError(f"order {order_id} non-fill event contains a fill")

            expected_remaining = requested - cumulative_fill
            tolerance = max(
                1e-12,
                32 * math.ulp(abs(requested)),
                32 * math.ulp(abs(expected_remaining)),
            )
            if not math.isclose(
                event.remaining_quantity,
                expected_remaining,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError(f"order {order_id} remaining quantity is inconsistent")
            if abs(cumulative_fill) > abs(requested) + tolerance:
                raise ValueError(f"order {order_id} cumulative fill exceeds request")
            if (
                event.new_status is OrderStatus.FILLED
                and abs(expected_remaining) > tolerance
            ):
                raise ValueError(
                    f"order {order_id} filled status has remaining quantity"
                )
            if (
                event.new_status is OrderStatus.PARTIALLY_FILLED
                and abs(expected_remaining) <= tolerance
            ):
                raise ValueError(f"order {order_id} partial fill is actually complete")
            previous = event
    return normalized


@dataclass(frozen=True, slots=True)
class ExecutionReplayIdentity:
    """Exact candidate and evaluation cell that produced one replay."""

    candidate_config_digest: str
    evaluation_run_digest: str
    fold: int
    seed: int
    schema_version: str = EXECUTION_REPLAY_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(
            self.candidate_config_digest,
            field="replay_identity.candidate_config_digest",
        )
        require_sha256(
            self.evaluation_run_digest,
            field="replay_identity.evaluation_run_digest",
        )
        _non_negative_integer(self.fold, field="replay_identity.fold")
        _non_negative_integer(self.seed, field="replay_identity.seed")
        if self.schema_version != EXECUTION_REPLAY_IDENTITY_SCHEMA:
            raise ValueError("unsupported execution replay identity schema")

    def to_mapping(self) -> dict[str, object]:
        return {
            "candidate_config_digest": self.candidate_config_digest,
            "evaluation_run_digest": self.evaluation_run_digest,
            "fold": self.fold,
            "schema_version": self.schema_version,
            "seed": self.seed,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExecutionReplayIdentity:
        required = {
            "candidate_config_digest",
            "evaluation_run_digest",
            "fold",
            "schema_version",
            "seed",
        }
        if set(value) != required:
            raise ValueError("execution replay identity field closure mismatch")
        return cls(
            candidate_config_digest=_string(
                value["candidate_config_digest"],
                field="candidate_config_digest",
            ),
            evaluation_run_digest=_string(
                value["evaluation_run_digest"],
                field="evaluation_run_digest",
            ),
            fold=_non_negative_integer(value["fold"], field="fold"),
            seed=_non_negative_integer(value["seed"], field="seed"),
            schema_version=_string(value["schema_version"], field="schema_version"),
        )


@dataclass(frozen=True, slots=True)
class StatefulReplayEvidence:
    """Identity-bound digests for one deterministic execution replay."""

    dataset_id: str
    seed: int
    execution_policy_digest: str
    step_count: int
    order_event_count: int
    action_digest: str
    order_event_digest: str
    equity_curve_digest: str
    observation_trace_digest: str
    schema_version: str = REPLAY_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="replay.dataset_id")
        require_sha256(
            self.execution_policy_digest,
            field="replay.execution_policy_digest",
        )
        for field_name, integer_value in (
            ("seed", self.seed),
            ("step_count", self.step_count),
            ("order_event_count", self.order_event_count),
        ):
            _non_negative_integer(integer_value, field=field_name)
        for field_name, digest_value in (
            ("action_digest", self.action_digest),
            ("order_event_digest", self.order_event_digest),
            ("equity_curve_digest", self.equity_curve_digest),
            ("observation_trace_digest", self.observation_trace_digest),
        ):
            require_sha256(digest_value, field=f"replay.{field_name}")
        if self.schema_version != REPLAY_EVIDENCE_SCHEMA:
            raise ValueError("unsupported replay evidence schema")

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_digest": self.action_digest,
            "dataset_id": self.dataset_id,
            "equity_curve_digest": self.equity_curve_digest,
            "execution_policy_digest": self.execution_policy_digest,
            "observation_trace_digest": self.observation_trace_digest,
            "order_event_count": self.order_event_count,
            "order_event_digest": self.order_event_digest,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "step_count": self.step_count,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> StatefulReplayEvidence:
        required = {
            "action_digest",
            "dataset_id",
            "equity_curve_digest",
            "execution_policy_digest",
            "observation_trace_digest",
            "order_event_count",
            "order_event_digest",
            "schema_version",
            "seed",
            "step_count",
        }
        if set(value) != required:
            raise ValueError("stateful replay evidence field closure mismatch")
        return cls(
            dataset_id=_string(value["dataset_id"], field="dataset_id"),
            seed=_non_negative_integer(value["seed"], field="seed"),
            execution_policy_digest=_string(
                value["execution_policy_digest"],
                field="execution_policy_digest",
            ),
            step_count=_non_negative_integer(value["step_count"], field="step_count"),
            order_event_count=_non_negative_integer(
                value["order_event_count"], field="order_event_count"
            ),
            action_digest=_string(value["action_digest"], field="action_digest"),
            order_event_digest=_string(
                value["order_event_digest"], field="order_event_digest"
            ),
            equity_curve_digest=_string(
                value["equity_curve_digest"], field="equity_curve_digest"
            ),
            observation_trace_digest=_string(
                value["observation_trace_digest"],
                field="observation_trace_digest",
            ),
            schema_version=_string(value["schema_version"], field="schema_version"),
        )


def _book_payload(book: BookState) -> dict[str, object]:
    reason = book.termination_reason
    if isinstance(reason, EconomicTerminationReason):
        termination_reason: str | None = reason.value
    elif reason is None:
        termination_reason = None
    else:  # pragma: no cover - BookState normalizes this defensively
        termination_reason = str(reason)
    multipliers = book.contract_multipliers
    if multipliers is None:  # pragma: no cover - BookState normalizes this
        raise RuntimeError("book contract multipliers were not normalized")
    return {
        "borrow_cost": float(book.borrow_cost),
        "cash": float(book.cash),
        "contract_multipliers": multipliers.tolist(),
        "fill_count": book.fill_count,
        "funding_pnl": float(book.funding_pnl),
        "insolvent": book.insolvent,
        "maintenance_margin": float(book.maintenance_margin),
        "maintenance_requirement": float(book.maintenance_requirement),
        "margin_deficit": float(book.margin_deficit),
        "margin_used": float(book.margin_used),
        "mark_prices": book.mark_prices.tolist(),
        "max_drawdown": float(book.max_drawdown),
        "peak_value": float(book.peak_value),
        "quantities": book.quantities.tolist(),
        "rebalance_events": book.rebalance_events,
        "returns_history": tuple(float(value) for value in book.returns_history),
        "schema_version": _TERMINAL_BOOK_SCHEMA,
        "termination_reason": termination_reason,
        "total_cost": float(book.total_cost),
        "turnover_total": float(book.turnover_total),
    }


def _book_from_payload(value: Mapping[str, object]) -> BookState:
    required = {
        "borrow_cost",
        "cash",
        "contract_multipliers",
        "fill_count",
        "funding_pnl",
        "insolvent",
        "maintenance_margin",
        "maintenance_requirement",
        "margin_deficit",
        "margin_used",
        "mark_prices",
        "max_drawdown",
        "peak_value",
        "quantities",
        "rebalance_events",
        "returns_history",
        "schema_version",
        "termination_reason",
        "total_cost",
        "turnover_total",
    }
    if set(value) != required:
        raise ValueError("terminal book field closure mismatch")
    if value["schema_version"] != _TERMINAL_BOOK_SCHEMA:
        raise ValueError("terminal book schema identity mismatch")

    def vector(field: str) -> np.ndarray:
        raw = _sequence(value[field], field=field)
        return np.asarray(
            tuple(_number(item, field=f"{field}[]") for item in raw),
            dtype=np.float64,
        )

    raw_reason = value["termination_reason"]
    if raw_reason is not None and not isinstance(raw_reason, str):
        raise ValueError("termination_reason must be a string or null")
    book = BookState(
        quantities=vector("quantities"),
        cash=_number(value["cash"], field="cash"),
        mark_prices=vector("mark_prices"),
        peak_value=_number(value["peak_value"], field="peak_value"),
        contract_multipliers=vector("contract_multipliers"),
        max_drawdown=_number(value["max_drawdown"], field="max_drawdown"),
        turnover_total=_number(value["turnover_total"], field="turnover_total"),
        total_cost=_number(value["total_cost"], field="total_cost"),
        funding_pnl=_number(value["funding_pnl"], field="funding_pnl"),
        fill_count=_non_negative_integer(value["fill_count"], field="fill_count"),
        rebalance_events=_non_negative_integer(
            value["rebalance_events"], field="rebalance_events"
        ),
        returns_history=list(
            _finite_vector(
                cast(
                    Sequence[float],
                    _sequence(value["returns_history"], field="returns_history"),
                ),
                field="returns_history",
            )
        ),
        borrow_cost=_number(value["borrow_cost"], field="borrow_cost"),
        margin_used=_number(value["margin_used"], field="margin_used"),
        maintenance_margin=_number(
            value["maintenance_margin"], field="maintenance_margin"
        ),
        maintenance_requirement=_number(
            value["maintenance_requirement"], field="maintenance_requirement"
        ),
        margin_deficit=_number(value["margin_deficit"], field="margin_deficit"),
        insolvent=_boolean(value["insolvent"], field="insolvent"),
        termination_reason=raw_reason,
    )
    if to_json_value(_book_payload(book)) != to_json_value(dict(value)):
        raise ValueError("terminal book payload is not canonical")
    return book


def _pending_order_payload(order: PendingOrder) -> dict[str, object]:
    payload = to_json_value(order)
    if not isinstance(payload, dict):  # pragma: no cover - dataclass conversion
        raise TypeError("pending order did not serialize as an object")
    return cast(dict[str, object], payload)


def _order_book_payload(order_book: OrderBookState) -> dict[str, object]:
    return {
        "active_orders": tuple(
            _pending_order_payload(order) for order in order_book.active_orders
        ),
        "schema_version": _TERMINAL_ORDER_BOOK_SCHEMA,
        "terminal_orders": tuple(
            _pending_order_payload(order) for order in order_book.terminal_orders
        ),
    }


def _order_book_from_payload(value: Mapping[str, object]) -> OrderBookState:
    required = {"active_orders", "schema_version", "terminal_orders"}
    if set(value) != required:
        raise ValueError("terminal order book field closure mismatch")
    if value["schema_version"] != _TERMINAL_ORDER_BOOK_SCHEMA:
        raise ValueError("terminal order book schema identity mismatch")

    def orders(field: str) -> tuple[PendingOrder, ...]:
        raw = _sequence(value[field], field=field)
        return tuple(
            PendingOrder.from_mapping(_mapping(item, field=f"{field}[]"))
            for item in raw
        )

    order_book = OrderBookState(
        active_orders=orders("active_orders"),
        terminal_orders=orders("terminal_orders"),
    )
    if to_json_value(_order_book_payload(order_book)) != to_json_value(dict(value)):
        raise ValueError("terminal order book payload is not canonical")
    return order_book


def _validate_terminal_states(
    events: tuple[OrderEvent, ...],
    terminal_book: BookState,
    terminal_order_book: OrderBookState,
) -> None:
    fill_events = tuple(
        event for event in events if event.event_type in {"partial_fill", "filled"}
    )
    if terminal_book.fill_count != len(fill_events):
        raise ValueError("terminal book fill count does not match fill events")

    histories: dict[str, list[OrderEvent]] = {}
    for event in events:
        histories.setdefault(event.order_id, []).append(event)
    states = {
        order.order_id: order
        for order in (
            *terminal_order_book.active_orders,
            *terminal_order_book.terminal_orders,
        )
    }
    if set(states) != set(histories):
        raise ValueError("terminal order book does not match event stream orders")
    for order_id, history in histories.items():
        order = states[order_id]
        last = history[-1]
        if order.status is not last.new_status:
            raise ValueError("terminal order book status does not match event stream")
        if order.intent.dataset_id != last.dataset_id:
            raise ValueError("terminal order book dataset does not match event stream")
        if order.intent.execution_policy_digest != last.execution_policy_digest:
            raise ValueError("terminal order book policy does not match event stream")
        if order.intent.symbol_index != last.symbol_index:
            raise ValueError("terminal order book symbol does not match event stream")
        if order.intent.requested_quantity != last.requested_quantity:
            raise ValueError("terminal order book request does not match event stream")
        tolerance = max(1e-12, abs(last.requested_quantity) * 1e-12)
        if not math.isclose(
            order.remaining_quantity,
            last.remaining_quantity,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("terminal order book remaining quantity mismatch")
        expected_fill = sum(
            event.filled_quantity
            for event in history
            if event.event_type in {"partial_fill", "filled"}
        )
        expected_notional = sum(
            event.filled_notional
            for event in history
            if event.event_type in {"partial_fill", "filled"}
        )
        if not math.isclose(
            order.cumulative_filled_quantity,
            expected_fill,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("terminal order book cumulative fill mismatch")
        if not math.isclose(
            order.cumulative_filled_notional,
            expected_notional,
            rel_tol=0.0,
            abs_tol=max(1e-12, abs(expected_notional) * 1e-12),
        ):
            raise ValueError("terminal order book cumulative notional mismatch")
        expected_last_processed = None if len(history) == 1 else last.processing_index
        if order.last_processed_index != expected_last_processed:
            raise ValueError("terminal order book processing index mismatch")
        if order.evidence_version != len(history) - 1:
            raise ValueError("terminal order book evidence version mismatch")
        if order.terminal:
            if order.terminal_reason != last.reason:
                raise ValueError("terminal order book terminal reason mismatch")
        elif order.terminal_reason is not None:
            raise ValueError("active terminal order book state has a terminal reason")


def build_stateful_replay_evidence(
    *,
    dataset_id: str,
    seed: int,
    execution_policy_digest: str,
    actions: Sequence[Sequence[float]],
    order_events: Sequence[OrderEvent],
    equity_curve: Sequence[float],
    observation_digests: Sequence[str],
) -> StatefulReplayEvidence:
    """Build canonical replay evidence from one completed action trace."""

    require_sha256(dataset_id, field="dataset_id")
    require_sha256(execution_policy_digest, field="execution_policy_digest")
    _non_negative_integer(seed, field="seed")

    normalized_actions = tuple(
        _finite_vector(action, field=f"actions[{index}]")
        for index, action in enumerate(actions)
    )
    step_count = len(normalized_actions)
    normalized_equity = _finite_vector(equity_curve, field="equity_curve")
    if len(normalized_equity) != step_count + 1:
        raise ValueError("equity_curve must include initial equity and every step")
    if len(observation_digests) != step_count + 1:
        raise ValueError(
            "observation_digests must include initial observation and every step"
        )
    normalized_observations = tuple(observation_digests)
    for index, digest in enumerate(normalized_observations):
        require_sha256(digest, field=f"observation_digests[{index}]")

    normalized_events = validate_order_event_stream(order_events)
    event_payloads: list[dict[str, object]] = []
    for index, event in enumerate(normalized_events):
        if event.dataset_id != dataset_id:
            raise ValueError(f"order_events[{index}] dataset identity mismatch")
        if event.execution_policy_digest != execution_policy_digest:
            raise ValueError(
                f"order_events[{index}] execution policy identity mismatch"
            )
        event_payloads.append(event.canonical_payload())

    return StatefulReplayEvidence(
        dataset_id=dataset_id,
        seed=seed,
        execution_policy_digest=execution_policy_digest,
        step_count=step_count,
        order_event_count=len(event_payloads),
        action_digest=content_digest(normalized_actions),
        order_event_digest=content_digest(event_payloads),
        equity_curve_digest=content_digest(normalized_equity),
        observation_trace_digest=content_digest(normalized_observations),
    )


@dataclass(frozen=True, slots=True)
class ExecutionEventArtifact:
    """Canonical replay traces plus strict terminal accounting states."""

    dataset_id: str
    execution_policy_digest: str
    replay_identity: ExecutionReplayIdentity
    replay_evidence: StatefulReplayEvidence
    order_event_schema: str
    actions: tuple[tuple[float, ...], ...]
    observation_digests: tuple[str, ...]
    equity_curve: tuple[float, ...]
    events: tuple[OrderEvent, ...]
    terminal_book: dict[str, object]
    terminal_order_book: dict[str, object]
    schema_version: str = EXECUTION_EVENT_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="event_artifact.dataset_id")
        require_sha256(
            self.execution_policy_digest,
            field="event_artifact.execution_policy_digest",
        )
        if not isinstance(self.replay_identity, ExecutionReplayIdentity):
            raise ValueError("event artifact replay identity is invalid")
        if not isinstance(self.replay_evidence, StatefulReplayEvidence):
            raise ValueError("event artifact replay evidence is invalid")
        if not isinstance(self.order_event_schema, str) or not self.order_event_schema:
            raise ValueError("order_event_schema must be a non-empty string")
        actions = tuple(
            _finite_vector(action, field=f"actions[{index}]")
            for index, action in enumerate(self.actions)
        )
        observations = tuple(self.observation_digests)
        for index, digest in enumerate(observations):
            require_sha256(digest, field=f"observation_digests[{index}]")
        equity = _finite_vector(self.equity_curve, field="equity_curve")
        events = validate_order_event_stream(self.events)
        for index, event in enumerate(events):
            if event.schema_version != self.order_event_schema:
                raise ValueError(f"events[{index}] schema identity mismatch")
            if event.dataset_id != self.dataset_id:
                raise ValueError(f"events[{index}] dataset identity mismatch")
            if event.execution_policy_digest != self.execution_policy_digest:
                raise ValueError(f"events[{index}] execution policy identity mismatch")
        normalized_book = to_json_value(dict(self.terminal_book))
        normalized_order_book = to_json_value(dict(self.terminal_order_book))
        if not isinstance(normalized_book, dict) or not isinstance(
            normalized_order_book, dict
        ):  # pragma: no cover - both inputs are mappings
            raise TypeError("terminal execution states must serialize as objects")
        terminal_book_mapping = cast(dict[str, object], normalized_book)
        terminal_order_book_mapping = cast(dict[str, object], normalized_order_book)
        book = _book_from_payload(terminal_book_mapping)
        order_book = _order_book_from_payload(terminal_order_book_mapping)
        _validate_terminal_states(events, book, order_book)
        recomputed = build_stateful_replay_evidence(
            dataset_id=self.dataset_id,
            seed=self.replay_identity.seed,
            execution_policy_digest=self.execution_policy_digest,
            actions=actions,
            order_events=events,
            equity_curve=equity,
            observation_digests=observations,
        )
        if recomputed != self.replay_evidence:
            raise ValueError("execution replay evidence does not match embedded traces")
        if self.replay_evidence.seed != self.replay_identity.seed:
            raise ValueError("execution replay seed identity mismatch")
        if self.schema_version != EXECUTION_EVENT_ARTIFACT_SCHEMA:
            raise ValueError("unsupported execution event artifact schema")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "observation_digests", observations)
        object.__setattr__(self, "equity_curve", equity)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "terminal_book", terminal_book_mapping)
        object.__setattr__(self, "terminal_order_book", terminal_order_book_mapping)

    @property
    def order_event_count(self) -> int:
        return len(self.events)

    @property
    def terminal_book_digest(self) -> str:
        return content_digest(self.terminal_book)

    @property
    def terminal_order_book_digest(self) -> str:
        return content_digest(self.terminal_order_book)

    def to_mapping(self) -> dict[str, object]:
        return {
            "actions": self.actions,
            "dataset_id": self.dataset_id,
            "equity_curve": self.equity_curve,
            "events": tuple(event.canonical_payload() for event in self.events),
            "execution_policy_digest": self.execution_policy_digest,
            "observation_digests": self.observation_digests,
            "order_event_schema": self.order_event_schema,
            "replay_evidence": self.replay_evidence.to_mapping(),
            "replay_identity": self.replay_identity.to_mapping(),
            "schema_version": self.schema_version,
            "terminal_book": self.terminal_book,
            "terminal_order_book": self.terminal_order_book,
        }

    @property
    def raw_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping()) + b"\n"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExecutionEventArtifact:
        required = {
            "actions",
            "dataset_id",
            "equity_curve",
            "events",
            "execution_policy_digest",
            "observation_digests",
            "order_event_schema",
            "replay_evidence",
            "replay_identity",
            "schema_version",
            "terminal_book",
            "terminal_order_book",
        }
        if set(value) != required:
            raise ValueError("execution event artifact field closure mismatch")
        raw_events = _sequence(value["events"], field="events")
        events = tuple(
            OrderEvent.from_mapping(_mapping(item, field=f"events[{index}]"))
            for index, item in enumerate(raw_events)
        )
        raw_actions = _sequence(value["actions"], field="actions")
        actions = tuple(
            tuple(
                _number(number, field=f"actions[{index}][]")
                for number in _sequence(action, field=f"actions[{index}]")
            )
            for index, action in enumerate(raw_actions)
        )
        observations = tuple(
            _string(item, field="observation_digests[]")
            for item in _sequence(
                value["observation_digests"], field="observation_digests"
            )
        )
        equity = tuple(
            _number(item, field="equity_curve[]")
            for item in _sequence(value["equity_curve"], field="equity_curve")
        )
        return cls(
            dataset_id=_string(value["dataset_id"], field="dataset_id"),
            execution_policy_digest=_string(
                value["execution_policy_digest"],
                field="execution_policy_digest",
            ),
            replay_identity=ExecutionReplayIdentity.from_mapping(
                _mapping(value["replay_identity"], field="replay_identity")
            ),
            replay_evidence=StatefulReplayEvidence.from_mapping(
                _mapping(value["replay_evidence"], field="replay_evidence")
            ),
            order_event_schema=_string(
                value["order_event_schema"], field="order_event_schema"
            ),
            actions=actions,
            observation_digests=observations,
            equity_curve=equity,
            events=events,
            terminal_book=dict(_mapping(value["terminal_book"], field="terminal_book")),
            terminal_order_book=dict(
                _mapping(value["terminal_order_book"], field="terminal_order_book")
            ),
            schema_version=_string(value["schema_version"], field="schema_version"),
        )


def build_execution_event_artifact(
    *,
    candidate_config_digest: str,
    evaluation_run_digest: str,
    fold: int,
    seed: int,
    dataset_id: str,
    execution_policy_digest: str,
    actions: Sequence[Sequence[float]],
    observation_digests: Sequence[str],
    equity_curve: Sequence[float],
    order_events: Sequence[OrderEvent],
    terminal_book: BookState,
    terminal_order_book: OrderBookState,
) -> ExecutionEventArtifact:
    """Build one strict replay artifact from an actual completed execution."""

    events = validate_order_event_stream(order_events)
    schemas = {event.schema_version for event in events}
    if len(schemas) != 1:
        raise ValueError("execution event artifact requires one event schema")
    identity = ExecutionReplayIdentity(
        candidate_config_digest=candidate_config_digest,
        evaluation_run_digest=evaluation_run_digest,
        fold=fold,
        seed=seed,
    )
    replay = build_stateful_replay_evidence(
        dataset_id=dataset_id,
        seed=seed,
        execution_policy_digest=execution_policy_digest,
        actions=actions,
        order_events=events,
        equity_curve=equity_curve,
        observation_digests=observation_digests,
    )
    return ExecutionEventArtifact(
        dataset_id=dataset_id,
        execution_policy_digest=execution_policy_digest,
        replay_identity=identity,
        replay_evidence=replay,
        order_event_schema=next(iter(schemas)),
        actions=tuple(tuple(float(value) for value in action) for action in actions),
        observation_digests=tuple(observation_digests),
        equity_curve=tuple(float(value) for value in equity_curve),
        events=events,
        terminal_book=_book_payload(terminal_book),
        terminal_order_book=_order_book_payload(terminal_order_book),
    )


def _write_exclusive(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)
    return path


def write_execution_event_artifact(
    path: str | Path,
    artifact: ExecutionEventArtifact,
) -> Path:
    """Write one immutable replay artifact to an explicitly selected path."""

    return _write_exclusive(Path(path), artifact.raw_bytes)


def write_execution_event_artifact_content_addressed(
    root: str | Path,
    artifact: ExecutionEventArtifact,
) -> Path:
    """Publish replay bytes by their SHA-256 identity, idempotently."""

    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{artifact.digest}{_CONTENT_ADDRESSED_SUFFIX}"
    try:
        return _write_exclusive(output, artifact.raw_bytes)
    except FileExistsError:
        with open_regular_binary(output, field="execution replay artifact") as handle:
            existing = handle.read()
        if existing != artifact.raw_bytes:
            raise FileExistsError(
                "content-addressed execution replay path contains different bytes"
            ) from None
        return output


def load_execution_event_artifact_bytes(raw: bytes) -> ExecutionEventArtifact:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("execution event artifact must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("execution event artifact must be an object")
    artifact = ExecutionEventArtifact.from_mapping(value)
    if raw != artifact.raw_bytes:
        raise ValueError("execution event artifact must use canonical encoding")
    return artifact


def load_execution_event_artifact(path: str | Path) -> ExecutionEventArtifact:
    with open_regular_binary(Path(path), field="execution replay artifact") as handle:
        return load_execution_event_artifact_bytes(handle.read())


__all__ = [
    "EXECUTION_EVENT_ARTIFACT_FILE_NAME",
    "EXECUTION_EVENT_ARTIFACT_SCHEMA",
    "EXECUTION_REPLAY_IDENTITY_SCHEMA",
    "REPLAY_EVIDENCE_SCHEMA",
    "ExecutionEventArtifact",
    "ExecutionReplayIdentity",
    "StatefulReplayEvidence",
    "build_execution_event_artifact",
    "build_stateful_replay_evidence",
    "load_execution_event_artifact",
    "load_execution_event_artifact_bytes",
    "validate_order_event_stream",
    "write_execution_event_artifact",
    "write_execution_event_artifact_content_addressed",
]
