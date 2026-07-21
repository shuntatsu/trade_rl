"""Persistent order-domain types and canonical execution evidence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Mapping

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes

_QUANTITY_TOLERANCE = 1e-12


class OrderDomainError(ValueError):
    """Raised when an order violates a domain invariant."""


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"


class TimeInForce(StrEnum):
    IOC = "ioc"
    DAY = "day"
    GTC = "gtc"


class OrderStatus(StrEnum):
    SUBMITTED = "submitted"
    LATENCY_WAIT = "latency_wait"
    ELIGIBLE = "eligible"
    TRIGGERED = "triggered"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.CANCELLED,
        }


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _validate_positive(name: str, value: float) -> None:
    if not _is_finite(value) or value <= 0.0:
        raise OrderDomainError(f"{name} must be finite and positive")


def _validate_digest(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise OrderDomainError(f"{name} must be a lowercase SHA-256 digest")


def execution_policy_digest(payload: Mapping[str, object]) -> str:
    """Return a stable SHA-256 digest for a canonical execution policy payload."""

    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()


@dataclass(frozen=True, slots=True)
class OrderIntent:
    order_id: str
    dataset_id: str
    target_identity: str
    execution_policy_digest: str
    symbol_index: int
    requested_quantity: float
    order_type: OrderType
    time_in_force: TimeInForce
    limit_price: float | None
    stop_price: float | None
    submit_index: int
    eligible_index: int
    expiry_index: int | None
    submission_reference_price: float
    decision_equity: float
    replaced_order_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        target_identity: str,
        execution_policy_digest: str,
        symbol_index: int,
        requested_quantity: float,
        order_type: OrderType,
        time_in_force: TimeInForce,
        limit_price: float | None,
        stop_price: float | None,
        submit_index: int,
        eligible_index: int,
        expiry_index: int | None,
        submission_reference_price: float,
        decision_equity: float,
        replaced_order_id: str | None = None,
    ) -> OrderIntent:
        if not dataset_id:
            raise OrderDomainError("dataset_id must be non-empty")
        if not target_identity:
            raise OrderDomainError("target_identity must be non-empty")
        _validate_digest("execution_policy_digest", execution_policy_digest)
        if isinstance(symbol_index, bool) or not isinstance(symbol_index, int) or symbol_index < 0:
            raise OrderDomainError("symbol_index must be a non-negative integer")
        if not _is_finite(requested_quantity) or abs(requested_quantity) <= _QUANTITY_TOLERANCE:
            raise OrderDomainError("requested_quantity must be finite and non-zero")
        if isinstance(submit_index, bool) or not isinstance(submit_index, int) or submit_index < 0:
            raise OrderDomainError("submit_index must be a non-negative integer")
        if (
            isinstance(eligible_index, bool)
            or not isinstance(eligible_index, int)
            or eligible_index < submit_index
        ):
            raise OrderDomainError("eligible_index must not precede submit_index")
        if expiry_index is not None and (
            isinstance(expiry_index, bool)
            or not isinstance(expiry_index, int)
            or expiry_index < eligible_index
        ):
            raise OrderDomainError("expiry_index must not precede eligible_index")
        if time_in_force is TimeInForce.DAY and expiry_index is None:
            raise OrderDomainError("day orders require an expiry_index")
        _validate_positive("submission_reference_price", submission_reference_price)
        _validate_positive("decision_equity", decision_equity)

        if order_type is OrderType.MARKET:
            if limit_price is not None or stop_price is not None:
                raise OrderDomainError("market orders may not define limit or stop prices")
        elif order_type is OrderType.LIMIT:
            if limit_price is None:
                raise OrderDomainError("limit_price is required for limit orders")
            _validate_positive("limit_price", limit_price)
            if stop_price is not None:
                raise OrderDomainError("limit orders may not define stop_price")
        elif order_type is OrderType.STOP_MARKET:
            if stop_price is None:
                raise OrderDomainError("stop_price is required for stop-market orders")
            _validate_positive("stop_price", stop_price)
            if limit_price is not None:
                raise OrderDomainError("stop-market orders may not define limit_price")
        else:  # pragma: no cover - StrEnum type makes this defensive
            raise OrderDomainError(f"unsupported order type: {order_type}")

        if replaced_order_id is not None:
            _validate_digest("replaced_order_id", replaced_order_id)

        identity_payload: dict[str, object] = {
            "dataset_id": dataset_id,
            "decision_equity": decision_equity,
            "eligible_index": eligible_index,
            "execution_policy_digest": execution_policy_digest,
            "expiry_index": expiry_index,
            "limit_price": limit_price,
            "order_type": order_type.value,
            "replaced_order_id": replaced_order_id,
            "requested_quantity": requested_quantity,
            "schema_version": "order_intent_v1",
            "stop_price": stop_price,
            "submission_reference_price": submission_reference_price,
            "submit_index": submit_index,
            "symbol_index": symbol_index,
            "target_identity": target_identity,
            "time_in_force": time_in_force.value,
        }
        order_id = hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
        return cls(
            order_id=order_id,
            dataset_id=dataset_id,
            target_identity=target_identity,
            execution_policy_digest=execution_policy_digest,
            symbol_index=symbol_index,
            requested_quantity=float(requested_quantity),
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=None if limit_price is None else float(limit_price),
            stop_price=None if stop_price is None else float(stop_price),
            submit_index=submit_index,
            eligible_index=eligible_index,
            expiry_index=expiry_index,
            submission_reference_price=float(submission_reference_price),
            decision_equity=float(decision_equity),
            replaced_order_id=replaced_order_id,
        )


@dataclass(frozen=True, slots=True)
class PendingOrder:
    intent: OrderIntent
    remaining_quantity: float
    cumulative_filled_quantity: float = 0.0
    cumulative_filled_notional: float = 0.0
    status: OrderStatus = OrderStatus.SUBMITTED
    trigger_index: int | None = None
    last_processed_index: int | None = None
    terminal_reason: str | None = None
    evidence_version: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("remaining_quantity", self.remaining_quantity),
            ("cumulative_filled_quantity", self.cumulative_filled_quantity),
            ("cumulative_filled_notional", self.cumulative_filled_notional),
        ):
            if not _is_finite(value):
                raise OrderDomainError(f"{name} must be finite")
        expected = self.cumulative_filled_quantity + self.remaining_quantity
        if not math.isclose(
            expected,
            self.intent.requested_quantity,
            rel_tol=0.0,
            abs_tol=_QUANTITY_TOLERANCE,
        ):
            raise OrderDomainError("requested quantity identity is inconsistent")
        requested_sign = math.copysign(1.0, self.intent.requested_quantity)
        for name, value in (
            ("remaining_quantity", self.remaining_quantity),
            ("cumulative_filled_quantity", self.cumulative_filled_quantity),
        ):
            if abs(value) > _QUANTITY_TOLERANCE and math.copysign(1.0, value) != requested_sign:
                raise OrderDomainError(f"{name} has the wrong direction")
        if self.cumulative_filled_notional < 0.0:
            raise OrderDomainError("cumulative_filled_notional must be non-negative")
        if self.evidence_version < 0:
            raise OrderDomainError("evidence_version must be non-negative")
        if self.status.terminal and not self.terminal_reason:
            raise OrderDomainError("terminal orders require a terminal_reason")
        if not self.status.terminal and self.terminal_reason is not None:
            raise OrderDomainError("active orders may not have a terminal_reason")

    @classmethod
    def from_intent(cls, intent: OrderIntent) -> PendingOrder:
        return cls(intent=intent, remaining_quantity=intent.requested_quantity)

    @property
    def order_id(self) -> str:
        return self.intent.order_id

    @property
    def terminal(self) -> bool:
        return self.status.terminal

    def _ensure_active(self) -> None:
        if self.terminal:
            raise OrderDomainError("terminal order state cannot be mutated")

    def _validate_processing_index(self, processing_index: int) -> None:
        if isinstance(processing_index, bool) or not isinstance(processing_index, int):
            raise OrderDomainError("processing_index must be an integer")
        if processing_index < self.intent.submit_index:
            raise OrderDomainError("processing_index precedes order submission")
        if self.last_processed_index is not None and processing_index < self.last_processed_index:
            raise OrderDomainError("processing_index must be monotonic")

    def _transition(
        self,
        *,
        status: OrderStatus,
        processing_index: int,
        terminal_reason: str | None = None,
        trigger_index: int | None = None,
    ) -> PendingOrder:
        self._ensure_active()
        self._validate_processing_index(processing_index)
        if status.terminal and not terminal_reason:
            raise OrderDomainError("terminal transitions require a reason")
        if not status.terminal and terminal_reason is not None:
            raise OrderDomainError("active transitions may not define a terminal reason")
        return replace(
            self,
            status=status,
            trigger_index=self.trigger_index if trigger_index is None else trigger_index,
            last_processed_index=processing_index,
            terminal_reason=terminal_reason,
            evidence_version=self.evidence_version + 1,
        )

    def mark_latency_wait(self, *, processing_index: int) -> PendingOrder:
        if self.status not in {OrderStatus.SUBMITTED, OrderStatus.LATENCY_WAIT}:
            raise OrderDomainError("invalid state transition to latency_wait")
        return self._transition(status=OrderStatus.LATENCY_WAIT, processing_index=processing_index)

    def mark_eligible(self, *, processing_index: int) -> PendingOrder:
        if processing_index < self.intent.eligible_index:
            raise OrderDomainError("order is not eligible at this processing_index")
        if self.status not in {
            OrderStatus.SUBMITTED,
            OrderStatus.LATENCY_WAIT,
            OrderStatus.ELIGIBLE,
        }:
            raise OrderDomainError("invalid state transition to eligible")
        return self._transition(status=OrderStatus.ELIGIBLE, processing_index=processing_index)

    def mark_triggered(self, *, processing_index: int) -> PendingOrder:
        if self.status not in {
            OrderStatus.ELIGIBLE,
            OrderStatus.TRIGGERED,
            OrderStatus.PARTIALLY_FILLED,
        }:
            raise OrderDomainError("invalid state transition to triggered")
        trigger_index = self.trigger_index if self.trigger_index is not None else processing_index
        return self._transition(
            status=OrderStatus.TRIGGERED,
            processing_index=processing_index,
            trigger_index=trigger_index,
        )

    def apply_fill(
        self,
        *,
        quantity: float,
        notional: float,
        processing_index: int,
    ) -> PendingOrder:
        self._ensure_active()
        self._validate_processing_index(processing_index)
        if not _is_finite(quantity) or abs(quantity) <= _QUANTITY_TOLERANCE:
            raise OrderDomainError("fill quantity must be finite and non-zero")
        if math.copysign(1.0, quantity) != math.copysign(1.0, self.remaining_quantity):
            raise OrderDomainError("fill quantity direction does not match remaining quantity")
        if abs(quantity) > abs(self.remaining_quantity) + _QUANTITY_TOLERANCE:
            raise OrderDomainError("fill quantity exceeds remaining quantity")
        if not _is_finite(notional) or notional < 0.0:
            raise OrderDomainError("fill notional must be finite and non-negative")

        cumulative_quantity = self.cumulative_filled_quantity + quantity
        remaining = self.intent.requested_quantity - cumulative_quantity
        if abs(remaining) <= _QUANTITY_TOLERANCE:
            remaining = 0.0
            status = OrderStatus.FILLED
            terminal_reason = "filled"
        else:
            status = OrderStatus.PARTIALLY_FILLED
            terminal_reason = None
        return replace(
            self,
            remaining_quantity=remaining,
            cumulative_filled_quantity=cumulative_quantity,
            cumulative_filled_notional=self.cumulative_filled_notional + notional,
            status=status,
            last_processed_index=processing_index,
            terminal_reason=terminal_reason,
            evidence_version=self.evidence_version + 1,
        )

    def reject(self, *, processing_index: int, reason: str) -> PendingOrder:
        return self._transition(
            status=OrderStatus.REJECTED,
            processing_index=processing_index,
            terminal_reason=reason,
        )

    def expire(self, *, processing_index: int, reason: str) -> PendingOrder:
        return self._transition(
            status=OrderStatus.EXPIRED,
            processing_index=processing_index,
            terminal_reason=reason,
        )

    def cancel(self, *, processing_index: int, reason: str) -> PendingOrder:
        return self._transition(
            status=OrderStatus.CANCELLED,
            processing_index=processing_index,
            terminal_reason=reason,
        )


@dataclass(frozen=True, slots=True)
class OrderBookState:
    active_orders: tuple[PendingOrder, ...]
    terminal_orders: tuple[PendingOrder, ...]

    def __post_init__(self) -> None:
        all_orders = self.active_orders + self.terminal_orders
        order_ids = tuple(order.order_id for order in all_orders)
        if len(order_ids) != len(set(order_ids)):
            raise OrderDomainError("duplicate order IDs are not allowed")
        if any(order.terminal for order in self.active_orders):
            raise OrderDomainError("active_orders may not contain terminal orders")
        if any(not order.terminal for order in self.terminal_orders):
            raise OrderDomainError("terminal_orders must contain only terminal orders")

    @classmethod
    def empty(cls) -> OrderBookState:
        return cls(active_orders=(), terminal_orders=())

    def active_for_symbol(self, symbol_index: int) -> tuple[PendingOrder, ...]:
        return tuple(
            order for order in self.active_orders if order.intent.symbol_index == symbol_index
        )

    def active_remaining_quantities(self, n_symbols: int) -> np.ndarray:
        if isinstance(n_symbols, bool) or not isinstance(n_symbols, int) or n_symbols <= 0:
            raise OrderDomainError("n_symbols must be a positive integer")
        values = np.zeros(n_symbols, dtype=np.float64)
        for order in self.active_orders:
            if order.intent.symbol_index >= n_symbols:
                raise OrderDomainError("active order symbol is outside n_symbols")
            values[order.intent.symbol_index] += order.remaining_quantity
        return values

    def add(self, *orders: PendingOrder) -> OrderBookState:
        if any(order.terminal for order in orders):
            raise OrderDomainError("cannot add terminal orders as active")
        return OrderBookState(
            active_orders=self.active_orders + tuple(orders),
            terminal_orders=self.terminal_orders,
        )

    def replace(self, updated: PendingOrder) -> OrderBookState:
        matching = [index for index, order in enumerate(self.active_orders) if order.order_id == updated.order_id]
        if not matching:
            raise OrderDomainError("cannot replace unknown active order")
        index = matching[0]
        active = list(self.active_orders)
        active.pop(index)
        terminal = self.terminal_orders
        if updated.terminal:
            terminal += (updated,)
        else:
            active.insert(index, updated)
        return OrderBookState(active_orders=tuple(active), terminal_orders=terminal)


@dataclass(frozen=True, slots=True)
class OrderEvent:
    schema_version: str
    sequence: int
    order_id: str
    replaced_order_id: str | None
    dataset_id: str
    execution_policy_digest: str
    symbol_index: int
    event_type: str
    processing_index: int
    timestamp_ns: int
    previous_status: OrderStatus
    new_status: OrderStatus
    requested_quantity: float
    remaining_quantity: float
    filled_quantity: float
    execution_price: float | None
    filled_notional: float
    capacity_before: float
    capacity_after: float
    participation_rate: float
    trigger_segment: str | None
    available_volume_fraction: float
    reason: str | None
    path_mode: str
    path_points: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise OrderDomainError("schema_version must be non-empty")
        if self.sequence < 0:
            raise OrderDomainError("event sequence must be non-negative")
        _validate_digest("order_id", self.order_id)
        _validate_digest("execution_policy_digest", self.execution_policy_digest)
        if self.replaced_order_id is not None:
            _validate_digest("replaced_order_id", self.replaced_order_id)
        for name, value in (
            ("requested_quantity", self.requested_quantity),
            ("remaining_quantity", self.remaining_quantity),
            ("filled_quantity", self.filled_quantity),
            ("filled_notional", self.filled_notional),
            ("capacity_before", self.capacity_before),
            ("capacity_after", self.capacity_after),
            ("participation_rate", self.participation_rate),
            ("available_volume_fraction", self.available_volume_fraction),
        ):
            if not _is_finite(value):
                raise OrderDomainError(f"{name} must be finite")
        if self.execution_price is not None and not _is_finite(self.execution_price):
            raise OrderDomainError("execution_price must be finite when present")
        if not all(_is_finite(point) for point in self.path_points):
            raise OrderDomainError("path_points must be finite")

    def canonical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["previous_status"] = self.previous_status.value
        payload["new_status"] = self.new_status.value
        payload["path_points"] = list(self.path_points)
        return payload
