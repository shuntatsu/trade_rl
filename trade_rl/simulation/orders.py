"""Persistent order-domain types and canonical execution evidence."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Mapping, overload

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes

_QUANTITY_TOLERANCE = 1e-12
_QUANTITY_ULPS = 32


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


def _quantity_tolerance(*values: float) -> float:
    """Resolve an absolute quantity tolerance that survives lot arithmetic."""

    return max(
        _QUANTITY_TOLERANCE,
        *(
            _QUANTITY_ULPS * math.ulp(abs(float(value)))
            for value in values
            if _is_finite(value)
        ),
    )


def _validate_positive(name: str, value: float) -> None:
    if not _is_finite(value) or value <= 0.0:
        raise OrderDomainError(f"{name} must be finite and positive")


def _validate_digest(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
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
        if (
            isinstance(symbol_index, bool)
            or not isinstance(symbol_index, int)
            or symbol_index < 0
        ):
            raise OrderDomainError("symbol_index must be a non-negative integer")
        if (
            not _is_finite(requested_quantity)
            or abs(requested_quantity) <= _QUANTITY_TOLERANCE
        ):
            raise OrderDomainError("requested_quantity must be finite and non-zero")
        if (
            isinstance(submit_index, bool)
            or not isinstance(submit_index, int)
            or submit_index < 0
        ):
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
                raise OrderDomainError(
                    "market orders may not define limit or stop prices"
                )
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

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> OrderIntent:
        """Reconstruct and re-derive one canonical order intent identity."""

        required = {
            "decision_equity",
            "dataset_id",
            "eligible_index",
            "execution_policy_digest",
            "expiry_index",
            "limit_price",
            "order_id",
            "order_type",
            "replaced_order_id",
            "requested_quantity",
            "stop_price",
            "submission_reference_price",
            "submit_index",
            "symbol_index",
            "target_identity",
            "time_in_force",
        }
        if set(value) != required:
            raise OrderDomainError("order intent field closure mismatch")

        def string(field: str) -> str:
            raw = value[field]
            if not isinstance(raw, str):
                raise OrderDomainError(f"{field} must be a string")
            return raw

        def optional_string(field: str) -> str | None:
            raw = value[field]
            if raw is None:
                return None
            if not isinstance(raw, str):
                raise OrderDomainError(f"{field} must be a string or null")
            return raw

        def integer(field: str) -> int:
            raw = value[field]
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise OrderDomainError(f"{field} must be an integer")
            return raw

        def optional_integer(field: str) -> int | None:
            raw = value[field]
            if raw is None:
                return None
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise OrderDomainError(f"{field} must be an integer or null")
            return raw

        def number(field: str) -> float:
            raw = value[field]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise OrderDomainError(f"{field} must be numeric")
            return float(raw)

        def optional_number(field: str) -> float | None:
            raw = value[field]
            if raw is None:
                return None
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise OrderDomainError(f"{field} must be numeric or null")
            return float(raw)

        try:
            order_type = OrderType(string("order_type"))
            time_in_force = TimeInForce(string("time_in_force"))
        except ValueError as error:
            raise OrderDomainError("order intent enum value is unsupported") from error
        restored = cls.create(
            dataset_id=string("dataset_id"),
            target_identity=string("target_identity"),
            execution_policy_digest=string("execution_policy_digest"),
            symbol_index=integer("symbol_index"),
            requested_quantity=number("requested_quantity"),
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=optional_number("limit_price"),
            stop_price=optional_number("stop_price"),
            submit_index=integer("submit_index"),
            eligible_index=integer("eligible_index"),
            expiry_index=optional_integer("expiry_index"),
            submission_reference_price=number("submission_reference_price"),
            decision_equity=number("decision_equity"),
            replaced_order_id=optional_string("replaced_order_id"),
        )
        if restored.order_id != string("order_id"):
            raise OrderDomainError("order intent identity digest mismatch")
        return restored


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
        identity_tolerance = _quantity_tolerance(
            self.intent.requested_quantity,
            self.cumulative_filled_quantity,
            self.remaining_quantity,
            expected,
        )
        if not math.isclose(
            expected,
            self.intent.requested_quantity,
            rel_tol=0.0,
            abs_tol=identity_tolerance,
        ):
            raise OrderDomainError("requested quantity identity is inconsistent")
        requested_sign = math.copysign(1.0, self.intent.requested_quantity)
        for name, value in (
            ("remaining_quantity", self.remaining_quantity),
            ("cumulative_filled_quantity", self.cumulative_filled_quantity),
        ):
            if (
                abs(value) > _QUANTITY_TOLERANCE
                and math.copysign(1.0, value) != requested_sign
            ):
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

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PendingOrder:
        """Reconstruct one canonical pending-order state from untrusted JSON."""

        required = {
            "cumulative_filled_notional",
            "cumulative_filled_quantity",
            "evidence_version",
            "intent",
            "last_processed_index",
            "remaining_quantity",
            "status",
            "terminal_reason",
            "trigger_index",
        }
        if set(value) != required:
            raise OrderDomainError("pending order field closure mismatch")
        raw_intent = value["intent"]
        if not isinstance(raw_intent, Mapping):
            raise OrderDomainError("pending order intent must be an object")

        def number(field: str) -> float:
            raw = value[field]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise OrderDomainError(f"{field} must be numeric")
            return float(raw)

        def integer(field: str) -> int:
            raw = value[field]
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise OrderDomainError(f"{field} must be an integer")
            return raw

        def optional_integer(field: str) -> int | None:
            raw = value[field]
            if raw is None:
                return None
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise OrderDomainError(f"{field} must be an integer or null")
            return raw

        raw_reason = value["terminal_reason"]
        if raw_reason is not None and not isinstance(raw_reason, str):
            raise OrderDomainError("terminal_reason must be a string or null")
        raw_status = value["status"]
        if not isinstance(raw_status, str):
            raise OrderDomainError("status must be a string")
        try:
            status = OrderStatus(raw_status)
        except ValueError as error:
            raise OrderDomainError("pending order status is unsupported") from error
        return cls(
            intent=OrderIntent.from_mapping(raw_intent),
            remaining_quantity=number("remaining_quantity"),
            cumulative_filled_quantity=number("cumulative_filled_quantity"),
            cumulative_filled_notional=number("cumulative_filled_notional"),
            status=status,
            trigger_index=optional_integer("trigger_index"),
            last_processed_index=optional_integer("last_processed_index"),
            terminal_reason=raw_reason,
            evidence_version=integer("evidence_version"),
        )

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
        if (
            self.last_processed_index is not None
            and processing_index < self.last_processed_index
        ):
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
            raise OrderDomainError(
                "active transitions may not define a terminal reason"
            )
        return replace(
            self,
            status=status,
            trigger_index=self.trigger_index
            if trigger_index is None
            else trigger_index,
            last_processed_index=processing_index,
            terminal_reason=terminal_reason,
            evidence_version=self.evidence_version + 1,
        )

    def mark_latency_wait(self, *, processing_index: int) -> PendingOrder:
        if self.status not in {OrderStatus.SUBMITTED, OrderStatus.LATENCY_WAIT}:
            raise OrderDomainError("invalid state transition to latency_wait")
        return self._transition(
            status=OrderStatus.LATENCY_WAIT, processing_index=processing_index
        )

    def mark_eligible(self, *, processing_index: int) -> PendingOrder:
        if processing_index < self.intent.eligible_index:
            raise OrderDomainError("order is not eligible at this processing_index")
        if self.status not in {
            OrderStatus.SUBMITTED,
            OrderStatus.LATENCY_WAIT,
            OrderStatus.ELIGIBLE,
        }:
            raise OrderDomainError("invalid state transition to eligible")
        return self._transition(
            status=OrderStatus.ELIGIBLE, processing_index=processing_index
        )

    def mark_triggered(self, *, processing_index: int) -> PendingOrder:
        if self.status not in {
            OrderStatus.ELIGIBLE,
            OrderStatus.TRIGGERED,
            OrderStatus.PARTIALLY_FILLED,
        }:
            raise OrderDomainError("invalid state transition to triggered")
        trigger_index = (
            self.trigger_index if self.trigger_index is not None else processing_index
        )
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
            raise OrderDomainError(
                "fill quantity direction does not match remaining quantity"
            )
        fill_tolerance = _quantity_tolerance(
            self.intent.requested_quantity,
            self.remaining_quantity,
            self.cumulative_filled_quantity,
            quantity,
        )
        if abs(quantity) > abs(self.remaining_quantity) + fill_tolerance:
            raise OrderDomainError("fill quantity exceeds remaining quantity")
        if not _is_finite(notional) or notional < 0.0:
            raise OrderDomainError("fill notional must be finite and non-negative")

        cumulative_quantity = self.cumulative_filled_quantity + quantity
        remaining = self.intent.requested_quantity - cumulative_quantity
        completion_tolerance = _quantity_tolerance(
            self.intent.requested_quantity,
            cumulative_quantity,
            remaining,
        )
        if abs(remaining) <= completion_tolerance:
            cumulative_quantity = self.intent.requested_quantity
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
class _TerminalOrderNode:
    order: PendingOrder
    previous: _TerminalOrderNode | None
    length: int
    max_submit_index: int
    order_ids_at_max_submit: frozenset[str]


class TerminalOrderArchive(Sequence[PendingOrder]):
    """Persistent chronological archive with constant-time tail appends.

    Execution produces many immutable ``OrderBookState`` snapshots. Copying a
    tuple containing every completed order for each snapshot makes an episode
    quadratic. A linked archive preserves every terminal order and historical
    snapshot while sharing the unchanged prefix.
    """

    __slots__ = ("_tail",)

    def __init__(self, orders: Sequence[PendingOrder] = ()) -> None:
        tail: _TerminalOrderNode | None = None
        for order in orders:
            tail = self._next_node(tail, order)
        self._tail = tail

    @staticmethod
    def _next_node(
        previous: _TerminalOrderNode | None,
        order: PendingOrder,
    ) -> _TerminalOrderNode:
        submit_index = order.intent.submit_index
        previous_max = -1 if previous is None else previous.max_submit_index
        if submit_index > previous_max:
            max_submit_index = submit_index
            ids = frozenset((order.order_id,))
        elif submit_index == previous_max:
            max_submit_index = previous_max
            previous_ids = (
                frozenset() if previous is None else previous.order_ids_at_max_submit
            )
            ids = previous_ids | frozenset((order.order_id,))
        else:
            max_submit_index = previous_max
            ids = frozenset() if previous is None else previous.order_ids_at_max_submit
        return _TerminalOrderNode(
            order=order,
            previous=previous,
            length=1 if previous is None else previous.length + 1,
            max_submit_index=max_submit_index,
            order_ids_at_max_submit=ids,
        )

    @classmethod
    def _from_tail(cls, tail: _TerminalOrderNode) -> TerminalOrderArchive:
        archive = object.__new__(cls)
        archive._tail = tail
        return archive

    def append(self, order: PendingOrder) -> TerminalOrderArchive:
        return self._from_tail(self._next_node(self._tail, order))

    def canonical_payload(self) -> tuple[PendingOrder, ...]:
        """Expose the historical tuple shape at serialization boundaries."""

        return tuple(self)

    @property
    def max_submit_index(self) -> int:
        return -1 if self._tail is None else self._tail.max_submit_index

    @property
    def order_ids_at_max_submit(self) -> frozenset[str]:
        return frozenset() if self._tail is None else self._tail.order_ids_at_max_submit

    def __len__(self) -> int:
        return 0 if self._tail is None else self._tail.length

    def __iter__(self) -> Iterator[PendingOrder]:
        reverse: list[PendingOrder] = []
        node = self._tail
        while node is not None:
            reverse.append(node.order)
            node = node.previous
        yield from reversed(reverse)

    @overload
    def __getitem__(self, index: int) -> PendingOrder: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[PendingOrder, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> PendingOrder | tuple[PendingOrder, ...]:
        if isinstance(index, slice):
            return tuple(self)[index]
        size = len(self)
        resolved = index + size if index < 0 else index
        if not 0 <= resolved < size:
            raise IndexError("terminal order archive index out of range")
        steps = size - 1 - resolved
        node = self._tail
        for _ in range(steps):
            assert node is not None
            node = node.previous
        assert node is not None
        return node.order

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"TerminalOrderArchive({tuple(self)!r})"


@dataclass(frozen=True, slots=True)
class OrderBookState:
    active_orders: tuple[PendingOrder, ...]
    terminal_orders: Sequence[PendingOrder]

    def __post_init__(self) -> None:
        archive = (
            self.terminal_orders
            if isinstance(self.terminal_orders, TerminalOrderArchive)
            else TerminalOrderArchive(self.terminal_orders)
        )
        all_orders = self.active_orders + tuple(archive)
        order_ids = tuple(order.order_id for order in all_orders)
        if len(order_ids) != len(set(order_ids)):
            raise OrderDomainError("duplicate order IDs are not allowed")
        if any(order.terminal for order in self.active_orders):
            raise OrderDomainError("active_orders may not contain terminal orders")
        if any(not order.terminal for order in archive):
            raise OrderDomainError("terminal_orders must contain only terminal orders")
        object.__setattr__(self, "terminal_orders", archive)

    @classmethod
    def _from_validated(
        cls,
        *,
        active_orders: tuple[PendingOrder, ...],
        terminal_orders: TerminalOrderArchive,
    ) -> OrderBookState:
        state = object.__new__(cls)
        object.__setattr__(state, "active_orders", active_orders)
        object.__setattr__(state, "terminal_orders", terminal_orders)
        return state

    @classmethod
    def empty(cls) -> OrderBookState:
        return cls(active_orders=(), terminal_orders=())

    def active_for_symbol(self, symbol_index: int) -> tuple[PendingOrder, ...]:
        return tuple(
            order
            for order in self.active_orders
            if order.intent.symbol_index == symbol_index
        )

    def active_remaining_quantities(self, n_symbols: int) -> np.ndarray:
        if (
            isinstance(n_symbols, bool)
            or not isinstance(n_symbols, int)
            or n_symbols <= 0
        ):
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

    def _add_generated(self, *orders: PendingOrder) -> OrderBookState:
        """Add canonical runtime intents without rescanning the full archive."""

        if any(order.terminal for order in orders):
            raise OrderDomainError("cannot add terminal orders as active")
        archive = self.terminal_orders
        if not isinstance(archive, TerminalOrderArchive):  # pragma: no cover
            raise RuntimeError("terminal order archive was not normalized")
        existing_active = {order.order_id for order in self.active_orders}
        generated_ids = tuple(order.order_id for order in orders)
        if len(generated_ids) != len(set(generated_ids)) or any(
            order_id in existing_active for order_id in generated_ids
        ):
            raise OrderDomainError("duplicate order IDs are not allowed")
        latest_submit = max(
            (
                archive.max_submit_index,
                *(order.intent.submit_index for order in self.active_orders),
            )
        )
        for order in orders:
            submit_index = order.intent.submit_index
            if submit_index < latest_submit or (
                submit_index == archive.max_submit_index
                and order.order_id in archive.order_ids_at_max_submit
            ):
                raise OrderDomainError("generated order identity is not monotonic")
        return self._from_validated(
            active_orders=self.active_orders + tuple(orders),
            terminal_orders=archive,
        )

    def replace(self, updated: PendingOrder) -> OrderBookState:
        matching = [
            index
            for index, order in enumerate(self.active_orders)
            if order.order_id == updated.order_id
        ]
        if not matching:
            raise OrderDomainError("cannot replace unknown active order")
        index = matching[0]
        active = list(self.active_orders)
        active.pop(index)
        terminal = self.terminal_orders
        if not isinstance(terminal, TerminalOrderArchive):  # pragma: no cover
            raise RuntimeError("terminal order archive was not normalized")
        if updated.terminal:
            terminal = terminal.append(updated)
        else:
            active.insert(index, updated)
        return self._from_validated(
            active_orders=tuple(active),
            terminal_orders=terminal,
        )


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
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise OrderDomainError("event sequence must be non-negative")
        if not self.dataset_id:
            raise OrderDomainError("dataset_id must be non-empty")
        _validate_digest("order_id", self.order_id)
        _validate_digest("execution_policy_digest", self.execution_policy_digest)
        if self.replaced_order_id is not None:
            _validate_digest("replaced_order_id", self.replaced_order_id)
        for name, value in (
            ("symbol_index", self.symbol_index),
            ("processing_index", self.processing_index),
            ("timestamp_ns", self.timestamp_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise OrderDomainError(f"{name} must be a non-negative integer")
        if self.event_type not in {
            "submitted",
            "latency_wait",
            "eligible",
            "triggered",
            "no_fill",
            "partial_fill",
            "filled",
            "rejected",
            "expired",
            "cancelled",
        }:
            raise OrderDomainError("event_type is unsupported")
        if not isinstance(self.previous_status, OrderStatus) or not isinstance(
            self.new_status, OrderStatus
        ):
            raise OrderDomainError("event statuses must be OrderStatus values")
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
        if abs(self.requested_quantity) <= _QUANTITY_TOLERANCE:
            raise OrderDomainError("requested_quantity must be non-zero")
        if self.filled_notional < 0.0:
            raise OrderDomainError("filled_notional must be non-negative")
        if self.capacity_before < 0.0 or self.capacity_after < 0.0:
            raise OrderDomainError("event capacities must be non-negative")
        if not 0.0 <= self.participation_rate <= 1.0:
            raise OrderDomainError("participation_rate must be within [0, 1]")
        if not 0.0 <= self.available_volume_fraction <= 1.0:
            raise OrderDomainError(
                "available_volume_fraction must be within [0, 1]"
            )
        if self.execution_price is not None and not _is_finite(self.execution_price):
            raise OrderDomainError("execution_price must be finite when present")
        if self.execution_price is not None and self.execution_price <= 0.0:
            raise OrderDomainError("execution_price must be positive when present")
        if not all(_is_finite(point) for point in self.path_points):
            raise OrderDomainError("path_points must be finite")
        if self.path_points and (
            len(self.path_points) != 4 or any(point <= 0.0 for point in self.path_points)
        ):
            raise OrderDomainError("path_points must be four positive prices")
        if self.path_mode not in {"optimistic", "neutral", "conservative"}:
            raise OrderDomainError("path_mode is unsupported")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> OrderEvent:
        """Reconstruct one exact canonical event from an untrusted mapping."""

        required = {
            "available_volume_fraction",
            "capacity_after",
            "capacity_before",
            "dataset_id",
            "event_type",
            "execution_policy_digest",
            "execution_price",
            "filled_notional",
            "filled_quantity",
            "new_status",
            "order_id",
            "participation_rate",
            "path_mode",
            "path_points",
            "previous_status",
            "processing_index",
            "reason",
            "remaining_quantity",
            "replaced_order_id",
            "requested_quantity",
            "schema_version",
            "sequence",
            "symbol_index",
            "timestamp_ns",
            "trigger_segment",
        }
        if set(value) != required:
            raise OrderDomainError("order event field closure mismatch")

        def string(field: str) -> str:
            raw = value[field]
            if not isinstance(raw, str):
                raise OrderDomainError(f"{field} must be a string")
            return raw

        def optional_string(field: str) -> str | None:
            raw = value[field]
            if raw is None:
                return None
            if not isinstance(raw, str):
                raise OrderDomainError(f"{field} must be a string or null")
            return raw

        def integer(field: str) -> int:
            raw = value[field]
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise OrderDomainError(f"{field} must be an integer")
            return raw

        def number(field: str) -> float:
            raw = value[field]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise OrderDomainError(f"{field} must be numeric")
            return float(raw)

        raw_execution_price = value["execution_price"]
        if raw_execution_price is not None and (
            isinstance(raw_execution_price, bool)
            or not isinstance(raw_execution_price, (int, float))
        ):
            raise OrderDomainError("execution_price must be numeric or null")
        raw_path_points = value["path_points"]
        if not isinstance(raw_path_points, list) or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in raw_path_points
        ):
            raise OrderDomainError("path_points must be a numeric list")
        try:
            previous_status = OrderStatus(string("previous_status"))
            new_status = OrderStatus(string("new_status"))
        except ValueError as error:
            raise OrderDomainError("event status is unsupported") from error
        return cls(
            schema_version=string("schema_version"),
            sequence=integer("sequence"),
            order_id=string("order_id"),
            replaced_order_id=optional_string("replaced_order_id"),
            dataset_id=string("dataset_id"),
            execution_policy_digest=string("execution_policy_digest"),
            symbol_index=integer("symbol_index"),
            event_type=string("event_type"),
            processing_index=integer("processing_index"),
            timestamp_ns=integer("timestamp_ns"),
            previous_status=previous_status,
            new_status=new_status,
            requested_quantity=number("requested_quantity"),
            remaining_quantity=number("remaining_quantity"),
            filled_quantity=number("filled_quantity"),
            execution_price=(
                None
                if raw_execution_price is None
                else float(raw_execution_price)
            ),
            filled_notional=number("filled_notional"),
            capacity_before=number("capacity_before"),
            capacity_after=number("capacity_after"),
            participation_rate=number("participation_rate"),
            trigger_segment=optional_string("trigger_segment"),
            available_volume_fraction=number("available_volume_fraction"),
            reason=optional_string("reason"),
            path_mode=string("path_mode"),
            path_points=tuple(float(item) for item in raw_path_points),
        )

    def canonical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["previous_status"] = self.previous_status.value
        payload["new_status"] = self.new_status.value
        payload["path_points"] = list(self.path_points)
        return payload
