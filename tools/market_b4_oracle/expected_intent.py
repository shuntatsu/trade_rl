"""Independent two-symbol expected order-intent arithmetic.

This deliberately consumes caller-owned state and does not import the native
execution stack. It models only MARKET/GTC intent sizing and pending replacement.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

_TOLERANCE: Final[float] = 1e-12
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVE_STATUSES = frozenset(
    {"submitted", "latency_wait", "eligible", "triggered", "partially_filled"}
)


def exact_quantity(value: float) -> Fraction:
    """Reproduce the source quantity boundary without importing native code."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("quantity must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("quantity must be finite")
    return Fraction(str(converted))


def project_quantity(value: Fraction) -> float:
    """Project a rational toward zero until its reported float is conservative."""
    if type(value) is not Fraction:
        raise ValueError("quantity must be Fraction")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("quantity is outside finite float range") from error
    if not math.isfinite(result):
        raise ValueError("quantity is outside finite float range")
    while abs(exact_quantity(result)) > abs(value):
        result = math.nextafter(result, 0.0)
    return result


def canonical_json_bytes(value: object) -> bytes:
    """Canonical JSON used only for the closed native OrderIntent identity shape."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("identity payload is not finite canonical JSON") from error
    return text.encode("utf-8")


def _finite_number(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _native_identity_float(
    name: str, value: object, *, positive: bool = False
) -> float:
    """Keep identity inputs at the native pre-normalization float boundary."""
    if type(value) is not float:
        raise ValueError(f"{name} must be a float")
    return _finite_number(name, value, positive=positive)


def _pair_floats(
    name: str, values: object, *, positive: bool = False
) -> tuple[float, float]:
    if type(values) is not tuple or len(values) != 2:
        raise ValueError(f"{name} must be a two-item tuple")
    first = _finite_number(f"{name}[0]", values[0], positive=positive)
    second = _finite_number(f"{name}[1]", values[1], positive=positive)
    return first, second


def _nonnegative_index(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ExpectedIntent:
    order_id: str
    dataset_id: str
    target_identity: str
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
    replaced_order_id: str | None = None
    reduce_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if not isinstance(self.target_identity, str) or not self.target_identity:
            raise ValueError("target_identity must be non-empty")
        _digest("execution_policy_digest", self.execution_policy_digest)
        _digest("order_id", self.order_id)
        if (
            isinstance(self.symbol_index, bool)
            or not isinstance(self.symbol_index, int)
            or self.symbol_index < 0
        ):
            raise ValueError("symbol_index must be a non-negative integer")
        quantity = _native_identity_float("requested_quantity", self.requested_quantity)
        if abs(quantity) <= _TOLERANCE:
            raise ValueError("requested_quantity must be finite and non-zero")
        reference = _native_identity_float(
            "submission_reference_price", self.submission_reference_price, positive=True
        )
        equity = _native_identity_float(
            "decision_equity", self.decision_equity, positive=True
        )
        submit = _nonnegative_index("submit_index", self.submit_index)
        eligible = _nonnegative_index("eligible_index", self.eligible_index)
        if eligible < submit:
            raise ValueError("eligible_index must not precede submit_index")
        if self.expiry_index is not None:
            expiry = _nonnegative_index("expiry_index", self.expiry_index)
            if expiry < eligible:
                raise ValueError("expiry_index must not precede eligible_index")
        if self.replaced_order_id is not None:
            _digest("replaced_order_id", self.replaced_order_id)
        if self.order_type != "market" or self.time_in_force != "gtc":
            raise ValueError("expected-intent module supports only MARKET/GTC intents")
        if self.limit_price is not None or self.stop_price is not None:
            raise ValueError("MARKET intent cannot carry limit or stop price")
        if type(self.reduce_only) is not bool:
            raise ValueError("reduce_only must be boolean")
        if self.reduce_only and self.order_type != "market":
            raise ValueError("reduce_only requires MARKET")
        object.__setattr__(self, "requested_quantity", quantity)
        object.__setattr__(self, "submission_reference_price", reference)
        object.__setattr__(self, "decision_equity", equity)
        if self.recompute_order_id() != self.order_id:
            raise ValueError("expected intent identity digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        target_identity: str,
        execution_policy_digest: str,
        symbol_index: int,
        requested_quantity: float,
        order_type: str,
        time_in_force: str,
        limit_price: float | None,
        stop_price: float | None,
        submit_index: int,
        eligible_index: int,
        expiry_index: int | None,
        submission_reference_price: float,
        decision_equity: float,
        replaced_order_id: str | None = None,
        reduce_only: bool = False,
    ) -> ExpectedIntent:
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if not isinstance(target_identity, str) or not target_identity:
            raise ValueError("target_identity must be non-empty")
        policy = _digest("execution_policy_digest", execution_policy_digest)
        if (
            isinstance(symbol_index, bool)
            or not isinstance(symbol_index, int)
            or symbol_index < 0
        ):
            raise ValueError("symbol_index must be a non-negative integer")
        quantity = _native_identity_float("requested_quantity", requested_quantity)
        if abs(quantity) <= _TOLERANCE:
            raise ValueError("requested_quantity must be finite and non-zero")
        if order_type != "market" or time_in_force != "gtc":
            raise ValueError("expected-intent module supports only MARKET/GTC intents")
        if limit_price is not None or stop_price is not None:
            raise ValueError("MARKET intent cannot carry limit or stop price")
        submit = _nonnegative_index("submit_index", submit_index)
        eligible = _nonnegative_index("eligible_index", eligible_index)
        if eligible < submit:
            raise ValueError("eligible_index must not precede submit_index")
        if expiry_index is not None:
            expiry = _nonnegative_index("expiry_index", expiry_index)
            if expiry < eligible:
                raise ValueError("expiry_index must not precede eligible_index")
        else:
            expiry = None
        reference = _native_identity_float(
            "submission_reference_price", submission_reference_price, positive=True
        )
        equity = _native_identity_float(
            "decision_equity", decision_equity, positive=True
        )
        if replaced_order_id is not None:
            _digest("replaced_order_id", replaced_order_id)
        if type(reduce_only) is not bool:
            raise ValueError("reduce_only must be boolean")
        identity: dict[str, object] = {
            "dataset_id": dataset_id,
            "decision_equity": equity,
            "eligible_index": eligible,
            "execution_policy_digest": policy,
            "expiry_index": expiry,
            "limit_price": None,
            "order_type": "market",
            "replaced_order_id": replaced_order_id,
            "requested_quantity": quantity,
            "schema_version": "order_intent_v1",
            "stop_price": None,
            "submission_reference_price": reference,
            "submit_index": submit,
            "symbol_index": symbol_index,
            "target_identity": target_identity,
            "time_in_force": "gtc",
        }
        if reduce_only:
            identity["reduce_only"] = True
        order_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        return cls(
            order_id=order_id,
            dataset_id=dataset_id,
            target_identity=target_identity,
            execution_policy_digest=policy,
            symbol_index=symbol_index,
            requested_quantity=quantity,
            order_type="market",
            time_in_force="gtc",
            limit_price=None,
            stop_price=None,
            submit_index=submit,
            eligible_index=eligible,
            expiry_index=expiry,
            submission_reference_price=reference,
            decision_equity=equity,
            replaced_order_id=replaced_order_id,
            reduce_only=reduce_only,
        )

    def identity_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dataset_id": self.dataset_id,
            "decision_equity": self.decision_equity,
            "eligible_index": self.eligible_index,
            "execution_policy_digest": self.execution_policy_digest,
            "expiry_index": self.expiry_index,
            "limit_price": self.limit_price,
            "order_type": self.order_type,
            "replaced_order_id": self.replaced_order_id,
            "requested_quantity": self.requested_quantity,
            "schema_version": "order_intent_v1",
            "stop_price": self.stop_price,
            "submission_reference_price": self.submission_reference_price,
            "submit_index": self.submit_index,
            "symbol_index": self.symbol_index,
            "target_identity": self.target_identity,
            "time_in_force": self.time_in_force,
        }
        if self.reduce_only:
            payload["reduce_only"] = True
        return payload

    def canonical_payload(self) -> dict[str, object]:
        """Match the native public payload, distinct from the ID preimage."""
        payload: dict[str, object] = {
            "order_id": self.order_id,
            "dataset_id": self.dataset_id,
            "target_identity": self.target_identity,
            "execution_policy_digest": self.execution_policy_digest,
            "symbol_index": self.symbol_index,
            "requested_quantity": self.requested_quantity,
            "order_type": self.order_type,
            "time_in_force": self.time_in_force,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "submit_index": self.submit_index,
            "eligible_index": self.eligible_index,
            "expiry_index": self.expiry_index,
            "submission_reference_price": self.submission_reference_price,
            "decision_equity": self.decision_equity,
            "replaced_order_id": self.replaced_order_id,
            "reduce_only": self.reduce_only,
        }
        if not self.reduce_only:
            payload.pop("reduce_only")
        return payload

    def fresh_payload(self) -> dict[str, object]:
        """Return a newly allocated identity mapping, never the internal state."""
        return dict(self.identity_payload())

    def recompute_order_id(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.identity_payload())).hexdigest()


@dataclass(frozen=True, slots=True)
class ExpectedPending:
    intent: ExpectedIntent
    exact_filled_quantity: Fraction = Fraction(0)
    status: str = "submitted"

    def __post_init__(self) -> None:
        if not isinstance(self.intent, ExpectedIntent):
            raise ValueError("pending intent must be ExpectedIntent")
        if type(self.exact_filled_quantity) is not Fraction:
            raise ValueError("exact_filled_quantity must be Fraction")
        if not isinstance(self.status, str) or self.status not in _ACTIVE_STATUSES:
            raise ValueError("pending order must have an active supported status")
        requested = exact_quantity(self.intent.requested_quantity)
        remaining = requested - self.exact_filled_quantity
        if (
            requested
            and self.exact_filled_quantity
            and requested * self.exact_filled_quantity < 0
        ):
            raise ValueError("exact fill quantity has the wrong direction")
        if abs(self.exact_filled_quantity) > abs(requested):
            raise ValueError("exact fill quantity exceeds requested quantity")
        if not remaining:
            raise ValueError("a fully filled order cannot remain active")

    @classmethod
    def from_intent(
        cls,
        intent: ExpectedIntent,
        *,
        exact_filled_quantity: Fraction = Fraction(0),
        status: str = "submitted",
    ) -> ExpectedPending:
        return cls(intent, exact_filled_quantity, status)

    @property
    def order_id(self) -> str:
        return self.intent.order_id

    @property
    def remaining_exact_quantity(self) -> Fraction:
        return (
            exact_quantity(self.intent.requested_quantity) - self.exact_filled_quantity
        )

    @property
    def remaining_quantity(self) -> float:
        return project_quantity(self.remaining_exact_quantity)


@dataclass(frozen=True, slots=True)
class ExpectedCancellation:
    order: ExpectedPending
    reason: str

    @property
    def order_id(self) -> str:
        return self.order.order_id


@dataclass(frozen=True, slots=True)
class ExpectedTransitionResult:
    new_intents: tuple[ExpectedIntent, ...]
    kept_orders: tuple[ExpectedPending, ...]
    cancelled_orders: tuple[ExpectedCancellation, ...]
    desired_quantities: tuple[float, float] | None
    residual_quantities: tuple[float, float]


def reconcile_expected(
    *,
    exact_quantities: tuple[Fraction, Fraction],
    current_weights: tuple[float, float],
    final_risk_weights: tuple[float, float],
    decision_equity: float,
    valuation_prices: tuple[float, float],
    multipliers: tuple[float, float],
    reference_prices: tuple[float, float],
    dataset_id: str,
    target_identity: str | None,
    execution_policy_digest: str,
    submit_index: int,
    latency_bars: int,
    pending_orders: tuple[ExpectedPending, ...],
) -> ExpectedTransitionResult:
    """Reconcile caller-owned state; this confers no admission or B4 authority."""
    if (
        type(exact_quantities) is not tuple
        or len(exact_quantities) != 2
        or any(type(value) is not Fraction for value in exact_quantities)
    ):
        raise ValueError("exact_quantities must be a pair of Fractions")
    current = _pair_floats("current_weights", current_weights)
    final = _pair_floats("final_risk_weights", final_risk_weights)
    valuations = _pair_floats("valuation_prices", valuation_prices, positive=True)
    multipliers_pair = _pair_floats("multipliers", multipliers, positive=True)
    references = _pair_floats("reference_prices", reference_prices, positive=True)
    equity = _native_identity_float("decision_equity", decision_equity, positive=True)
    index = _nonnegative_index("submit_index", submit_index)
    latency = _nonnegative_index("latency_bars", latency_bars)
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be non-empty")
    policy = _digest("execution_policy_digest", execution_policy_digest)
    hold = current == final
    if hold:
        if target_identity is not None and (
            not isinstance(target_identity, str) or not target_identity
        ):
            raise ValueError("target_identity must be non-empty or null on hold")
    elif not isinstance(target_identity, str) or not target_identity:
        raise ValueError("target_identity must be non-empty outside hold")
    if type(pending_orders) is not tuple or any(
        not isinstance(order, ExpectedPending) for order in pending_orders
    ):
        raise ValueError("pending_orders must be a tuple of ExpectedPending")
    ids = tuple(order.order_id for order in pending_orders)
    if len(set(ids)) != len(ids):
        raise ValueError("active pending order IDs must be unique")
    for order in pending_orders:
        if order.intent.symbol_index not in (0, 1):
            raise ValueError(
                "pending order symbol_index is outside the two-symbol scope"
            )

    if hold:
        return ExpectedTransitionResult(
            new_intents=(),
            kept_orders=(),
            cancelled_orders=tuple(
                ExpectedCancellation(order, "quantity_hold") for order in pending_orders
            ),
            desired_quantities=None,
            residual_quantities=(0.0, 0.0),
        )

    assert target_identity is not None
    gross = abs(final[0]) + abs(final[1])
    if gross > 1.0 + _TOLERANCE:
        raise ValueError("target gross exceeds frozen maximum_gross")
    desired = (
        final[0] * equity / (valuations[0] * multipliers_pair[0]),
        final[1] * equity / (valuations[1] * multipliers_pair[1]),
    )
    if not all(math.isfinite(value) for value in desired):
        raise ValueError("desired quantity is non-finite")
    wanted = (exact_quantity(desired[0]), exact_quantity(desired[1]))
    reported_current = (
        project_quantity(exact_quantities[0]),
        project_quantity(exact_quantities[1]),
    )
    desired_rows: list[ExpectedIntent] = []
    kept_ids: set[str] = set()
    cancelled: list[ExpectedCancellation] = []
    residuals = [0.0, 0.0]

    for symbol_index in range(2):
        current_exact = exact_quantities[symbol_index]
        target_delta = desired[symbol_index] - reported_current[symbol_index]
        legacy_flatten = bool(current_exact and wanted[symbol_index] == 0)
        reduce_only = legacy_flatten
        if reduce_only:
            target_delta = project_quantity(wanted[symbol_index] - current_exact)
        active = tuple(
            order
            for order in pending_orders
            if order.intent.symbol_index == symbol_index
        )
        active_residual = float(sum(order.remaining_quantity for order in active))
        compatible = all(
            order.intent.dataset_id == dataset_id
            and order.intent.execution_policy_digest == policy
            and order.intent.reduce_only == reduce_only
            for order in active
        )
        if compatible and math.isclose(
            target_delta, active_residual, rel_tol=0.0, abs_tol=_TOLERANCE
        ):
            kept_ids.update(order.order_id for order in active)
            continue

        replaced_order_id: str | None = None
        if active:
            ordered_active = tuple(
                sorted(
                    active,
                    key=lambda order: (
                        order.intent.eligible_index,
                        order.order_id,
                    ),
                )
            )
            replaced_order_id = ordered_active[0].order_id
            cancelled.extend(
                ExpectedCancellation(order, "superseded") for order in ordered_active
            )
        if abs(target_delta) <= _TOLERANCE:
            continue
        residuals[symbol_index] = target_delta
        desired_rows.append(
            ExpectedIntent.create(
                dataset_id=dataset_id,
                target_identity=target_identity,
                execution_policy_digest=policy,
                symbol_index=symbol_index,
                requested_quantity=target_delta,
                order_type="market",
                time_in_force="gtc",
                limit_price=None,
                stop_price=None,
                submit_index=index,
                eligible_index=index + latency,
                expiry_index=None,
                submission_reference_price=references[symbol_index],
                decision_equity=equity,
                replaced_order_id=replaced_order_id,
                reduce_only=reduce_only,
            )
        )
    return ExpectedTransitionResult(
        new_intents=tuple(desired_rows),
        kept_orders=tuple(
            order for order in pending_orders if order.order_id in kept_ids
        ),
        cancelled_orders=tuple(cancelled),
        desired_quantities=desired,
        residual_quantities=(residuals[0], residuals[1]),
    )
