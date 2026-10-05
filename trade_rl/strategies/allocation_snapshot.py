"""Immutable evidence of one live independent account before a decision.

These facts are not a numeric policy observation or a terminal-account reader.
There is no cash reservation or position-age model in this schema.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Any, cast

import numpy as np

from trade_rl._validation import require_non_empty, require_sha256
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.artifacts.canonical import freeze_json_value
from trade_rl.artifacts.hashing import content_digest

_BOOK_FIELDS = frozenset(
    "quantities cash mark_prices peak_value contract_multipliers max_drawdown "
    "turnover_total total_cost funding_pnl fill_count rebalance_events "
    "returns_history borrow_cost margin_used maintenance_margin "
    "maintenance_requirement margin_deficit insolvent termination_reason "
    "as_of_index as_of_dataset_id exact_quantities equity current_drawdown".split()
)
_ORDER_FIELDS = frozenset(
    "intent remaining_quantity cumulative_filled_quantity cumulative_filled_notional "
    "status trigger_index last_processed_index terminal_reason evidence_version "
    "exact_cumulative_filled_quantity exact_requested_quantity exact_remaining_quantity".split()
)
_INTENT_FIELDS = frozenset(
    "order_id dataset_id target_identity execution_policy_digest symbol_index "
    "requested_quantity order_type time_in_force limit_price stop_price submit_index "
    "eligible_index expiry_index submission_reference_price decision_equity "
    "replaced_order_id reduce_only".split()
)


def _index(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _time(value: str) -> np.datetime64:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{9}", value
        )
        is None
    ):
        raise ValueError("snapshot time must be a canonical nanosecond string")
    try:
        time = np.datetime64(value, "ns")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "snapshot time must be a canonical nanosecond string"
        ) from error
    if np.isnat(time) or np.datetime_as_string(time, unit="ns") != value:
        raise ValueError("snapshot time must be a canonical nanosecond string")
    return time


def _fraction(value: object) -> Fraction:
    if not isinstance(value, str):
        raise ValueError("snapshot quantity must be a canonical rational string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(
            "snapshot quantity must be a canonical rational string"
        ) from error
    if str(result) != value:
        raise ValueError("snapshot quantity must be a canonical rational string")
    return result


def _number(value: object, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(
            "snapshot financial value must be finite and appropriately signed"
        )
    return float(value)


def _reporting_quantity(exact: Fraction, reported: object) -> None:
    value = _number(reported)
    represented = Fraction(str(value))
    if represented * exact < 0 or abs(represented) > abs(exact):
        raise ValueError(
            "snapshot reporting quantity must conservatively retain its direction"
        )
    try:
        projected = float(exact)
    except OverflowError as error:
        raise ValueError(
            "snapshot exact quantity is outside its reporting range"
        ) from error
    if not math.isfinite(projected) or abs(projected - value) > math.ulp(projected):
        raise ValueError("snapshot exact quantity differs from its reporting view")


def _book_shape(book: Mapping[str, Any]) -> None:
    _index(book["as_of_index"], "book processing index")
    _number(book["cash"])
    for name in ("equity", "peak_value"):
        _number(book[name], positive=True)
    for name in (
        "quantities",
        "exact_quantities",
        "mark_prices",
        "contract_multipliers",
    ):
        if not isinstance(book[name], tuple) or len(book[name]) != 1:
            raise ValueError("snapshot vectors must contain only the selected symbol")
    _reporting_quantity(_fraction(book["exact_quantities"][0]), book["quantities"][0])
    for name in ("mark_prices", "contract_multipliers"):
        _number(book[name][0], positive=True)


def _order_shape(order: Mapping[str, Any], decision_index: int) -> None:
    intent = order.get("intent")
    if (
        set(order) != _ORDER_FIELDS
        or not isinstance(intent, Mapping)
        or set(intent) != _INTENT_FIELDS
    ):
        raise ValueError("snapshot order facts differ from the declared nested schema")
    if (
        intent["order_type"] != "market"
        or intent["time_in_force"] not in ("ioc", "day", "gtc")
        or order["status"]
        not in (
            "submitted",
            "latency_wait",
            "eligible",
            "triggered",
            "partially_filled",
        )
        or order["terminal_reason"] is not None
        or not isinstance(intent["reduce_only"], bool)
        or intent["limit_price"] is not None
        or intent["stop_price"] is not None
        or (intent["time_in_force"] == "day" and intent["expiry_index"] is None)
    ):
        raise ValueError("snapshot supports only active native MARKET order facts")
    _index(order["evidence_version"], "evidence_version")
    for name in ("submission_reference_price", "decision_equity"):
        _number(intent[name], positive=True)
    if _number(order["cumulative_filled_notional"]) < 0:
        raise ValueError("snapshot filled notional must be nonnegative")
    for name in ("submit_index", "eligible_index"):
        _index(intent[name], name)
    if intent["expiry_index"] is not None:
        _index(intent["expiry_index"], "expiry_index")
    for name in ("trigger_index", "last_processed_index"):
        if order[name] is not None:
            _index(order[name], name)
            if order[name] > decision_index:
                raise ValueError("snapshot order transition is ahead of the decision")
            if order[name] < intent["submit_index"]:
                raise ValueError("snapshot order transition precedes submission")
    last = order["last_processed_index"]
    trigger = order["trigger_index"]
    if (order["status"] == "submitted" and last is not None) or (
        order["status"] == "latency_wait" and last is None
    ):
        raise ValueError("snapshot waiting status differs from its transition progress")
    if order["status"] in ("eligible", "triggered", "partially_filled") and (
        last is None or last < intent["eligible_index"]
    ):
        raise ValueError("snapshot order transition precedes eligibility")
    if trigger is not None and (
        last is None or trigger > last or trigger < intent["eligible_index"]
    ):
        raise ValueError("snapshot trigger differs from its transition clock")
    if order["status"] == "triggered" and trigger is None:
        raise ValueError("snapshot triggered order requires a trigger transition")
    if (
        intent["submit_index"] > decision_index
        or intent["eligible_index"] < intent["submit_index"]
        or intent["eligible_index"] > decision_index + 1
        or (
            intent["expiry_index"] is not None
            and intent["expiry_index"] < decision_index + 1
        )
    ):
        raise ValueError("snapshot order clock differs from the current decision")
    requested = _fraction(order["exact_requested_quantity"])
    cumulative = _fraction(order["exact_cumulative_filled_quantity"])
    remaining = _fraction(order["exact_remaining_quantity"])
    if (
        requested != Fraction(str(_number(intent["requested_quantity"])))
        or requested != cumulative + remaining
        or requested == 0
        or any(
            value * requested < 0 or abs(value) > abs(requested)
            for value in (cumulative, remaining)
        )
    ):
        raise ValueError("snapshot exact order quantities do not reconcile")
    if (
        (
            order["status"] in ("submitted", "latency_wait", "eligible")
            and (cumulative != 0 or trigger is not None)
        )
        or remaining == 0
        or (order["status"] == "partially_filled" and cumulative == 0)
    ):
        raise ValueError(
            "snapshot active status differs from its exact filled progress"
        )
    _reporting_quantity(cumulative, order["cumulative_filled_quantity"])
    _reporting_quantity(remaining, order["remaining_quantity"])


def _facts(value: Mapping[str, object]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("snapshot facts must be a mapping")
    return cast(Mapping[str, Any], freeze_json_value(value))


@dataclass(frozen=True, slots=True)
class AllocationAccountSnapshot:
    """Detached canonical facts; evaluation owns their source validation.

    ``book_facts`` vectors contain only the selected global symbol slot.
    ``source_state_digest`` retains the existing complete context identity,
    including terminal order history. The separate digest binds this projection.
    Order IDs remain evidence, and no numeric encoding is provided here.
    Structural consistency does not establish account or source authenticity.
    """

    account_id: str
    dataset_id: str
    symbol: str
    symbol_index: int
    decision_index: int
    decision_time: str
    available_at: str
    execution_policy_digest: str
    risk_digest: str
    source_state_digest: str
    observable_tradable: bool
    book_facts: Mapping[str, object]
    active_orders: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        for name in ("account_id", "symbol"):
            require_non_empty(getattr(self, name), field=name)
        for name in (
            "dataset_id",
            "execution_policy_digest",
            "risk_digest",
            "source_state_digest",
        ):
            require_sha256(getattr(self, name), field=name)
        for name in ("symbol_index", "decision_index"):
            _index(getattr(self, name), name)
        if _time(self.available_at) > _time(self.decision_time):
            raise ValueError("snapshot source is unavailable at the decision")
        if not isinstance(self.observable_tradable, bool):
            raise ValueError("observable tradability must be boolean")
        book = _facts(self.book_facts)
        if set(book) != _BOOK_FIELDS:
            raise ValueError("snapshot book facts differ from the declared schema")
        _book_shape(book)
        if (
            book["as_of_index"] != self.decision_index
            or book["as_of_dataset_id"] != self.dataset_id
        ):
            raise ValueError("snapshot book processing clock differs from its source")
        if book["insolvent"] is not False or book["termination_reason"] is not None:
            raise ValueError(
                "snapshot supports only live, solvent predecision accounts"
            )
        if not isinstance(self.active_orders, tuple):
            raise ValueError("snapshot orders must be an immutable tuple")
        orders = tuple(_facts(order) for order in self.active_orders)
        for order in orders:
            _order_shape(order, self.decision_index)
            intent = order["intent"]
            if (
                intent["dataset_id"] != self.dataset_id
                or intent["symbol_index"] != self.symbol_index
                or intent["execution_policy_digest"] != self.execution_policy_digest
            ):
                raise ValueError("snapshot order differs from its account/source scope")
        object.__setattr__(self, "book_facts", book)
        object.__setattr__(self, "active_orders", orders)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "independent_allocation_account_snapshot_v1",
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())

    def canonical_bytes(self) -> bytes:
        """Serialize immutable facts without returning a mutable alias."""
        return canonical_json_bytes(self.payload())


__all__ = ["AllocationAccountSnapshot"]
