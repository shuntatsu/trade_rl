"""Pure, narrow readback of native settled accounts, without learner fiction.

Canonical bytes establish a detached identity, not runtime authenticity. The
four additional joins apply only to the declared normal single-symbol profile.
Failed prefixes and adopted but unrecorded frontiers never establish admission.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC
from fractions import Fraction
from hashlib import sha256
from types import MappingProxyType
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.contracts import MarketCalendarKind, VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.rl_allocation import settled_transition_validation as native
from trade_rl.evaluation.rl_allocation.settled_binding import (
    BoundSettledAllocationClock,
)
from trade_rl.evaluation.rl_allocation.settled_training_source import (
    settled_training_source_receipt,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    _json,
    _pending,
    validate_allocation_execution_facts_v2,
    validate_terminal_allocation_facts,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.bar_path import PathMode, select_bar_path
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.liquidity import (
    LiquidityPriority,
    LiquidityRequest,
    allocate_symbol_capacity,
)
from trade_rl.simulation.orders.admission import OrderAdmissionPolicy
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderEvent,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    TimeInForce,
)
from trade_rl.simulation.orders.reconciliation import reconcile_target
from trade_rl.simulation.quantities import exact_quantity, project_quantity
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile
from trade_rl.strategies.rl.allocation_recipe_v4 import allocation_recipe_payload_v4

_STAGES = frozenset(
    "actor_preflight actor_execution actor_returned actor_validation closing_preflight closing_execution closing_returned closing_validation actor_observation terminal_observation reset_observation capture_encoding unknown".split()
)


def _integer(value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("native capture clock/counter must be an actual integer")
    return value


def _digest(value: Any) -> str:
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("native capture digest must be lowercase SHA256")
    return value


def _error(value: Any) -> None:
    native._closed(value, {"type_name", "message"})
    if (
        type(value["type_name"]) is not str
        or not value["type_name"]
        or (value["message"] is not None and type(value["message"]) is not str)
    ):
        raise ValueError("native capture error identity is malformed")


def _decode(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise ValueError("native capture requires immutable bytes")
    try:
        value = json.loads(raw)
        native._native(value)
        if canonical_json_bytes(value) != raw:
            raise ValueError("native capture requires exact canonical JSON")
    except (UnicodeError, TypeError, OverflowError) as error:
        raise ValueError("invalid native capture bytes") from error
    if type(value) is not dict:
        raise ValueError("native capture must be an object")
    return value


def _hex(value: Any) -> bytes:
    if type(value) is not str:
        raise ValueError("native fragment hex must be a string")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("invalid native fragment hex") from error
    if raw.hex() != value:
        raise ValueError("native fragment hex is not canonical lowercase")
    return raw


def _frontier(value: Any) -> None:
    native._closed(
        value, {"index", "time_ns", "native_account", "state_digest", "encoding_error"}
    )
    if value["encoding_error"] is not None:
        _error(value["encoding_error"])
        if any(
            value[k] is not None
            for k in ("index", "time_ns", "native_account", "state_digest")
        ):
            raise ValueError("absent frontier cannot invent partial account facts")
        return
    _integer(value["index"])
    _integer(value["time_ns"], minimum=-(2**63))
    _digest(value["state_digest"])
    native._closed(value["native_account"], {"book", "active_orders"})


def _capture(value: dict[str, Any]) -> None:
    native._closed(
        value,
        {
            "schema",
            "context_digest",
            "opening",
            "actor_calls",
            "returned_frontier",
            "failure",
        },
    )
    if value["schema"] != "allocation_settled_native_capture_v1":
        raise ValueError("unknown native capture schema")
    _digest(value["context_digest"])
    _frontier(value["returned_frontier"])
    failure = value["failure"]
    if failure is not None:
        native._closed(
            failure,
            {"origin", "actor_call_ordinal", "stage", "attempted_start_index", "error"},
        )
        if (
            failure["origin"] not in {"reset", "native_call", "capture"}
            or failure["stage"] not in _STAGES
        ):
            raise ValueError("unknown native capture failure stage")
        for key in ("actor_call_ordinal", "attempted_start_index"):
            if failure[key] is not None:
                _integer(failure[key])
        _error(failure["error"])
    opening = value["opening"]
    if opening is None:
        if (
            failure is None
            or failure["origin"] not in {"reset", "capture"}
            or value["actor_calls"]
        ):
            raise ValueError(
                "missing opening requires an explicit reset/capture failure"
            )
    else:
        native._closed(
            opening,
            {
                "episode_id",
                "index",
                "time_ns",
                "reset_info",
                "native_account",
                "state_digest",
            },
        )
        _integer(opening["episode_id"], minimum=1)
        _integer(opening["index"])
        _integer(opening["time_ns"], minimum=-(2**63))
        _digest(opening["state_digest"])
        native._closed(opening["reset_info"], {"symbol", "episode_id"})
        _integer(opening["reset_info"]["episode_id"], minimum=1)
        native._closed(opening["native_account"], {"book", "active_orders"})
    calls = value["actor_calls"]
    if type(calls) is not list:
        raise ValueError("actor calls must be an ordered list")
    for ordinal, call in enumerate(calls):
        native._closed(
            call,
            {
                "ordinal",
                "start_index",
                "start_time_ns",
                "action",
                "action_type",
                "outcome",
                "transition_hex",
                "failure_fragment_hex",
            },
        )
        _integer(call["ordinal"])
        _integer(call["start_index"])
        _integer(call["start_time_ns"], minimum=-(2**63))
        if (
            call["ordinal"] != ordinal
            or type(call["action"]) is not int
            or call["action"] not in range(4)
            or call["action_type"] != "int"
        ):
            raise ValueError("native capture has an invalid ordered actor action")
        if call["outcome"] == "returned":
            _hex(call["transition_hex"])
            if call["failure_fragment_hex"] is not None:
                raise ValueError("a successful call cannot contain a failed fragment")
        elif call["outcome"] == "raised":
            if (
                ordinal != len(calls) - 1
                or call["transition_hex"] is not None
                or failure is None
                or failure["origin"] != "native_call"
                or failure["actor_call_ordinal"] != ordinal
            ):
                raise ValueError("a raised call cannot return success or continue")
            if call["failure_fragment_hex"] is not None:
                _hex(call["failure_fragment_hex"])
        else:
            raise ValueError("unknown actor call outcome")
    if (
        failure is not None
        and failure["origin"] == "native_call"
        and (not calls or calls[-1]["outcome"] != "raised")
    ):
        raise ValueError("native failure requires its actual raised call")


@dataclass(frozen=True, slots=True)
class SettledNativeActorIntent:
    ordinal: int
    start_index: int
    action: int

    def __post_init__(self) -> None:
        if (
            any(
                type(v) is not int or v < 0
                for v in (self.ordinal, self.start_index, self.action)
            )
            or self.action > 3
        ):
            raise ValueError(
                "actor intent requires strict integer ordinal/clock/action"
            )


@dataclass(frozen=True, slots=True)
class SettledNativeCapture:
    _bytes: bytes

    def __post_init__(self) -> None:
        _capture(_decode(self._bytes))

    @classmethod
    def from_bytes(cls, raw: bytes) -> SettledNativeCapture:
        return cls(raw)

    def payload(self) -> dict[str, Any]:
        return _decode(self._bytes)

    def to_bytes(self) -> bytes:
        return self._bytes


@dataclass(frozen=True, slots=True)
class SettledNativeValidationContext:
    """Passive declared sources; source mutation is checked before and after."""

    dataset: MarketDataset
    stream: FrozenSimpleReturnStream
    estimates: tuple[HorizonCostEstimates, ...]
    bound: BoundSettledAllocationClock
    action_contract: AllocationActionContract
    allocator: AfterCostTargetAllocator
    execution_cost: ExecutionCostConfig
    risk_config: PreTradeRiskConfig
    feature_indices: tuple[int, ...]
    symbol_index: int
    start_index: int
    account_id: str
    observation_schema: AllocationObservationSchema
    expected_horizon_seconds: int
    _source_bytes: bytes = field(init=False, repr=False)
    _recipe_bytes: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.estimates) is not tuple
            or type(self.feature_indices) is not tuple
            or any(type(i) is not int for i in self.feature_indices)
        ):
            raise ValueError("native context requires immutable declared selections")
        _integer(self.start_index)
        _integer(self.symbol_index)
        _integer(self.expected_horizon_seconds, minimum=1)
        if type(self.account_id) is not str or not self.account_id.strip():
            raise ValueError("native context requires an account identity")
        if not (
            0
            <= self.start_index
            < self.actor_stop_index
            < self.settlement_stop_index
            < self.dataset.n_bars
            and self.symbol_index < self.dataset.n_symbols
        ):
            raise ValueError("native context exceeds its declared Dataset interval")
        objective = self.bound.objective
        if (
            objective.economics_digest != self.economics_digest
            or objective.risk_digest != content_digest(self.risk_config)
            or objective.deployment_recipe_digest != self.recipe_digest
            or objective.maximum_drawdown != self.risk_config.drawdown_stop
            or not 0
            <= self.risk_config.drawdown_start
            < self.risk_config.drawdown_stop
            <= 0.20
        ):
            raise ValueError(
                "native objective differs from its declared source contracts"
            )
        times = self.dataset.timestamps[
            self.start_index : self.settlement_stop_index + 1
        ].astype("datetime64[ns]")
        ticks = times.astype(np.int64)
        if np.any(
            np.diff(ticks) != self.bound.clock.execution_interval_seconds * 10**9
        ):
            raise ValueError("native Dataset intervals differ from the declared clock")
        for actual, declared in (
            (times[0], objective.evaluation_start),
            (times[-1], objective.evaluation_stop_exclusive),
        ):
            expected = np.datetime64(
                declared.astimezone(UTC).replace(tzinfo=None), "us"
            ).astype("datetime64[ns]")
            if actual != expected:
                raise ValueError("native Dataset endpoint differs from the objective")
        if len(self._estimates) != len(self.estimates):
            raise ValueError("native context has duplicate cost clocks")
        object.__setattr__(self, "_recipe_bytes", canonical_json_bytes(self.recipe))
        object.__setattr__(
            self,
            "_source_bytes",
            canonical_json_bytes(settled_training_source_receipt(self)),
        )

    @property
    def actor_stop_index(self) -> int:
        return self.start_index + self.bound.clock.actor_steps

    @property
    def settlement_stop_index(self) -> int:
        return self.start_index + self.bound.clock.physical_horizon_steps

    @property
    def initial_capital(self) -> float:
        return self.bound.objective.capital.initial_equities[0]

    @property
    def _estimates(self) -> Mapping[int, HorizonCostEstimates]:
        return MappingProxyType(
            {
                int(c.decision_time.astype("datetime64[ns]").astype(np.int64)): c
                for c in self.estimates
            }
        )

    @property
    def economics_digest(self) -> str:
        return content_digest(
            {
                "schema_version": "insolvency_execution_policy_v1",
                "base_policy_digest": self.execution_cost.execution_policy_digest,
                "insolvency_valuation": "retain_debt",
            }
        )

    @property
    def recipe(self) -> dict[str, object]:
        clock, capital = self.bound.clock, self.bound.objective.capital
        return allocation_recipe_payload_v4(
            self.action_contract,
            tuple(self.dataset.feature_names[i] for i in self.feature_indices),
            allocator=self.allocator,
            expected_horizon_seconds=self.expected_horizon_seconds,
            runtime_profile=AllocationRuntimeProfile(
                self.economics_digest,
                content_digest(self.risk_config),
                self.initial_capital,
                capital.currency,
                clock.execution_interval_seconds,
                clock.physical_horizon_steps * clock.execution_interval_seconds,
                calendar_kind=MarketCalendarKind(self.dataset.calendar_kind).value,
                execution_bar_hours=self.dataset.bar_hours,
            ),
            settled_clock=clock,
            observation_schema=self.observation_schema,
        )

    @property
    def recipe_digest(self) -> str:
        return content_digest(self.recipe)

    @property
    def _source_envelope(self) -> dict[str, Any]:
        return json.loads(self._source_bytes)

    @property
    def _source_digest(self) -> str:
        return sha256(self._source_bytes).hexdigest()

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema": "allocation_settled_native_context_v1",
                "source_envelope_digest": self._source_digest,
                "recipe_digest": self.recipe_digest,
            }
        )

    def validate_sources(self) -> None:
        if (
            canonical_json_bytes(self.recipe) != self._recipe_bytes
            or canonical_json_bytes(settled_training_source_receipt(self))
            != self._source_bytes
        ):
            raise ValueError("native context declared sources changed")


def _normal_drawdown_sites(
    peak: float, drawdown: float, opened: float, filled: float | None, marked: float
) -> tuple[float, float]:
    """Exactly the open refresh, optional accepted fill and final mark sites."""
    for equity in (opened, filled, marked):
        if equity is None:
            continue
        value = max(equity, 0.0)
        peak = max(peak, value)
        drawdown = max(
            drawdown, float(1.0 - value / max(peak, float(np.finfo(np.float64).tiny)))
        )
    return peak, drawdown


def _counter_turnover_join(
    before: dict[str, Any],
    after: dict[str, Any],
    filled_notional: float,
    exact_delta: Fraction,
) -> None:
    count = int(abs(exact_delta) > exact_quantity(1e-12))
    if (
        after["fill_count"] != before["fill_count"] + count
        or after["rebalance_events"] != before["rebalance_events"] + count
    ):
        raise ValueError("native Book counter differs from actual accepted quantity")
    native._amount(
        after["turnover_total"],
        before["turnover_total"] + filled_notional / max(before["equity"], 1e-12),
        "Book turnover",
    )


def _normal_profile(context: SettledNativeValidationContext) -> bool:
    dataset, cost = context.dataset, context.execution_cost
    start, stop = context.start_index, context.settlement_stop_index
    if not (
        dataset.n_symbols == 1
        and context.symbol_index == 0
        and start == 6
        and context.initial_capital == 1000.0
        and context.bound.objective.capital.currency == "USD"
        and context.bound.clock.actor_steps in (1, 2)
        and context.bound.clock.reserve_steps in (1, 2)
        and dataset.calendar_kind == MarketCalendarKind.CONTINUOUS
        and dataset.bar_hours == 1.0
        and context.bound.clock.execution_interval_seconds == 3600
        and dataset.volume_units == (VolumeUnit.BASE_ASSET,)
        and cost.order_type == "market"
        and cost.order_latency_bars == 0
        and cost.processing_bar_volume_capacity
        and cost.partial_fill_carry
        and cost.slippage_std == cost.tail_slippage_probability == 0.0
        and cost.tick_size == cost.lot_size == cost.minimum_notional == 0.0
        and cost.trigger_volume_fractions[0] == 1.0
        and context.bound.objective.maximum_drawdown == 0.2
        and cost
        == replace(
            ExecutionCostConfig.zero(), fee_rate=1 / 128, max_participation_rate=1.0
        )
        and context.risk_config
        == PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.1,
            drawdown_stop=0.2,
        )
    ):
        return False
    if any(dataset.elapsed_hours(i, i + 1) != 1.0 for i in range(start, stop)):
        return False
    for name, expected in (
        ("tradable", True),
        ("asset_active", True),
        ("buy_allowed", True),
        ("sell_allowed", True),
        ("borrow_available", True),
        ("split_factor", 1.0),
        ("tick_size", 0.0),
        ("lot_size", 0.0),
        ("minimum_notional", 0.0),
        ("funding_rate", 0.0),
        ("funding_price_rate", 0.0),
        ("funding_event_count", 0),
        ("borrow_rate", 0.0),
        ("dividend", 0.0),
    ):
        if np.any(dataset.resolved_array(name)[start : stop + 1] != expected):
            return False
    return True


def _normal_event(
    context: SettledNativeValidationContext,
    previous: PendingOrder,
    updated: PendingOrder,
    *,
    event_type: str,
    processing_index: int,
    sequence: int,
    **values: Any,
) -> dict[str, Any]:
    """The canonical immutable event projection, without a runtime/executor."""
    return _json(
        OrderEvent(
            schema_version="order_event_v1",
            sequence=sequence,
            order_id=updated.order_id,
            replaced_order_id=updated.intent.replaced_order_id,
            dataset_id=updated.intent.dataset_id,
            execution_policy_digest=updated.intent.execution_policy_digest,
            symbol_index=updated.intent.symbol_index,
            reduce_only=updated.intent.reduce_only,
            event_type=event_type,
            processing_index=processing_index,
            timestamp_ns=native._time(context, processing_index),
            previous_status=previous.status,
            new_status=updated.status,
            requested_quantity=updated.intent.requested_quantity,
            remaining_quantity=updated.remaining_quantity,
            filled_quantity=0.0,
            execution_price=None,
            filled_notional=0.0,
            capacity_before=0.0,
            capacity_after=0.0,
            participation_rate=0.0,
            trigger_segment=None,
            available_volume_fraction=0.0,
            reason=None,
            path_mode=context.execution_cost.path_mode,
            path_points=(),
        ).canonical_payload()
        | values
    )


def _normal_pending_progress(
    order: PendingOrder,
    *,
    processing_index: int,
    filled_quantity: float,
    filled_notional: float,
) -> tuple[PendingOrder, tuple[tuple[PendingOrder, PendingOrder, str], ...]]:
    transitions = []
    if order.status in {OrderStatus.SUBMITTED, OrderStatus.LATENCY_WAIT}:
        eligible = order.mark_eligible(processing_index=processing_index)
        transitions.append((order, eligible, "eligible"))
        order = eligible
    if abs(filled_quantity) <= 1e-12:
        transitions.append((order, order, "no_fill"))
        return order, tuple(transitions)
    updated = order.apply_fill(
        quantity=filled_quantity,
        notional=filled_notional,
        processing_index=processing_index,
        lot_size=0.0,
    )
    transitions.append(
        (
            order,
            updated,
            "filled" if updated.status is OrderStatus.FILLED else "partial_fill",
        )
    )
    return updated, tuple(transitions)


def _reconciled_orders(
    context: SettledNativeValidationContext,
    row: dict[str, Any],
    before: dict[str, Any],
    target_identity: str | None,
) -> tuple[PendingOrder, ...]:
    """Normative sizing from the prior account, not from submitted witnesses."""
    index, cost, dataset = row["start_index"], context.execution_cost, context.dataset
    book = native._account(context, before, index)
    prior = tuple(PendingOrder.from_mapping(o) for o in before["active_orders"])
    if any(order.intent.order_type is not OrderType.MARKET for order in prior):
        raise ValueError("normal profile cannot carry a non-MARKET order")
    orders = OrderBookState(active_orders=prior, terminal_orders=())
    active: tuple[PendingOrder, ...]
    intents: tuple[OrderIntent, ...]
    target = np.asarray(row["risk"]["weights"], dtype=np.float64)
    if np.array_equal(target, book.weights):
        cancelled = tuple(
            order.cancel(processing_index=index, reason="quantity_hold")
            for order in prior
        )
        active, intents = (), ()
    else:
        if target_identity is None:
            raise ValueError("a changed risk target requires a target identity")
        result = reconcile_target(
            dataset_id=dataset.dataset_id,
            target_identity=target_identity,
            execution_policy_digest=context.economics_digest,
            target_weights=target,
            book=book,
            order_book=orders,
            reference_prices=dataset.close[index],
            decision_equity=max(book.portfolio_value, 1e-12),
            submit_index=index,
            latency_bars=cost.order_latency_bars,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            expiry_index=None,
            limit_offset_rate=cost.limit_offset_rate,
            tick_sizes=np.maximum(
                dataset.resolved_array("tick_size")[index], cost.tick_size
            ),
            maximum_gross=cost.max_leverage,
            reduce_only_symbols=None,
            valuation_prices=book.mark_prices
            if any(book.exact_quantities)
            else dataset.resolved_array("mark_price")[index],
        )
        cancelled, active, intents = (
            result.cancelled_orders,
            result.order_book.active_orders,
            result.new_intents,
        )
    by_id = {order["intent"]["order_id"]: order for order in row["accepted_orders"]}
    events = row["native"]["order_events"]
    submissions = [e for e in events if e["event_type"] == "submitted"]
    if [e["order_id"] for e in submissions] != [intent.order_id for intent in intents]:
        raise ValueError("native submitted intent differs from causal target sizing")
    for intent in intents:
        native._same(by_id[intent.order_id]["intent"], intent.canonical_payload())
    cancellations = [e for e in events if e["event_type"] == "cancelled"]
    if [e["order_id"] for e in cancellations] != [
        order.order_id for order in cancelled
    ]:
        raise ValueError("native reconciliation cancellations differ from prior orders")
    for order, event in zip(cancelled, cancellations, strict=True):
        native._same(by_id[order.order_id], _pending(order))
        native._same(
            event,
            _normal_event(
                context,
                next(o for o in prior if o.order_id == order.order_id),
                order,
                event_type="cancelled",
                processing_index=index,
                sequence=cancelled.index(order),
                reason=order.terminal_reason,
            ),
        )
    for ordinal, (intent, event) in enumerate(zip(intents, submissions, strict=True)):
        submitted = PendingOrder.from_intent(intent)
        native._same(
            event,
            _normal_event(
                context,
                submitted,
                submitted,
                event_type="submitted",
                processing_index=index,
                sequence=len(cancelled) + ordinal,
            ),
        )
    if [e["event_type"] for e in events[: len(cancelled) + len(intents)]] != [
        *(["cancelled"] * len(cancelled)),
        *(["submitted"] * len(intents)),
    ]:
        raise ValueError("normal reconciliation events are out of causal order")
    return (*active, *(PendingOrder.from_intent(intent) for intent in intents))


@dataclass(slots=True)
class _AdmissionView:
    quantities: np.ndarray
    contract_multipliers: np.ndarray
    insolvent: bool
    portfolio_value: float


class _UnsupportedNormalProfile(ValueError):
    """A truthful source outcome lacks the bounded strong-phase proof."""


def _capacity_cost_join(
    context: SettledNativeValidationContext,
    row: dict[str, Any],
    before: dict[str, Any],
    orders: tuple[PendingOrder, ...],
) -> None:
    """Source-derived capacity and effective MARKET fees; no executor replay."""
    dataset, cost, i = context.dataset, context.execution_cost, row["processing_index"]
    actual = row["native"]["execution"]
    if len(orders) > 1:
        raise ValueError("normal profile admits at most one active request")
    multiplier = float(dataset.resolved_array("contract_multipliers")[0])
    requested = math.fsum(
        abs(o.remaining_quantity) * o.intent.submission_reference_price * multiplier
        for o in orders
    )
    native._amount(actual["requested_notional"], requested, "requested notional")
    native._same(actual["requested_notional_by_symbol"], [requested])
    outcomes = [
        e
        for e in row["native"]["order_events"]
        if e["event_type"] in {"filled", "partial_fill", "no_fill"}
    ]
    if not orders:
        native._same(actual["capacity_evidence"], [])
        native._same(outcomes, [])
        filled = reference_filled = charge = participation = 0.0
        if any(
            e["event_type"] not in {"cancelled", "submitted"}
            for e in row["native"]["order_events"]
        ):
            raise ValueError("quantity HOLD cannot invent processing events")
    else:
        order = orders[0]
        quantities = np.asarray(before["book"]["quantities"], dtype=np.float64)
        multipliers = dataset.resolved_array("contract_multipliers")
        equity = float(
            before["book"]["cash"] + (quantities * dataset.open[i] * multipliers).sum()
        )
        admission = OrderAdmissionPolicy(
            dataset.dataset_id,
            context.economics_digest,
            cost.allow_short,
            cost.max_leverage,
        ).evaluate(
            order.intent,
            remaining_quantity=order.remaining_quantity,
            book=_AdmissionView(
                quantities,
                multipliers,
                bool(before["book"]["insolvent"] or equity <= 0.0),
                equity,
            ),
            processing_index=i,
            asset_active=bool(dataset.resolved_array("asset_active")[i, 0]),
            tradable=bool(dataset.tradable[i, 0]),
            buy_allowed=bool(dataset.resolved_array("buy_allowed")[i, 0]),
            sell_allowed=bool(dataset.resolved_array("sell_allowed")[i, 0]),
            borrow_available=bool(dataset.resolved_array("borrow_available")[i, 0]),
            tick_size=0.0,
            lot_size=0.0,
            minimum_notional=0.0,
            reference_prices=dataset.open[i],
            actual_position=Fraction(before["book"]["exact_quantities"][0]),
        )
        if not admission.accepted or order.intent.eligible_index > i:
            raise _UnsupportedNormalProfile(
                "source-derived request is not eligible/admitted"
            )
        path = select_bar_path(
            open_price=float(dataset.open[i, 0]),
            high=float(dataset.high[i, 0]),
            low=float(dataset.low[i, 0]),
            close=float(dataset.close[i, 0]),
            mode=PathMode(cost.path_mode),
            active_directions=frozenset((1 if order.remaining_quantity > 0 else -1,)),
        )
        quantity_capacity = float(dataset.volume[i, 0]) / multiplier
        allocations, capacity = allocate_symbol_capacity(
            [
                LiquidityRequest(
                    order.order_id,
                    order.remaining_quantity,
                    float(dataset.open[i, 0]),
                    1.0,
                    LiquidityPriority.MARKET,
                    order.intent.eligible_index,
                    reduce_only=order.intent.reduce_only,
                    minimum_notional=0.0,
                )
            ],
            processing_volume=float(dataset.volume[i, 0]),
            processing_market_notional=float(
                dataset.market_notional(i, dataset.open[i])[0]
            ),
            processing_quantity_capacity=quantity_capacity,
            price=float(dataset.open[i, 0]),
            contract_multiplier=multiplier,
            participation_limit=min(
                cost.max_participation_rate,
                float(dataset.resolved_array("max_participation_rate")[i, 0]),
            ),
            lot_size=0.0,
            minimum_notional=0.0,
            initial_position=Fraction(before["book"]["exact_quantities"][0]),
        )
        allocation = allocations[0]
        native._same(actual["capacity_evidence"], [_json(capacity)])
        if len(outcomes) != 1:
            raise ValueError("native capacity lacks its unique fill/no-fill event")
        event = outcomes[0]
        native._same(
            [
                event[k]
                for k in (
                    "order_id",
                    "filled_quantity",
                    "filled_notional",
                    "capacity_before",
                    "capacity_after",
                    "available_volume_fraction",
                    "trigger_segment",
                    "path_mode",
                    "path_points",
                )
            ],
            [
                order.order_id,
                allocation.filled_quantity,
                allocation.filled_notional,
                allocation.capacity_before,
                allocation.capacity_after,
                1.0,
                "open",
                path.mode.value,
                list(path.points),
            ],
        )
        filled, participation = (
            allocation.filled_notional,
            allocation.participation_rate,
        )
        if abs(allocation.filled_quantity) <= 1e-12:
            native._same(
                [
                    event["event_type"],
                    event["execution_price"],
                    event["participation_rate"],
                    event["reason"],
                ],
                ["no_fill", None, 0.0, allocation.no_fill_reason],
            )
            participation = 0.0
        else:
            native._same(
                [event["execution_price"], event["participation_rate"]],
                [float(dataset.open[i, 0]), participation],
            )
        updated, transitions = _normal_pending_progress(
            order,
            processing_index=i,
            filled_quantity=allocation.filled_quantity,
            filled_notional=allocation.filled_notional,
        )
        accepted = next(
            o
            for o in row["accepted_orders"]
            if o["intent"]["order_id"] == order.order_id
        )
        native._same(accepted, _pending(updated))
        processing_events = [
            e
            for e in row["native"]["order_events"]
            if e["event_type"] not in {"cancelled", "submitted"}
        ]
        sequence = len(row["native"]["order_events"]) - len(processing_events)
        expected_events: list[dict[str, Any]] = []
        for prior_order, next_order, event_type in transitions:
            values = (
                {}
                if event_type == "eligible"
                else dict(
                    filled_quantity=allocation.filled_quantity,
                    filled_notional=allocation.filled_notional,
                    execution_price=None
                    if event_type == "no_fill"
                    else float(dataset.open[i, 0]),
                    capacity_before=allocation.capacity_before,
                    capacity_after=allocation.capacity_after,
                    participation_rate=participation,
                    trigger_segment="open",
                    available_volume_fraction=1.0,
                    path_mode=path.mode.value,
                    path_points=list(path.points),
                    reason=allocation.no_fill_reason
                    if event_type == "no_fill"
                    else next_order.terminal_reason,
                )
            )
            expected_events.append(
                _normal_event(
                    context,
                    prior_order,
                    next_order,
                    event_type=event_type,
                    processing_index=i,
                    sequence=sequence + len(expected_events),
                    **values,
                )
            )
        native._same(processing_events, expected_events)
        reference_filled = (
            abs(allocation.filled_quantity)
            * order.intent.submission_reference_price
            * multiplier
        )
        venue_fee = cost.taker_fee_rate + dataset.resolved_array("taker_fee_rate")[i, 0]
        spread = cost.spread_rate + dataset.resolved_array("spread_rate")[i, 0]
        unit_cost = cost.multiplier * (
            cost.fee_rate
            + dataset.resolved_array("fee_rate")[i, 0]
            + venue_fee
            + 1.0 * spread
            + cost.impact_rate * math.sqrt(participation)
            + 0.0
        )
        charge = filled * float(unit_cost)
    starting = max(before["book"]["equity"], 1e-12)
    native._same([actual["rejected_count"], actual["expired_count"]], [0, 0])
    for key, value in (
        ("filled_notional", filled),
        ("interval_cost", charge),
        ("requested_turnover", requested / starting),
        ("filled_turnover", filled / starting),
        (
            "unfilled_turnover",
            max(0.0, requested / starting - reference_filled / starting),
        ),
        (
            "fill_ratio",
            1.0 if requested <= 1e-12 else min(1.0, reference_filled / requested),
        ),
        ("max_participation", participation),
    ):
        native._amount(actual[key], value, "capacity " + key)
    for key, value in (
        ("filled_notional_by_symbol", filled),
        ("cost_by_symbol", charge),
        ("participation_by_symbol", participation),
    ):
        native._same(actual[key], [value])


def _normal_relations(
    context: SettledNativeValidationContext,
    row: dict[str, Any],
    before: dict[str, Any],
    target_identity: str | None,
) -> None:
    if before["book"]["insolvent"] or row["native"]["book"]["insolvent"]:
        raise ValueError("normal native rows cannot hide an insolvent account")
    orders = _reconciled_orders(context, row, before, target_identity)
    _capacity_cost_join(context, row, before, orders)
    previous, after = before["book"], row["native"]["book"]
    execution, dataset, i = (
        row["native"]["execution"],
        context.dataset,
        row["processing_index"],
    )
    delta = Fraction(after["exact_quantities"][0]) - Fraction(
        previous["exact_quantities"][0]
    )
    _counter_turnover_join(previous, after, execution["filled_notional"], delta)
    native._same(
        execution["rebalance_events"],
        after["rebalance_events"] - previous["rebalance_events"],
    )
    q, multiplier = (
        float(previous["quantities"][0]),
        float(dataset.resolved_array("contract_multipliers")[0]),
    )
    opened = float(previous["cash"]) + q * float(dataset.open[i, 0]) * multiplier
    fills = [
        e
        for e in row["native"]["order_events"]
        if e["event_type"] in {"filled", "partial_fill"}
    ]
    filled = None
    if fills:
        event = fills[0]
        signed = project_quantity(delta) * float(event["execution_price"]) * multiplier
        cash = float(previous["cash"]) - (signed + execution["interval_cost"])
        filled = (
            cash
            + float(after["quantities"][0])
            * float(event["execution_price"])
            * multiplier
        )
    peak, drawdown = _normal_drawdown_sites(
        previous["peak_value"],
        previous["max_drawdown"],
        opened,
        filled,
        after["equity"],
    )
    native._amount(after["peak_value"], peak, "peak update sites")
    native._amount(after["max_drawdown"], drawdown, "drawdown update sites")


def _row(
    context: SettledNativeValidationContext,
    raw: Any,
    before: dict[str, Any],
    *,
    episode: int,
    index: int,
    kind: str,
    action: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = native._closed(
        raw, native._ROW_KEYS | ({"proposal"} if kind == "actor" else set())
    )
    native._native(row)
    _integer(row["start_index"])
    _integer(row["processing_index"])
    native._same(
        {
            k: row[k]
            for k in (
                "kind",
                "start_index",
                "processing_index",
                "clock_valid",
                "returned_clock_types",
                "opening_equity",
                "before_state_digest",
            )
        },
        {
            "kind": kind,
            "start_index": index,
            "processing_index": index + 1,
            "clock_valid": True,
            "returned_clock_types": {"next_index": "int", "bars_advanced": "int"},
            "opening_equity": before["book"]["equity"],
            "before_state_digest": native._state(context, episode, index, before),
        },
    )
    current_book = native._account(context, before, index)
    current = current_book.weights
    result = row["native"]
    if kind == "actor":
        if type(action) is not int or action not in range(4):
            raise ValueError("native actor row requires an actual integer action")
        proposal = native._closed(
            row["proposal"],
            set(
                "decision raw_action action raw_target_weight target_weight is_hold".split()
            ),
        )
        if (
            type(proposal["raw_action"]) is not int
            or type(proposal["is_hold"]) is not bool
        ):
            raise ValueError("native proposal lost its strict actor scalar types")
        decision = native._decision(context, proposal["decision"])
        native._source_decision(context, index, before, decision)
        expected = decision.propose(action)
        native._same(proposal, _json(expected))
        complete = result | {
            "decision_index": index,
            "decision_time_ns": native._time(context, index),
            "processing_index": index + 1,
            "processing_time_ns": native._time(context, index + 1),
            "symbol_index": context.symbol_index,
            "proposal": proposal,
            "proposal_digest": expected.digest,
            "decision_digest": decision.decision_digest,
            "risk": row["risk"],
            "risk_config": _json(context.risk_config),
            "current_weights": current.tolist(),
        }
        validate_allocation_execution_facts_v2(
            complete,
            context.recipe,
            dataset_id=context.dataset.dataset_id,
            action_code=action,
            actual_cost_payload=context.execution_cost.execution_policy_payload(),
        )
        decision_identity = (
            decision.baseline.decision_digest
            if expected.target_weight == decision.baseline.target_weight
            else expected.digest
        )
    else:
        validate_terminal_allocation_facts(
            result,
            dataset_id=context.dataset.dataset_id,
            economics_digest=context.economics_digest,
            symbol_index=context.symbol_index,
            start_index=index,
            start_time_ns=native._time(context, index),
            processing_index=index + 1,
            processing_time_ns=native._time(context, index + 1),
        )
        native._same(
            row["risk"],
            _json(
                PreTradeRisk(context.risk_config).constrain(
                    np.zeros(context.dataset.n_symbols),
                    current=current,
                    drawdown=before["book"]["max_drawdown"],
                )
            ),
        )
        decision_identity = content_digest(
            {
                "schema": "allocation_settled_reserve_decision_v1",
                "bound_digest": context.bound.digest,
                "episode_id": episode,
                "index": index,
            }
        )
    target = np.asarray(row["risk"]["weights"], dtype=np.float64)
    target_identity = (
        None
        if np.array_equal(target, current)
        else content_digest({"decision": decision_identity, "target": target.tolist()})
    )
    native._native_sources(context, row, before, target_identity=target_identity)
    after = {"book": result["book"], "active_orders": result["active_orders"]}
    native._same(
        row["after_state_digest"], native._state(context, episode, index + 1, after)
    )
    economic = after["book"]["termination_reason"] is not None
    if not economic:
        _normal_relations(context, row, before, target_identity)
    check = {
        k: row[k]
        for k in (
            "kind",
            "start_index",
            "processing_index",
            "before_state_digest",
            "after_state_digest",
        )
    }
    check.update(
        source_selection_digest=content_digest(
            native._source_selection(context, episode, row)
        ),
        coverage="ECONOMIC_END_DIAGNOSTIC" if economic else "NORMAL_RELATIONS",
    )
    return after, check


def _header(
    context: SettledNativeValidationContext,
    payload: dict[str, Any],
    call: dict[str, Any],
    episode: int,
    *,
    failed: bool,
) -> None:
    native._closed(
        payload,
        native._PAYLOAD_KEYS
        | ({"failure_stage", "attempted_start_index"} if failed else set()),
    )
    native._native(payload)
    native._same(
        {
            k: payload[k]
            for k in (
                "schema",
                "episode_id",
                "account_id",
                "actor_step",
                "actor_index",
                "bound_digest",
                "recipe_digest",
                "source_envelope_digest",
                "initial_capital",
                "truncated",
            )
        },
        {
            "schema": "allocation_settled_transition_v1",
            "episode_id": episode,
            "account_id": context.account_id,
            "actor_step": call["ordinal"],
            "actor_index": call["start_index"],
            "bound_digest": context.bound.digest,
            "recipe_digest": context.recipe_digest,
            "source_envelope_digest": context._source_digest,
            "initial_capital": context.initial_capital,
            "truncated": False,
        },
    )
    for key in ("episode_id", "actor_step", "actor_index"):
        _integer(payload[key])
    if (
        type(payload["terminated"]) is not bool
        or type(payload["truncated"]) is not bool
        or type(payload["rows"]) is not list
    ):
        raise ValueError("native transition flags/rows have invalid types")
    if failed:
        _integer(payload["attempted_start_index"])
        if (
            payload["reward"] is not None
            or payload["terminated"] is not True
            or payload["failure_stage"] not in _STAGES
        ):
            raise ValueError("failed native fragment cannot be a successful return")
    elif type(payload["reward"]) not in (int, float):
        raise ValueError("native reward must be finite numeric, not bool")


def _witness(
    context: SettledNativeValidationContext,
    raw: Any,
    account: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    failed: bool,
) -> None:
    value = native._closed(raw, native._WITNESS_KEYS)
    for key in (
        "actor_steps_processed",
        "physical_steps_processed",
        "reserve_steps_processed",
        "frontier_index",
    ):
        _integer(value[key])
    for key in ("full_coverage", "settled", "eligible"):
        if type(value[key]) is not bool:
            raise ValueError("native settlement flag must be bool")
    actor = sum(row["kind"] == "actor" for row in checks)
    full = (
        actor == context.bound.clock.actor_steps
        and len(checks) == context.bound.clock.physical_horizon_steps
        and account["book"]["as_of_index"] == context.settlement_stop_index
    )
    settled = (
        full
        and not failed
        and account["book"]["termination_reason"] is None
        and not account["active_orders"]
        and all(Fraction(q) == 0 for q in account["book"]["exact_quantities"])
    )
    native._same(
        value,
        {
            "actor_steps_processed": actor,
            "physical_steps_processed": len(checks),
            "reserve_steps_processed": sum(row["kind"] == "reserve" for row in checks),
            "full_coverage": full,
            "settled": settled,
            "eligible": settled
            and account["book"]["max_drawdown"]
            <= context.bound.objective.maximum_drawdown,
            "maximum_drawdown": account["book"]["max_drawdown"],
            "frontier_index": account["book"]["as_of_index"],
            "native_frontier": account,
        },
    )


def _failed_witness(
    context: SettledNativeValidationContext,
    raw: Any,
    frontier: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    episode: int,
) -> None:
    if frontier["encoding_error"] is not None:
        raise ValueError("failed witness has no observed frontier to join")
    index = frontier["index"]
    prefix_index = checks[-1]["processing_index"] if checks else context.start_index
    if not prefix_index <= index <= context.settlement_stop_index:
        raise ValueError("failed witness frontier precedes the validated prefix")
    account = frontier["native_account"]
    native._account(context, account, index)
    native._same(frontier["time_ns"], native._time(context, index))
    native._same(
        frontier["state_digest"], native._state(context, episode, index, account)
    )
    _witness(context, raw, account, checks, failed=True)


@dataclass(frozen=True, slots=True)
class SettledNativeVerification:
    """Derived detached report; this is not a credential or admission input."""

    _bytes: bytes

    def payload(self) -> dict[str, Any]:
        return _decode(self._bytes)

    def to_bytes(self) -> bytes:
        return self._bytes

    @property
    def admitted(self) -> bool:
        return bool(self.payload()["admitted"])


def validate_settled_native_capture(
    context: SettledNativeValidationContext,
    capture: SettledNativeCapture,
    *,
    actor_intents: tuple[SettledNativeActorIntent, ...],
) -> SettledNativeVerification:
    """Derive coverage from declared sources and actual detached native bytes."""
    if (
        type(context) is not SettledNativeValidationContext
        or type(capture) is not SettledNativeCapture
        or type(actor_intents) is not tuple
        or any(type(i) is not SettledNativeActorIntent for i in actor_intents)
    ):
        raise ValueError("native verification requires its typed passive inputs")
    context.validate_sources()
    value, checks, limits = (
        capture.payload(),
        [],
        ["OPAQUE_HISTORY_RNG", "NO_RUNTIME_AUTHENTICATION"],
    )
    native._same(value["context_digest"], context.digest)
    calls, opening = value["actor_calls"], value["opening"]
    if len(calls) != len(actor_intents) or len(calls) > context.bound.clock.actor_steps:
        raise ValueError("native capture differs from ordered actor intents")
    for ordinal, (intent, call) in enumerate(zip(actor_intents, calls, strict=True)):
        native._same(
            [intent.ordinal, intent.start_index, intent.action],
            [ordinal, context.start_index + ordinal, call["action"]],
        )
        native._same(
            [call["start_index"], call["start_time_ns"]],
            [intent.start_index, native._time(context, intent.start_index)],
        )
    normal = _normal_profile(context)
    if not normal:
        limits.append("PROFILE_UNSUPPORTED")
    before = None
    if opening is not None:
        native._same(
            [opening["index"], opening["time_ns"], opening["reset_info"]],
            [
                context.start_index,
                native._time(context, context.start_index),
                {
                    "symbol": context.dataset.symbols[context.symbol_index],
                    "episode_id": opening["episode_id"],
                },
            ],
        )
        native._reset_account(context, opening["native_account"])
        native._same(
            opening["state_digest"],
            native._state(
                context,
                opening["episode_id"],
                context.start_index,
                opening["native_account"],
            ),
        )
        before = opening["native_account"]
    for call in calls:
        if before is None or opening is None:
            raise ValueError("actual actor calls require an observed opening account")
        failed = call["outcome"] == "raised"
        encoded = call["failure_fragment_hex"] if failed else call["transition_hex"]
        if encoded is None:
            limits.append("MISSING_FRAGMENT")
            break
        try:
            payload = _decode(_hex(encoded))
            _header(context, payload, call, opening["episode_id"], failed=failed)
            if failed:
                native._same(
                    [payload["failure_stage"], payload["attempted_start_index"]],
                    [
                        value["failure"]["stage"],
                        value["failure"]["attempted_start_index"],
                    ],
                )
            final_actor = call["ordinal"] + 1 == context.bound.clock.actor_steps
            maximum = 1 + context.bound.clock.reserve_steps if final_actor else 1
            if (
                not 0 <= len(payload["rows"]) <= maximum
                or not failed
                and not payload["rows"]
            ):
                raise ValueError("native call has missing or extra rows")
            previous_equity = before["book"]["equity"]
            for offset, row in enumerate(payload["rows"]):
                if not normal:
                    break
                before, check = _row(
                    context,
                    row,
                    before,
                    episode=opening["episode_id"],
                    index=call["start_index"] + offset,
                    kind="actor" if offset == 0 else "reserve",
                    action=call["action"] if offset == 0 else None,
                )
                checks.append(check)
                if check["coverage"] == "ECONOMIC_END_DIAGNOSTIC":
                    limits.append("ECONOMIC_END")
                    if offset != len(payload["rows"]) - 1:
                        raise ValueError("native call continued after economic end")
            if not normal:
                break
            if failed:
                _failed_witness(
                    context,
                    payload["settlement_witness"],
                    value["returned_frontier"],
                    checks,
                    episode=opening["episode_id"],
                )
            else:
                terminated = (
                    final_actor or before["book"]["termination_reason"] is not None
                )
                native._same(
                    [payload["reward"], payload["terminated"]],
                    [
                        (before["book"]["equity"] - previous_equity)
                        / context.initial_capital,
                        terminated,
                    ],
                )
                _witness(
                    context, payload["settlement_witness"], before, checks, failed=False
                )
                if terminated and call is not calls[-1]:
                    raise ValueError("native capture continued after termination")
        except _UnsupportedNormalProfile:
            limits.append("PROFILE_UNSUPPORTED")
            normal = False
            break
        except (ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
            if not failed:
                raise ValueError("invalid successful native transition") from error
            limits.append("MALFORMED_FRAGMENT")
            break
    frontier = value["returned_frontier"]
    coverage = (
        "ABSENT" if frontier["encoding_error"] is not None else "UNVALIDATED_TAIL"
    )
    observed_delta = None
    if coverage != "ABSENT":
        index = frontier["index"]
        if not context.start_index <= index <= context.settlement_stop_index:
            raise ValueError("returned frontier exceeds the declared native horizon")
        native._same(frontier["time_ns"], native._time(context, index))
        native._account(context, frontier["native_account"], index)
        episode = opening["episode_id"] if opening is not None else None
        if episode is not None:
            native._same(
                frontier["state_digest"],
                native._state(context, episode, index, frontier["native_account"]),
            )
        if before is not None and index == before["book"]["as_of_index"]:
            native._same(frontier["native_account"], before)
            coverage = "LAST_VALIDATED_ROW" if checks else "UNVALIDATED_TAIL"
        elif before is not None and index < before["book"]["as_of_index"]:
            raise ValueError("returned frontier precedes the validated native prefix")
        observed_delta = (
            frontier["native_account"]["book"]["equity"] - context.initial_capital
        )
    if coverage == "UNVALIDATED_TAIL":
        limits.append("UNVALIDATED_TAIL")
    actors = sum(row["kind"] == "actor" for row in checks)
    all_returned = (
        value["failure"] is None
        and len(calls) == context.bound.clock.actor_steps
        and all(c["outcome"] == "returned" for c in calls)
    )
    full = bool(
        all_returned
        and before is not None
        and actors == context.bound.clock.actor_steps
        and len(checks) == context.bound.clock.physical_horizon_steps
        and coverage == "LAST_VALIDATED_ROW"
        and before["book"]["as_of_index"] == context.settlement_stop_index
        and before["book"]["termination_reason"] is None
        and before["book"]["equity"] > 0.0
        and not before["book"]["insolvent"]
        and not before["active_orders"]
        and all(Fraction(q) == 0 for q in before["book"]["exact_quantities"])
    )
    normal_coverage = bool(
        full and normal and all(row["coverage"] == "NORMAL_RELATIONS" for row in checks)
    )
    eligible = bool(
        full
        and before is not None
        and before["book"]["max_drawdown"] <= context.bound.objective.maximum_drawdown
    )
    if not full:
        limits.append("INCOMPLETE")
    elif not eligible:
        limits.append("DD_INELIGIBLE")
    context.validate_sources()
    return SettledNativeVerification(
        canonical_json_bytes(
            {
                "schema": "allocation_settled_native_verification_v1",
                "context_digest": context.digest,
                "capture_digest": sha256(capture.to_bytes()).hexdigest(),
                "counts": {
                    "attempted_actor_calls": len(calls),
                    "successful_actor_calls": sum(
                        c["outcome"] == "returned" for c in calls
                    ),
                    "validated_actor_rows": actors,
                    "validated_reserve_rows": len(checks) - actors,
                    "validated_native_rows": len(checks),
                },
                "validated_prefix": checks,
                "returned_frontier": frontier,
                "frontier_coverage": coverage,
                "full_structural_settlement": full,
                "normal_phase_coverage": normal_coverage,
                "dd_eligible": eligible,
                "admitted": normal_coverage and eligible,
                "failure": value["failure"],
                "observed_equity_delta": observed_delta,
                "limitations": list(dict.fromkeys(limits)),
            }
        )
    )


def require_settled_native_admission(
    context: SettledNativeValidationContext,
    capture: SettledNativeCapture,
    *,
    actor_intents: tuple[SettledNativeActorIntent, ...],
) -> SettledNativeVerification:
    """Recompute from sources and bytes; never accept a supplied report flag."""
    result = validate_settled_native_capture(
        context, capture, actor_intents=actor_intents
    )
    if not result.admitted:
        raise ValueError("native capture does not establish bounded native admission")
    return result


__all__ = [
    "SettledNativeActorIntent",
    "SettledNativeCapture",
    "SettledNativeValidationContext",
    "SettledNativeVerification",
    "validate_settled_native_capture",
    "require_settled_native_admission",
]
