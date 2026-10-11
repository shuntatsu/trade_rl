"""Closed returned-state validation, without execution or research admission.

Visible book/order continuity and bounded algebra do not authenticate market
inputs, cost decomposition, RNG, intrabar DD or complete terminal fill archives.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_global_execution import (
    GlobalAllocationExecutionReceipt,
)
from trade_rl.evaluation.allocation_terminal_execution import (
    TerminalAllocationExecutionPlan,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    _construct,
    _json,
    terminal_allocation_state_digest,
    validate_allocation_execution_cost_payload,
    validate_terminal_allocation_facts,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.orders.model import OrderStatus, _quantity_tolerance
from trade_rl.strategies.rl.allocation_preprocessing import _native_json

_FIELDS = {
    "schema",
    "plan",
    "plan_digest",
    "status",
    "native_receipt",
    "closing_opening_state_digest",
    "closing_rows",
    "final_book",
    "final_active_orders",
    "full_close_coverage",
    "settlement_complete",
    "risk_eligible",
    "marked_prefix_profit_rate",
    "observed_profit_rate",
    "settled_profit_rate",
    "fixed_capital_rewards",
    "fixed_capital_reward_sum",
    "attempted_start_index",
    "failure",
    "failure_stage",
    "claim_scope",
}
_ROW = {
    "start_index",
    "processing_index",
    "processing_time_ns",
    "clock_valid",
    "returned_clock_types",
    "opening_equity",
    "risk_target",
    "risk_reasons",
    "risk",
    "before_state_digest",
    "after_state_digest",
    "native",
    "fixed_capital_reward",
}
_COUNTERS = {
    "turnover_total": "filled_turnover",
    "total_cost": "interval_cost",
    "funding_pnl": "interval_funding",
    "borrow_cost": "interval_borrow_cost",
    "fill_count": "fill_count",
    "rebalance_events": "rebalance_events",
}


def _closed(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"terminal {name} has missing or undeclared fields")
    return value


def _number(value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or type(value) not in (int, float)
        or not math.isfinite(value)
    ):
        raise ValueError("terminal financial value must be finite numeric content")
    return float(value)


def _equal(actual: object, expected: object) -> None:
    if not math.isclose(
        _number(actual), _number(expected), rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError("terminal financial value differs from returned-state algebra")


def _same(actual: object, expected: object) -> None:
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise ValueError(
            "terminal content differs from its declared or reconstructed value"
        )


def _timestamp(value: datetime) -> int:
    d = value - datetime(1970, 1, 1, tzinfo=UTC)
    return (d.days * 86400 + d.seconds) * 10**9 + d.microseconds * 1000


def _plan(
    plan: TerminalAllocationExecutionPlan,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(plan) is not TerminalAllocationExecutionPlan:
        raise ValueError(
            "terminal evidence requires the separately declared exact plan"
        )
    plan.__post_init__()
    common, cell = plan.native_plan.payload["common"], plan.native_plan.payload["cell"]
    if (
        _timestamp(plan.bound.objective.evaluation_start)
        != common["source_clocks_ns"][0]
        or _timestamp(plan.bound.objective.evaluation_stop_exclusive)
        != plan.closing_clocks_ns[-1]
    ):
        raise ValueError(
            "terminal objective endpoints differ from physical source clocks"
        )
    validate_allocation_execution_cost_payload(
        cell["runtime"]["cost_payload"], economics_digest=common["economics_digest"]
    )
    return common, cell


def _weights(book: dict[str, Any], size: int) -> np.ndarray:
    if "quantities" not in book or book["equity"] <= 0:
        return np.zeros(size)
    return np.array(
        [
            q * p * m / book["equity"]
            for q, p, m in zip(
                book["quantities"],
                book["mark_prices"],
                book["contract_multipliers"],
                strict=True,
            )
        ]
    )


def _retained_links(
    native: dict[str, Any], cost: dict[str, Any], symbol_index: int
) -> None:
    """Join returned interval facts without reconstructing an execution path."""
    book, ex = native["book"], native["execution"]
    fills = [
        event
        for event in native["order_events"]
        if event["event_type"] in ("filled", "partial_fill")
    ]
    _equal(math.fsum(e["filled_notional"] for e in fills), ex["filled_notional"])
    for index, total in enumerate(ex["filled_notional_by_symbol"]):
        _equal(
            math.fsum(
                e["filled_notional"] for e in fills if e["symbol_index"] == index
            ),
            total,
        )
        if not any(e["symbol_index"] == index for e in fills):
            _equal(ex["cost_by_symbol"][index], 0.0)
        _equal(
            ex["participation_by_symbol"][index],
            max(
                (e["participation_rate"] for e in fills if e["symbol_index"] == index),
                default=0.0,
            ),
        )
    _equal(
        ex["max_participation"],
        max((e["participation_rate"] for e in fills), default=0.0),
    )
    _same(ex["rebalance_events"], ex["fill_count"])
    for event in fills:
        quantity = _number(event["filled_quantity"])
        if abs(quantity) <= 1e-12 or quantity * event["requested_quantity"] <= 0:
            raise ValueError("terminal fill quantity differs from its order direction")
        _equal(
            event["filled_notional"],
            abs(quantity)
            * _number(event["execution_price"])
            * book["contract_multipliers"][event["symbol_index"]],
        )
    latest = {event["order_id"]: event for event in native["order_events"]}
    active = {order["intent"]["order_id"] for order in native["active_orders"]}
    if any(
        not OrderStatus(event["new_status"]).terminal and order_id not in active
        for order_id, event in latest.items()
    ):
        raise ValueError("terminal latest nonterminal event lost its active order")
    if len(ex["funding_evidence"]) > 1:
        raise ValueError("terminal one-bar interval has duplicate funding boundaries")
    if not book["insolvent"] and book["termination_reason"] is None:
        for boundary in ex["funding_evidence"]:
            _equal(boundary["equity_after_funding"], book["equity"])
            for field, book_field in (
                ("signed_quantities", "quantities"),
                ("mark_prices", "mark_prices"),
                ("contract_multipliers", "contract_multipliers"),
            ):
                for actual, expected in zip(
                    boundary[field], book[book_field], strict=True
                ):
                    _equal(actual, expected)
    if any(
        _number(value) < 0
        for value in (
            ex["unfilled_turnover"],
            ex["max_participation"],
            *ex["participation_by_symbol"],
        )
    ):
        raise ValueError("terminal native participation/turnover is negative")
    if ex["requested_notional"] <= 1e-12:
        _equal(ex["fill_ratio"], 1.0)
    if len(ex["capacity_evidence"]) > 1 or fills and not ex["capacity_evidence"]:
        raise ValueError("terminal independent interval requires one capacity pool")
    for capacity in ex["capacity_evidence"]:
        _same(
            capacity["contract_multiplier"],
            book["contract_multipliers"][symbol_index],
        )
        if capacity["participation_limit"] > cost["max_participation_rate"]:
            raise ValueError(
                "terminal capacity exceeds the declared participation policy"
            )
        _equal(
            capacity["initial_capacity_notional"],
            capacity["market_notional"] * capacity["participation_limit"],
        )
        _equal(
            capacity["initial_capacity_notional"],
            capacity["consumed_capacity_notional"]
            + capacity["remaining_capacity_notional"],
        )
        remaining = capacity["initial_capacity_notional"]
        for event in fills:
            _same(event["capacity_before"], remaining)
            remaining = max(0.0, remaining - event["filled_notional"])
            _same(event["capacity_after"], remaining)
        _same(capacity["remaining_capacity_notional"], remaining)
        if capacity["market_notional"] > 1e-12:
            for event in fills:
                ratio = event["filled_notional"] / capacity["market_notional"]
                if event["participation_rate"] < ratio and not math.isclose(
                    event["participation_rate"], ratio, rel_tol=1e-10, abs_tol=1e-12
                ):
                    raise ValueError(
                        "terminal fill participation is below its notional ratio"
                    )
    capacity_tolerance = math.fsum(
        max(
            1e-9,
            8 * math.ulp(c["initial_capacity_notional"]),
            8 * math.ulp(c["remaining_capacity_notional"]),
        )
        for c in ex["capacity_evidence"]
    )
    if not math.isclose(
        math.fsum(c["consumed_capacity_notional"] for c in ex["capacity_evidence"]),
        ex["filled_notional"],
        rel_tol=1e-10,
        abs_tol=capacity_tolerance,
    ):
        raise ValueError("terminal consumed capacity differs from returned fills")


def _order_continuity(
    previous_orders: list[dict[str, Any]], native: dict[str, Any]
) -> None:
    before = {order["intent"]["order_id"]: order for order in previous_orders}
    after = {order["intent"]["order_id"]: order for order in native["active_orders"]}
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in native["order_events"]:
        groups.setdefault(event["order_id"], []).append(event)
    fixed = (
        "dataset_id",
        "execution_policy_digest",
        "symbol_index",
        "requested_quantity",
        "replaced_order_id",
    )
    for order_id, events in groups.items():
        prior = before.get(order_id)
        if prior is None and events[0]["event_type"] != "submitted":
            raise ValueError("terminal new order has no native submission event")
        known = prior if prior is not None else after.get(order_id)
        intent = events[0] if known is None else known["intent"]
        status = "submitted" if prior is None else prior["status"]
        requested = _number(intent["requested_quantity"])
        remaining = requested if prior is None else _number(prior["remaining_quantity"])
        cumulative = (
            0.0 if prior is None else _number(prior["cumulative_filled_quantity"])
        )
        for event in events:
            if OrderStatus(status).terminal:
                raise ValueError("terminal order changed after its terminal event")
            for field in fixed:
                _same(event[field], intent[field])
            # Native canonical payloads omit this flag when it is false.
            _same(event.get("reduce_only", False), intent.get("reduce_only", False))
            _same(event["previous_status"], status)
            quantity = _number(event["filled_quantity"])
            tolerance = _quantity_tolerance(requested, cumulative, remaining, quantity)
            expected = remaining
            if event["event_type"] in ("filled", "partial_fill"):
                if (
                    quantity * remaining <= 0.0
                    or abs(quantity) > abs(remaining) + tolerance
                ):
                    raise ValueError(
                        "terminal fill exceeds its known remaining quantity"
                    )
                expected -= quantity
                cumulative += quantity
            actual = _number(event["remaining_quantity"])
            tolerance = max(
                tolerance, _quantity_tolerance(requested, cumulative, expected, actual)
            )
            if event["new_status"] == "filled":
                _same(event["remaining_quantity"], 0.0)
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(
                    "terminal event remaining quantity differs from its fills"
                )
            remaining = actual
            status = event["new_status"]
    for order_id, active in after.items():
        prior = before.get(order_id)
        events = groups.get(order_id, [])
        if prior is None and not events:
            raise ValueError("terminal new active order has no submission event")
        cumulative = 0.0 if prior is None else prior["cumulative_filled_notional"]
        version = 0 if prior is None else prior["evidence_version"]
        # Match native sequential addition; fsum has different rounding.
        for event in events:
            if event["event_type"] in ("filled", "partial_fill"):
                cumulative += event["filled_notional"]
            if event["event_type"] not in ("submitted", "no_fill"):
                version += 1
        _same(active["cumulative_filled_notional"], cumulative)
        _same(active["evidence_version"], version)
    for order_id, old in before.items():
        if order_id in after:
            _same(after[order_id]["intent"], old["intent"])
            if all(e["event_type"] == "no_fill" for e in groups.get(order_id, ())):
                _same(after[order_id], old)


def _interval_continuity(
    previous: dict[str, Any],
    previous_orders: list[dict[str, Any]],
    native: dict[str, Any],
) -> None:
    """Conserve known order identities and cash in intervals without inventory."""
    book, ex = native["book"], native["execution"]
    active = {order["intent"]["order_id"] for order in native["active_orders"]}
    terminals = {
        event["order_id"]
        for event in native["order_events"]
        if OrderStatus(event["new_status"]).terminal
    }
    if any(
        order["intent"]["order_id"] not in active | terminals
        for order in previous_orders
    ):
        raise ValueError(
            "terminal carried active order disappeared without termination"
        )
    _order_continuity(previous_orders, native)
    # An initially flat account without fills cannot acquire inventory through
    # splits, delisting or termination flattening; those operations preserve zero.
    if all(q == "0" for q in previous["exact_quantities"]) and not any(
        event["event_type"] in ("filled", "partial_fill")
        for event in native["order_events"]
    ):
        if any(q != "0" for q in book["exact_quantities"]):
            raise ValueError("terminal flat account acquired inventory without fills")
        if any(
            ex[field] != 0.0
            for field in (
                "interval_cost",
                "interval_funding",
                "interval_borrow_cost",
                "interval_dividend",
            )
        ):
            raise ValueError("terminal flat no-fill account has an inventory cash flow")
        cash, interest = (
            _number(previous["cash"]),
            _number(ex["interval_cash_interest"]),
        )
        expected = cash + interest
        # Gap and processing interest are applied separately but returned as a
        # rounded sum. Cancellation can amplify their ULP error at the endpoint.
        tolerance = 8 * math.fsum(
            math.ulp(value) for value in (cash, interest, expected)
        )
        if interest == 0.0 and _number(book["cash"]) != cash:
            raise ValueError("terminal zero-interest flat cash changed")
        if not math.isclose(
            _number(book["cash"]), expected, rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError("terminal flat cash differs from recorded interest")


def _book_links(
    previous: dict[str, Any],
    native: dict[str, Any],
    cost: dict[str, Any],
    symbol_index: int,
) -> None:
    book = native["book"]
    if "contract_multipliers" in previous:
        _same(book["contract_multipliers"], previous["contract_multipliers"])
    if any(
        q != "0"
        for index, q in enumerate(book["exact_quantities"])
        if index != symbol_index
    ):
        raise ValueError("terminal independent account holds a foreign symbol")
    if book["termination_reason"] is not None:
        for field in ("margin_used", "maintenance_margin", "maintenance_requirement"):
            _equal(book[field], 0.0)
    else:
        gross = float(
            np.abs(
                np.array(book["quantities"])
                * np.array(book["mark_prices"])
                * np.array(book["contract_multipliers"])
            ).sum()
        )
        _equal(book["margin_used"], gross / cost["max_leverage"])
        _equal(book["maintenance_margin"], cost["maintenance_margin_rate"])
        _equal(book["maintenance_requirement"], gross * cost["maintenance_margin_rate"])
        values = (
            np.array(book["quantities"])
            * np.array(book["mark_prices"])
            * np.array(book["contract_multipliers"])
        )
        requirement = gross * cost["maintenance_margin_rate"]
        collateral = float(
            book["cash"]
            + np.minimum(values, 0.0).sum()
            + cost["collateral_haircut"] * np.maximum(values, 0.0).sum()
        )
        deficit = max(
            0.0, requirement - max(book["equity"], 0.0), requirement - collateral
        )
        if cost["margin_mode"] == "isolated" and gross > 0.0:
            deficit = max(
                deficit,
                float(
                    np.max(
                        np.maximum(
                            cost["maintenance_margin_rate"] * np.abs(values)
                            - collateral * (np.abs(values) / gross),
                            0.0,
                        ),
                        initial=0.0,
                    )
                ),
            )
        _equal(book["margin_deficit"], deficit)


def _metrics(
    previous: dict[str, Any],
    native: dict[str, Any],
    *,
    previous_orders: list[dict[str, Any]],
    cost: dict[str, Any],
    symbol_index: int,
) -> None:
    book, ex = native["book"], native["execution"]
    equity, peak, dd = (
        _number(book[k]) for k in ("equity", "peak_value", "max_drawdown")
    )
    if (
        peak < previous["peak_value"] - 1e-12
        or peak < max(equity, 0) - 1e-12
        or dd < previous["max_drawdown"] - 1e-12
        or dd < 1 - max(equity, 0) / peak - 1e-12
        or not 0 <= dd <= 1
    ):
        raise ValueError("terminal cumulative peak/drawdown is inconsistent")
    if previous["equity"] <= 0:
        raise ValueError("terminal cannot continue a nonpositive account")
    net = max(max(equity, 0) / previous["equity"] - 1, -1 + 1e-12)
    _equal(ex["interval_net_return"], net)
    _equal(ex["interval_log_return"], math.log1p(net))
    gross = (
        max(equity, 0)
        + ex["interval_cost"]
        - ex["interval_funding"]
        + ex["interval_borrow_cost"]
        - ex["interval_dividend"]
        - ex["interval_cash_interest"]
    ) / previous["equity"] - 1
    _equal(ex["interval_gross_return"], gross)
    for name, interval in _COUNTERS.items():
        value, before = _number(book[name]), _number(previous[name])
        increment = _number(ex[interval])
        if name != "funding_pnl" and (value < 0 or value < before or increment < 0):
            raise ValueError("terminal unsigned cumulative counter decreased")
        if name in ("fill_count", "rebalance_events"):
            _same(book[name], previous[name] + ex[interval])
        else:
            expected = before + increment
            additions = sum(
                e["event_type"] in ("filled", "partial_fill")
                for e in native["order_events"]
            )
            tolerance = (8 + 2 * additions) * math.fsum(
                math.ulp(x) for x in (before, increment, expected)
            )
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError("terminal cumulative value differs from its interval")
    for vector, aggregate in (
        ("cost_by_symbol", "interval_cost"),
        ("requested_notional_by_symbol", "requested_notional"),
        ("filled_notional_by_symbol", "filled_notional"),
    ):
        if any(_number(v) < 0 for v in ex[vector]):
            raise ValueError("terminal unsigned symbol aggregate is negative")
        _equal(math.fsum(ex[vector]), ex[aggregate])
    turnover_denominator = max(previous["equity"], 1e-12)
    _equal(ex["filled_turnover"], ex["filled_notional"] / turnover_denominator)
    _equal(ex["requested_turnover"], ex["requested_notional"] / turnover_denominator)
    _equal(
        math.fsum(e["funding_amount"] for e in ex["funding_evidence"]),
        ex["interval_funding"],
    )
    for key, events in (
        ("fill_count", ("filled", "partial_fill")),
        ("completed_fill_count", ("filled",)),
        ("rejected_count", ("rejected",)),
        ("expired_count", ("expired",)),
    ):
        _same(ex[key], sum(e["event_type"] in events for e in native["order_events"]))
    _retained_links(native, cost, symbol_index)
    _interval_continuity(previous, previous_orders, native)
    _book_links(previous, native, cost, symbol_index)


def _risk_config(value: object, digest: str) -> PreTradeRiskConfig:
    config = _construct(PreTradeRiskConfig, value)
    if config.max_turnover is not None:
        _number(
            config.max_turnover
        )  # Optional annotations escape legacy scalar checks.
    _same(content_digest(config), digest)
    return config


def _receipt(value: object, plan: TerminalAllocationExecutionPlan) -> dict[str, Any]:
    _native_json(value)
    raw = _closed(value, _FIELDS, "receipt")
    _same(raw["schema"], "allocation_terminal_runtime_observation_v2")
    common, cell = _plan(plan)
    cost = cell["runtime"]["cost_payload"]
    _same(raw["plan"], plan.payload)
    _same(raw["plan_digest"], plan.digest)
    prefix = GlobalAllocationExecutionReceipt.from_payload(
        raw["native_receipt"]
    ).payload
    _same(prefix["plan"], plan.native_plan.payload)
    accepted = [
        r["native"]
        for r in prefix["rows"]
        if r["native"] is not None and r["native_validation_failure"] is None
    ]
    capital = common["initial_capital"]
    previous: dict[str, Any] = {
        "equity": capital,
        "cash": capital,
        "exact_quantities": ["0"] * len(accepted[0]["book"]["quantities"])
        if accepted
        else [],
        "peak_value": capital,
        "max_drawdown": 0.0,
        **{name: 0 for name in _COUNTERS},
    }
    rewards: list[float] = []
    config: PreTradeRiskConfig | None = None
    orders: list[dict[str, Any]] = []
    for native in accepted:
        config = _risk_config(native["risk_config"], common["risk_digest"])
        decision = native["proposal"]["decision"]
        _equal(decision["baseline"]["context"]["equity"], previous["equity"])
        _equal(decision["max_drawdown"], previous["max_drawdown"])
        expected_weights = _weights(previous, len(native["book"]["quantities"]))
        for a, b in zip(native["current_weights"], expected_weights, strict=True):
            _equal(a, float(b))
        _metrics(
            previous,
            native,
            previous_orders=orders,
            cost=cost,
            symbol_index=common["symbol_index"],
        )
        rewards.append((native["book"]["equity"] - previous["equity"]) / capital)
        previous = native["book"]
        orders = native["active_orders"]
    prefix_final = None if not accepted else previous
    final = prefix_final
    rows = raw["closing_rows"]
    if type(rows) is not list or len(rows) > len(plan.closing_clocks_ns) - 1:
        raise ValueError("terminal close rows exceed the reserved window")
    if rows and (prefix["status"] != "completed" or config is None):
        raise ValueError(
            "terminal closing requires a completed validated native prefix"
        )
    if raw["closing_opening_state_digest"] is not None:
        _same(raw["closing_opening_state_digest"], prefix["closing_state_digest"])
    if prefix["status"] == "completed" and raw["closing_opening_state_digest"] is None:
        raise ValueError("terminal prefix opening bridge is missing")
    for offset, item in enumerate(rows):
        row = _closed(item, _ROW, "close row")
        index = common["range"][1] + offset
        for name, expected in (
            ("start_index", index),
            ("processing_index", index + 1),
            ("processing_time_ns", plan.closing_clocks_ns[offset + 1]),
        ):
            if type(row[name]) is not int or row[name] != expected:
                raise ValueError(
                    "terminal close clock differs from its reserved physical bar"
                )
        _same(
            row["returned_clock_types"], {"next_index": "int", "bars_advanced": "int"}
        )
        _same(row["clock_valid"], True)
        _equal(row["opening_equity"], previous["equity"])
        _same(
            row["before_state_digest"],
            terminal_allocation_state_digest(
                common,
                plan.closing_source_digest,
                {"book": previous, "active_orders": orders},
                index=index,
                time_ns=plan.closing_clocks_ns[offset],
            ),
        )
        book = validate_terminal_allocation_facts(
            row["native"],
            dataset_id=common["dataset_id"],
            economics_digest=common["economics_digest"],
            symbol_index=common["symbol_index"],
            start_index=index,
            start_time_ns=plan.closing_clocks_ns[offset],
            processing_index=index + 1,
            processing_time_ns=plan.closing_clocks_ns[offset + 1],
        )
        assert config is not None
        risk = PreTradeRisk(config).constrain(
            np.zeros(len(book.quantities)),
            current=_weights(previous, len(book.quantities)),
            drawdown=previous["max_drawdown"],
        )
        _same(row["risk"], _json(risk))
        _same(row["risk_target"], risk.weights.tolist())
        _same(row["risk_reasons"], list(risk.reasons))
        _same(
            row["after_state_digest"],
            terminal_allocation_state_digest(
                common,
                plan.closing_source_digest,
                row["native"],
                index=index + 1,
                time_ns=plan.closing_clocks_ns[offset + 1],
            ),
        )
        _metrics(
            previous,
            row["native"],
            previous_orders=orders,
            cost=cost,
            symbol_index=common["symbol_index"],
        )
        reward = (book.portfolio_value - previous["equity"]) / capital
        _equal(row["fixed_capital_reward"], reward)
        rewards.append(reward)
        previous = final = row["native"]["book"]
        orders = row["native"]["active_orders"]
        if final["termination_reason"] is not None and offset != len(rows) - 1:
            raise ValueError("terminal rows continue after economic termination")
    failure = raw["failure"]
    if failure is not None:
        failure = _closed(failure, {"error_type", "message"}, "failure")
        if (
            type(failure["error_type"]) is not str
            or not failure["error_type"]
            or type(failure["message"]) is not str
        ):
            raise ValueError(
                "terminal failure requires the original exception diagnostic"
            )
        stage = raw["failure_stage"]
        if prefix["status"] != "completed" and stage != "prefix":
            raise ValueError("terminal closing failure has no completed native prefix")
        if stage == "prefix" and prefix["failure"] is not None:
            _same(failure, prefix["failure"])
        if stage not in (
            "prefix",
            "closing_preflight",
            "closing_execution",
            "closing_returned",
            "closing_validation",
            "final_validation",
        ):
            raise ValueError("terminal failure stage is undeclared")
        expected_attempt = (
            None
            if stage == "prefix"
            else common["range"][1] + len(rows)
            if stage in ("closing_preflight", "closing_execution", "closing_returned")
            else common["range"][1] + len(rows) - 1
        )
        _same(raw["attempted_start_index"], expected_attempt)
    else:
        _same(raw["failure_stage"], None)
        _same(raw["attempted_start_index"], None)
    stopped = (final is not None and final["termination_reason"] is not None) or prefix[
        "status"
    ] == "economic_stop"
    status = (
        "economic_stop"
        if stopped
        else "integrity_failure"
        if failure is not None
        else "completed"
    )
    _same(raw["status"], status)
    coverage = (
        prefix["status"] == "completed" and len(rows) == len(plan.closing_clocks_ns) - 1
    )
    if failure is None and not stopped and not coverage:
        raise ValueError("terminal success omits reserved processing bars")
    complete = (
        failure is None
        and coverage
        and not stopped
        and final is not None
        and final["equity"] > 0
        and not final["insolvent"]
        and all(q == "0" for q in final["exact_quantities"])
        and not orders
    )
    profit = None if final is None else (final["equity"] - capital) / capital
    expected = {
        "final_book": final,
        "final_active_orders": orders,
        "full_close_coverage": coverage,
        "settlement_complete": complete,
        "risk_eligible": complete
        and final is not None
        and final["max_drawdown"] <= plan.bound.objective.maximum_drawdown,
        "marked_prefix_profit_rate": None
        if prefix_final is None
        else (prefix_final["equity"] - capital) / capital,
        "observed_profit_rate": profit,
        "settled_profit_rate": profit if complete else None,
        "claim_scope": "bounded_native_software_observation_no_economic_qualification",
    }
    for name, actual in expected.items():
        if type(actual) is float:
            _equal(raw[name], actual)
        else:
            _same(raw[name], actual)
    if type(raw["fixed_capital_rewards"]) is not list or len(
        raw["fixed_capital_rewards"]
    ) != len(rewards):
        raise ValueError("terminal signed rewards omit a returned transition")
    for actual, expected_reward in zip(
        raw["fixed_capital_rewards"], rewards, strict=True
    ):
        _equal(actual, expected_reward)
    _equal(raw["fixed_capital_reward_sum"], math.fsum(rewards))
    if profit is not None:
        _equal(math.fsum(rewards), profit)
    return raw


@dataclass(frozen=True, slots=True)
class TerminalAllocationExecutionReceipt:
    """Externally plan-bound canonical observation; no authenticity or approval."""

    expected_plan: TerminalAllocationExecutionPlan
    _bytes: bytes

    def __post_init__(self) -> None:
        if type(self._bytes) is not bytes:
            raise ValueError("terminal evidence requires canonical detached bytes")
        raw = _receipt(json.loads(self._bytes), self.expected_plan)
        if canonical_json_bytes(raw) != self._bytes:
            raise ValueError("terminal receipt bytes are not canonical")

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._bytes)

    @property
    def digest(self) -> str:
        return content_digest(self.payload)

    @classmethod
    def from_payload(
        cls, value: object, *, expected_plan: TerminalAllocationExecutionPlan
    ) -> TerminalAllocationExecutionReceipt:
        try:
            raw = _receipt(value, expected_plan)
            return cls(expected_plan, canonical_json_bytes(raw))
        except (
            KeyError,
            TypeError,
            OverflowError,
            RecursionError,
            IndexError,
        ) as error:
            raise ValueError("invalid closed terminal evidence") from error


__all__ = ["TerminalAllocationExecutionReceipt"]
