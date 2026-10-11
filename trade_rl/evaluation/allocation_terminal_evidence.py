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


def _metrics(previous: dict[str, Any], native: dict[str, Any]) -> None:
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
        if name != "funding_pnl" and (value < 0 or value < before - 1e-12):
            raise ValueError("terminal unsigned cumulative counter decreased")
        _equal(value - before, ex[interval])
    for vector, aggregate in (
        ("cost_by_symbol", "interval_cost"),
        ("requested_notional_by_symbol", "requested_notional"),
        ("filled_notional_by_symbol", "filled_notional"),
    ):
        if any(_number(v) < 0 for v in ex[vector]):
            raise ValueError("terminal unsigned symbol aggregate is negative")
        _equal(math.fsum(ex[vector]), ex[aggregate])
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
    common, _ = _plan(plan)
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
        "peak_value": capital,
        "max_drawdown": 0.0,
        **{name: 0 for name in _COUNTERS},
    }
    rewards: list[float] = []
    config: PreTradeRiskConfig | None = None
    for native in accepted:
        config = _risk_config(native["risk_config"], common["risk_digest"])
        decision = native["proposal"]["decision"]
        _equal(decision["baseline"]["context"]["equity"], previous["equity"])
        _equal(decision["max_drawdown"], previous["max_drawdown"])
        expected_weights = _weights(previous, len(native["book"]["quantities"]))
        for a, b in zip(native["current_weights"], expected_weights, strict=True):
            _equal(a, float(b))
        _metrics(previous, native)
        rewards.append((native["book"]["equity"] - previous["equity"]) / capital)
        previous = native["book"]
    prefix_final = None if not accepted else previous
    final, orders = prefix_final, [] if not accepted else accepted[-1]["active_orders"]
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
        _metrics(previous, row["native"])
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
