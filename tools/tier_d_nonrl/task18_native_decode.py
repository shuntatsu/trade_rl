"""Bounded, closed decoding of completed Task18 native evidence.

No replay, fit, cost estimator or mutable BookState is created here. The
decoded immutable facts remain comparison targets for the source-backed oracle.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import Any

from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.decision_value_ledger_oracle import SettledBookView
from trade_rl.evaluation.replay import (
    SharedCashLedgerIntervalEvidence,
    SharedCashPortfolioReplayDecision,
    SharedCashPortfolioReplayLedgerEvidence,
    SharedCashReplayLedgerEvidence,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.simulation.diagnostics.funding import FundingBoundaryEvidence
from trade_rl.simulation.liquidity import SymbolCapacityEvidence
from trade_rl.simulation.orders.model import OrderEvent

_PROTOCOL = build_task18_protocol()
_SYMBOLS = len(_PROTOCOL.dataset.symbols)
_ADVANCES = _PROTOCOL.index_plan.native_advances_per_arm
_ACTOR = _ADVANCES - _PROTOCOL.index_plan.terminal_advance_count


def closed_record(value: object, keys: set[str], *, field: str) -> dict[str, Any]:
    if type(value) is not dict or value.keys() != keys:
        raise ValueError(f"{field} field closure differs")
    return value


def array(value: object, *, field: str, size: int | None = None) -> list[Any]:
    if type(value) is not list or (size is not None and len(value) != size):
        raise ValueError(f"{field} array shape differs")
    return value


def number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite numeric")
    try:
        resolved = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite numeric") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite numeric")
    return resolved


def integer(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative native integer")
    return value


def text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be nonempty text")
    return value


def _quantity_texts(value: object) -> tuple[str, ...]:
    values = tuple(
        text(v, field="exact quantity")
        for v in array(value, field="quantities", size=_SYMBOLS)
    )
    for item in values:
        if len(item) > 1024 or str(Fraction(item)) != item:
            raise ValueError("exact quantity is not bounded canonical rational text")
    return values


def _optional_reason(value: object) -> str | None:
    return None if value is None else text(value, field="termination reason")


def parse_json(raw: bytes) -> object:
    try:
        value = json.loads(raw.decode("utf-8"))
        if canonical_json_bytes(value) != raw:
            raise ValueError("artifact JSON is not exact canonical bytes")
        return value
    except (UnicodeError, RecursionError, OverflowError, TypeError) as exc:
        raise ValueError("invalid native artifact JSON") from exc


def _interval(value: object) -> SharedCashLedgerIntervalEvidence:
    keys = {f.name for f in fields(SharedCashLedgerIntervalEvidence)}
    raw = closed_record(value, keys, field="native interval")
    special = {
        "start_index",
        "next_index",
        "exact_quantities_before",
        "exact_quantities_after",
        "termination_reason",
        "order_events",
        "capacity_events",
        "funding_events",
    }
    args: dict[str, Any] = {
        name: number(raw[name], field=name) for name in keys - special
    }
    for name in ("start_index", "next_index"):
        args[name] = integer(raw[name], field=name)
    for name in ("exact_quantities_before", "exact_quantities_after"):
        args[name] = _quantity_texts(raw[name])
    args["termination_reason"] = _optional_reason(raw["termination_reason"])
    args["order_events"] = tuple(
        OrderEvent.from_mapping(item)
        for item in array(raw["order_events"], field="order events")
    )
    args["funding_events"] = tuple(
        FundingBoundaryEvidence.from_mapping(item)
        for item in array(raw["funding_events"], field="funding events")
    )
    cap_keys = {f.name for f in fields(SymbolCapacityEvidence)}
    args["capacity_events"] = tuple(
        SymbolCapacityEvidence(
            **{
                name: number(item, field=name)
                for name, item in closed_record(
                    event, cap_keys, field="capacity event"
                ).items()
            }
        )
        for event in array(raw["capacity_events"], field="capacity events")
    )
    return SharedCashLedgerIntervalEvidence(**args)


def _decision(value: object) -> SharedCashPortfolioReplayDecision:
    raw = closed_record(
        value,
        {f.name for f in fields(SharedCashPortfolioReplayDecision)},
        field="portfolio decision",
    )

    def weights(name: str) -> tuple[float, ...]:
        return tuple(
            number(x, field=name) for x in array(raw[name], field=name, size=_SYMBOLS)
        )

    def holds(name: str) -> tuple[int, ...]:
        result = tuple(integer(x, field=name) for x in array(raw[name], field=name))
        if len(set(result)) != len(result) or any(i >= _SYMBOLS for i in result):
            raise ValueError("invalid native hold index")
        return result

    return SharedCashPortfolioReplayDecision(
        index=integer(raw["index"], field="decision index"),
        proposal_weights=weights("proposal_weights"),
        target_weights=weights("target_weights"),
        risk_reasons=tuple(
            text(x, field="risk reason")
            for x in array(raw["risk_reasons"], field="risk reasons")
        ),
        requested_hold_symbols=holds("requested_hold_symbols"),
        effective_hold_symbols=holds("effective_hold_symbols"),
    )


def decode_ledger(raw_bytes: bytes) -> SharedCashPortfolioReplayLedgerEvidence:
    keys = {f.name for f in fields(SharedCashReplayLedgerEvidence)} - {"decisions"}
    raw = closed_record(
        parse_json(raw_bytes), keys | {"portfolio_decisions"}, field="portfolio ledger"
    )
    if raw["schema_version"] != "shared_cash_portfolio_replay_ledger_v1":
        raise ValueError("unsupported native ledger schema")
    if array(raw["active_order_remainders"], field="active orders"):
        raise ValueError("completed ledger cannot contain active orders")
    special = {
        "dataset_id",
        "execution_policy_digest",
        "start_index",
        "stop_index",
        "intervals",
        "terminal_exact_quantities",
        "termination_reason",
        "active_order_remainders",
        "terminal_order_reasons",
        "schema_version",
    }
    args: dict[str, Any] = {
        name: number(raw[name], field=name) for name in keys - special
    }
    for name in ("dataset_id", "execution_policy_digest"):
        args[name] = text(raw[name], field=name)
    for name in ("start_index", "stop_index"):
        args[name] = integer(raw[name], field=name)
    args.update(
        intervals=tuple(
            _interval(v)
            for v in array(raw["intervals"], field="intervals", size=_ADVANCES)
        ),
        terminal_exact_quantities=_quantity_texts(raw["terminal_exact_quantities"]),
        termination_reason=_optional_reason(raw["termination_reason"]),
        active_order_remainders=(),
        terminal_order_reasons=tuple(
            tuple(
                text(x, field="terminal order reason")
                for x in array(pair, field="terminal order pair", size=2)
            )
            for pair in array(
                raw["terminal_order_reasons"], field="terminal order reasons"
            )
        ),
    )
    decoded = SharedCashPortfolioReplayLedgerEvidence(
        SharedCashReplayLedgerEvidence(**args),
        tuple(
            _decision(v)
            for v in array(
                raw["portfolio_decisions"], field="portfolio decisions", size=_ACTOR
            )
        ),
    )
    if canonical_json_bytes(decoded.to_mapping()) != raw_bytes:
        raise ValueError("decoded native ledger differs from exact source bytes")
    return decoded


@dataclass(frozen=True, slots=True)
class TerminalNativeFacts:
    quantity_texts: tuple[str, ...]
    cash: float
    portfolio_value: float
    total_cost: float
    funding_pnl: float
    borrow_cost: float
    turnover_total: float
    max_drawdown: float
    termination_reason: str | None
    insolvent: bool

    @property
    def exact_quantities(self) -> tuple[Fraction, ...]:
        return tuple(Fraction(q) for q in self.quantity_texts)

    @classmethod
    def observe(cls, book: SettledBookView) -> TerminalNativeFacts:
        return cls(
            tuple(str(q) for q in book.exact_quantities),
            float(book.cash),
            float(book.portfolio_value),
            float(book.total_cost),
            float(book.funding_pnl),
            float(book.borrow_cost),
            float(book.turnover_total),
            float(book.max_drawdown),
            book.termination_reason,
            bool(book.insolvent),
        )

    @classmethod
    def from_mapping(cls, value: object) -> TerminalNativeFacts:
        keys = {f.name for f in fields(cls)}
        raw = closed_record(value, keys, field="terminal facts")
        if type(raw["insolvent"]) is not bool:
            raise ValueError("terminal insolvent flag must be boolean")
        special = {"quantity_texts", "termination_reason", "insolvent"}
        return cls(
            quantity_texts=_quantity_texts(raw["quantity_texts"]),
            termination_reason=_optional_reason(raw["termination_reason"]),
            insolvent=raw["insolvent"],
            **{name: number(raw[name], field=name) for name in keys - special},
        )


@dataclass(frozen=True, slots=True)
class RestoredNativeEvidence:
    book: TerminalNativeFacts
    returns: ReturnSeries
    decisions: tuple[SharedCashPortfolioReplayDecision, ...]
    ledger_evidence: SharedCashPortfolioReplayLedgerEvidence


def decode_returns(value: object) -> ReturnSeries:
    raw = closed_record(
        value, {"values", "kind", "periods_per_year", "elapsed_years"}, field="returns"
    )
    if raw["kind"] != ReturnKind.BASE_BAR.value:
        raise ValueError("completed native arm requires base-bar returns")
    return ReturnSeries(
        values=tuple(
            number(v, field="return")
            for v in array(raw["values"], field="returns", size=_ADVANCES)
        ),
        kind=ReturnKind.BASE_BAR,
        periods_per_year=integer(raw["periods_per_year"], field="annualization"),
        elapsed_years=None
        if raw["elapsed_years"] is None
        else number(raw["elapsed_years"], field="elapsed years"),
    )
