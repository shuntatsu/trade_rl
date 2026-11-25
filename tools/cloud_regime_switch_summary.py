"""Read-only integrity audit and fixed decision rule for the switch screen.

Consumes only authorized, previously recorded native replay receipts.
No new account, order simulator, parameter selection, or market data loader.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.divergence_cloud_diagnostics import (
    attribute_single_symbol_ledger,
)
from trade_rl.simulation.orders.model import OrderEvent
from trade_rl.simulation.quantities import parse_quantity

CellKey = tuple[str, int, str, str]


def roster(protocol: dict) -> tuple[CellKey, ...]:
    return tuple(
        (symbol, year, arm, scenario)
        for symbol in protocol["symbols"]
        for year in protocol["years"]
        for arm in protocol["arms"]
        for scenario in protocol["scenarios"]
    )


def cell_name(key: CellKey) -> str:
    return "_".join(map(str, key))


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite numeric")
    return number


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_object(raw: bytes) -> dict:
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("receipt is not a canonical JSON object")
    return value


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or nonregular receipt")
    return _canonical_object(path.read_bytes())


def _verify_native_flows(ledger: dict, intervals: list[dict]) -> None:
    # Native observer totals are cumulative from a fresh account. Reconcile
    # the redundant evidence without recomputing fills, funding or turnover.
    for total, interval_field in (
        ("total_cost", "interval_cost"),
        ("funding_pnl", "interval_funding"),
        ("borrow_cost", "interval_borrow_cost"),
        ("turnover_total", None),
    ):
        previous = 0.0
        flows: list[float] = []
        for row in intervals:
            before = _finite(row[total + "_before"], total)
            after = _finite(row[total + "_after"], total)
            if not math.isclose(before, previous, abs_tol=1e-7, rel_tol=1e-10):
                raise ValueError("native flow discontinuity: " + total)
            # Cost, borrow and turnover are cumulative nonnegative native
            # magnitudes; funding alone is signed.
            if total != "funding_pnl" and (before < 0 or after < before):
                raise ValueError("native flow outside cumulative bounds: " + total)
            if interval_field is not None:
                flow = _finite(row[interval_field], interval_field)
                if not math.isclose(after - before, flow, abs_tol=1e-7, rel_tol=1e-10):
                    raise ValueError("native flow increment mismatch: " + total)
                flows.append(flow)
            previous = after
        final = _finite(ledger["final_" + total], total)
        if not math.isclose(final, previous, abs_tol=1e-7, rel_tol=1e-10) or (
            interval_field is not None
            and not math.isclose(final, math.fsum(flows), abs_tol=1e-7, rel_tol=1e-10)
        ):
            raise ValueError("native flow terminal mismatch: " + total)


def _verify_native_terminal_trace(ledger: dict, intervals: list[dict]) -> None:
    # The frozen replay uses a fresh account and the native floor-zero
    # insolvency convention. Verify redundant observations, not fills/P&L.
    first, last = intervals[0], intervals[-1]
    prior_cash = 10000.0
    if first["exact_quantities_before"] != ["0"]:
        raise ValueError("native terminal trace has nonflat initial inventory")
    for i, row in enumerate(intervals):
        before = _finite(row["cash_before"], "cash before")
        if not math.isclose(before, prior_cash, abs_tol=1e-7, rel_tol=1e-10):
            raise ValueError("native cash discontinuity")
        prior_cash = _finite(row["cash_after"], "cash after")
        start = _finite(row["portfolio_value_before"], "interval start equity")
        end = _finite(row["portfolio_value_after"], "interval end equity")
        observed = _finite(row["interval_net_return"], "interval return")
        if start <= 0 or not math.isclose(
            observed,
            max(max(end, 0.0) / start - 1.0, -1.0 + 1e-12),
            abs_tol=1e-12,
            rel_tol=1e-10,
        ):
            raise ValueError("native interval return mismatch")
        if i != len(intervals) - 1 and row["termination_reason"] is not None:
            raise ValueError("native terminal reason before final interval")
    if not math.isclose(
        _finite(ledger["final_cash"], "final cash"),
        prior_cash,
        abs_tol=1e-7,
        rel_tol=1e-10,
    ):
        raise ValueError("native terminal cash mismatch")
    if ledger["terminal_exact_quantities"] != last["exact_quantities_after"]:
        raise ValueError("native terminal quantity mismatch")
    if ledger["termination_reason"] != last["termination_reason"]:
        raise ValueError("native terminal reason mismatch")


def _verify_order_events(ledger: dict, intervals: list[dict]) -> None:
    # Use the native event schema, then bind each observed event to this
    # single-symbol book and its decision/processing interval.
    for row in intervals:
        events = row["order_events"]
        if not isinstance(events, list):
            raise ValueError("native order event roster mismatch")
        for raw in events:
            if not isinstance(raw, dict):
                raise ValueError("native order event must be an object")
            try:
                event = OrderEvent.from_mapping(raw)
            except (ValueError, OverflowError) as error:
                raise ValueError("native order event schema mismatch") from error
            if (
                event.dataset_id != ledger["dataset_id"]
                or event.execution_policy_digest != ledger["execution_policy_digest"]
                or event.symbol_index != 0
                or not row["start_index"] <= event.processing_index <= row["next_index"]
            ):
                raise ValueError("native order event identity mismatch")


def _verify_decision_trace(
    ledger: dict, intervals: list[dict], arm: str, scenario: str
) -> None:
    # All eleven arms use the same single-symbol native portfolio actor.
    # A completed actor prefix is mandatory even for an invalid terminal result.
    label = "native control decision" if arm in ("CASH", "BH25") else "native decision"
    decisions = ledger["portfolio_decisions"]
    reserve = 2 if scenario == "latency_1h" else 1
    expected = min(
        len(intervals), ledger["stop_index"] - ledger["start_index"] - reserve
    )
    if not isinstance(decisions, list) or expected <= 0 or len(decisions) != expected:
        raise ValueError(label + " missing or extra decisions")
    for i, decision in enumerate(decisions):
        if (
            not isinstance(decision, dict)
            or type(decision.get("index")) is not int
            or decision["index"] != intervals[i]["start_index"]
        ):
            raise ValueError(label + " identity mismatch")
        for field in ("proposal_weights", "target_weights"):
            weights = decision.get(field)
            if not isinstance(weights, list) or len(weights) != 1:
                raise ValueError(label + " weight roster mismatch")
            _finite(weights[0], label + " weight")
        for field in ("requested_hold_symbols", "effective_hold_symbols"):
            holds = decision.get(field)
            if (
                not isinstance(holds, list)
                or len(holds) > 1
                or any(type(index) is not int or index != 0 for index in holds)
            ):
                raise ValueError(label + " HOLD roster mismatch")
        reasons = decision.get("risk_reasons")
        if not isinstance(reasons, list) or any(
            type(reason) is not str for reason in reasons
        ):
            raise ValueError(label + " risk reason mismatch")
        if arm not in ("CASH", "BH25"):
            continue
        weight = decision["proposal_weights"][0]
        holds = decision["requested_hold_symbols"]
        if arm == "CASH":
            if weight != 0 or holds != []:
                raise ValueError("native control CASH must request zero")
        elif i == 0:
            if weight != 0.25 or holds != []:
                raise ValueError("native control BH25 initial weight mismatch")
        elif holds != [0]:
            raise ValueError("native control BH25 must request HOLD")
    if arm not in ("CASH", "BH25"):
        return
    for row in intervals:
        for field in ("exact_quantities_before", "exact_quantities_after"):
            quantities = row[field]
            if not isinstance(quantities, list) or len(quantities) != 1:
                raise ValueError("native control quantity roster mismatch")
            quantity = parse_quantity(quantities[0])
            if quantity < 0 or (arm == "CASH" and quantity != 0):
                raise ValueError("native control inventory contradicts arm")
        if arm == "CASH" and any(
            event["event_type"] in ("filled", "partial_fill")
            for event in row["order_events"]
        ):
            raise ValueError("native control CASH contains fills")


def read_cell(
    path: Path,
    *,
    admission: dict,
    dataset_id: str,
    arm: str,
    scenario: str,
    expected_start: int | None = None,
    expected_stop: int | None = None,
) -> dict:
    """Check identity, ledger bytes and decisive scalars, without a shadow replay.

    A technical failure is an invalid recorded outcome, not permission to retry.
    Interrupted, incomplete or contradictory receipts fail closed.
    """
    if path.is_symlink() or not path.is_dir():
        raise ValueError("invalid cell directory")
    attempt = {"admission": admission, "cell": path.name}
    if _read(path / "attempt.json") != attempt:
        raise ValueError("attempt admission or cell identity mismatch")
    if (path / "failure.json").exists():
        failure = _read(path / "failure.json")
        if (
            failure.get("cell") != path.name
            or type(failure.get("exception_type")) is not str
            or (path / "result.json").exists()
        ):
            raise ValueError("contradictory failure receipt")
        return {"valid": False, "technical_failure": failure["exception_type"]}

    record = _read(path / "result.json")
    if any(record.get(k) != v for k, v in attempt.items()):
        raise ValueError("result admission or cell identity mismatch")
    if record.get("dataset_id") != dataset_id:
        raise ValueError("result dataset identity mismatch")
    m = record["metrics"]
    if m.get("arm") != arm or m.get("scenario") != scenario:
        raise ValueError("result arm or scenario mismatch")
    ledger_path = path / "ledger.json.gz"
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise ValueError("missing native ledger")
    compressed = ledger_path.read_bytes()
    if _sha(compressed) != record.get("ledger_gzip_sha256"):
        raise ValueError("compressed ledger digest mismatch")
    raw = gzip.decompress(compressed)
    if _sha(raw) != m.get("ledger_sha256"):
        raise ValueError("native ledger digest mismatch")
    ledger = _canonical_object(raw)
    if (
        ledger.get("schema_version") != "shared_cash_portfolio_replay_ledger_v1"
        or ledger.get("dataset_id") != dataset_id
    ):
        raise ValueError("native ledger schema or dataset identity mismatch")
    start, stop = ledger["start_index"], ledger["stop_index"]
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop
        or (expected_start is not None and start != expected_start)
        or (expected_stop is not None and stop != expected_stop)
    ):
        raise ValueError("native ledger window mismatch")
    intervals = ledger["intervals"]
    if not intervals:
        raise ValueError("empty native ledger")
    if intervals[0]["start_index"] != start:
        raise ValueError("native ledger start mismatch")
    for index, row in enumerate(intervals, start=start):
        if (
            type(row["start_index"]) is not int
            or type(row["next_index"]) is not int
            or row["start_index"] != index
            or row["next_index"] != index + 1
            or row["next_index"] > stop
        ):
            raise ValueError("noncontiguous native interval evidence")
    initial = _finite(intervals[0]["portfolio_value_before"], "initial equity")
    if initial != 10000.0:
        raise ValueError("initial equity differs from fixed protocol")
    last = intervals[-1]
    end = _finite(ledger["final_portfolio_value"], "terminal equity")
    if not math.isclose(
        last["portfolio_value_after"], end, abs_tol=1e-7, rel_tol=1e-10
    ):
        raise ValueError("native terminal equity mismatch")
    # Intrabar native observations may exceed the close-path drawdown, so
    # observed boundary equity is a lower bound, not a replacement metric.
    peak = initial
    observed_dd = 0.0
    prior_native_dd = 0.0
    for row in intervals:
        before_dd = _finite(row["max_drawdown_before"], "drawdown before")
        after_dd = _finite(row["max_drawdown_after"], "drawdown after")
        # This protocol uses native floor-zero loss semantics, not retain-debt.
        if not (0 <= before_dd <= 1 and 0 <= after_dd <= 1):
            raise ValueError("native drawdown outside floor-zero bounds")
        if not math.isclose(before_dd, prior_native_dd, abs_tol=1e-8, rel_tol=0):
            raise ValueError("native drawdown discontinuity")
        for field in ("portfolio_value_before", "portfolio_value_after"):
            equity = max(_finite(row[field], field), 0.0)
            peak = max(peak, equity)
            observed_dd = max(observed_dd, 1.0 - equity / peak)
        if after_dd < max(observed_dd, before_dd) - 1e-8:
            raise ValueError("native drawdown below observed boundary lower bound")
        prior_native_dd = after_dd
    if not math.isclose(
        _finite(ledger["final_max_drawdown"], "final drawdown"),
        prior_native_dd,
        abs_tol=1e-8,
        rel_tol=0,
    ):
        raise ValueError("native terminal drawdown mismatch")
    _verify_native_flows(ledger, intervals)
    _verify_native_terminal_trace(ledger, intervals)
    _verify_decision_trace(ledger, intervals, arm, scenario)
    _verify_order_events(ledger, intervals)
    expected = {
        "net_return_pct": 100 * (end / initial - 1),
        "max_drawdown_pct": 100 * ledger["final_max_drawdown"],
        "turnover": ledger["final_turnover_total"],
        "total_cost_usdt": ledger["final_total_cost"],
        "funding_pnl_usdt": ledger["final_funding_pnl"],
        "borrow_cost_usdt": ledger["final_borrow_cost"],
    }
    for name, value in expected.items():
        if not math.isclose(
            _finite(m[name], name), _finite(value, name), rel_tol=1e-10, abs_tol=1e-7
        ):
            raise ValueError("native scalar mismatch: " + name)
    # Reconstruct non-decisive diagnostics from the unchanged native receipt.
    returns = np.asarray(
        [_finite(row["interval_net_return"], "interval return") for row in intervals]
    )
    sd = float(np.std(returns))
    sharpe = float(np.mean(returns) / sd * np.sqrt(8760)) if sd > 1e-12 else None
    reported_sharpe = m.get("sharpe_hourly")
    if (
        (sharpe is None and reported_sharpe is not None)
        or (
            sharpe is not None
            and not math.isclose(
                _finite(reported_sharpe, "sharpe_hourly"),
                sharpe,
                rel_tol=1e-10,
                abs_tol=1e-7,
            )
        )
        or "sharpe_hourly" not in m
    ):
        raise ValueError("native scalar mismatch: sharpe_hourly")
    fills = sum(
        event["event_type"] in ("filled", "partial_fill")
        for row in intervals
        for event in row["order_events"]
    )
    if type(m.get("native_fills")) is not int or m["native_fills"] != fills:
        raise ValueError("native scalar mismatch: native_fills")
    attribution = asdict(
        attribute_single_symbol_ledger([SimpleNamespace(**row) for row in intervals])
    )
    reported = m.get("directional_equity_buckets")
    if not isinstance(reported, dict) or reported.keys() != attribution.keys():
        raise ValueError("native scalar mismatch: directional_equity_buckets")
    for name, value in attribution.items():
        actual = reported[name]
        if (type(value) is int and (type(actual) is not int or actual != value)) or (
            type(value) is not int
            and not math.isclose(
                _finite(actual, name), value, rel_tol=1e-10, abs_tol=1e-7
            )
        ):
            raise ValueError(
                "native scalar mismatch: directional_equity_buckets." + name
            )
    quantities = [
        parse_quantity(row["exact_quantities_before"][0]) for row in intervals
    ]
    terminal = bool(
        ledger["termination_reason"] is None
        and last["next_index"] == stop
        and len(ledger["terminal_exact_quantities"]) == 1
        and all(parse_quantity(q) == 0 for q in ledger["terminal_exact_quantities"])
        and not any(abs(q) > 0 for _, q in ledger["active_order_remainders"])
    )
    valid = terminal and (arm != "BH25" or any(q > 0 for q in quantities))
    for name, value in (
        ("terminal_ok", terminal),
        ("valid", valid),
        ("pre_interval_long_hours", sum(q > 0 for q in quantities)),
        ("pre_interval_short_hours", sum(q < 0 for q in quantities)),
        ("termination_reason", ledger["termination_reason"]),
    ):
        if type(m.get(name)) is not type(value) or m.get(name) != value:
            raise ValueError("native scalar mismatch: " + name)
    return m


def decide_comparison(protocol: dict, cells: dict[CellKey, dict]) -> dict:
    """Apply the predeclared S2-only decision gate, never choose an alternate."""
    expected = set(roster(protocol))
    unexpected = set(cells) - expected
    if unexpected:
        raise ValueError("unregistered comparison cell")
    missing = expected - set(cells)
    for key, row in cells.items():
        if type(row.get("valid")) is not bool:
            raise ValueError(f"invalid cell validity: {key}")
        if row["valid"]:
            _finite(row["net_return_pct"], "net_return_pct")
            if _finite(row["max_drawdown_pct"], "max_drawdown_pct") < 0:
                raise ValueError("negative drawdown")
    primary_invalid = [
        cell_name(key)
        for key in expected
        if key[3] != "latency_1h" and not cells.get(key, {}).get("valid", False)
    ]
    latency_all_valid = all(
        cells.get(key, {}).get("valid", False)
        for key in expected
        if key[3] == "latency_1h"
    )
    result: dict[str, Any] = {
        "disposition": "INCONCLUSIVE",
        "recorded_cells": len(cells),
        "required_cells": len(expected),
        "missing_cells": sorted(map(cell_name, missing)),
        "primary_invalid_cells": sorted(primary_invalid),
        "latency_all_valid": latency_all_valid,
        # Stress metrics are diagnostic; no latency-robust acceptance
        # rule was preregistered, even if all stress cells are valid.
        "latency_robustness_established": False,
        "scenario_comparisons": {},
    }
    if missing or primary_invalid:
        return result
    cohorts = [(s, y) for s in protocol["symbols"] for y in protocol["years"]]
    passes = []
    for scenario in ("normal", "double_cost", "latency_1h"):
        stats: dict[str, Any] = {}
        for arm in protocol["arms"]:
            rows = [cells[s, y, arm, scenario] for s, y in cohorts]
            if not all(row["valid"] for row in rows):
                stats[arm] = {"valid": False}
                continue
            stats[arm] = {
                "valid": True,
                "mean_return_pct": math.fsum(row["net_return_pct"] for row in rows)
                / len(rows),
                "worst_mdd_pct": max(row["max_drawdown_pct"] for row in rows),
            }
        result["scenario_comparisons"][scenario] = {"arms": stats}
        if scenario == "latency_1h":
            continue
        s2 = stats["S2"]
        hold = stats["BH25"]
        beats_hold = sum(
            cells[s, y, "S2", scenario]["net_return_pct"]
            > cells[s, y, "BH25", scenario]["net_return_pct"]
            for s, y in cohorts
        )
        checks = {
            "positive": s2["mean_return_pct"] > 0,
            "beats_every_other_arm": all(
                s2["mean_return_pct"] > row["mean_return_pct"]
                for arm, row in stats.items()
                if arm != "S2"
            ),
            "at_least_four_hold_wins": beats_hold >= 4,
            "worst_mdd_not_above_hold": s2["worst_mdd_pct"] <= hold["worst_mdd_pct"],
        }
        result["scenario_comparisons"][scenario].update(
            {"s2_hold_wins": beats_hold, "s2_checks": checks}
        )
        passes.append(all(checks.values()))
    result["disposition"] = "DEVELOPMENT_PROMISING" if all(passes) else "NO_WINNER"
    return result
