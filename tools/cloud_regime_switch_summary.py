"""Read-only integrity audit and fixed decision rule for the switch screen.

Consumes only authorized, previously recorded native replay receipts.
No new account, order simulator, parameter selection, or market data loader.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from trade_rl.artifacts import canonical_json_bytes
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
    if (expected_start is not None and start != expected_start) or (
        expected_stop is not None and stop != expected_stop
    ):
        raise ValueError("native ledger window mismatch")
    intervals = ledger["intervals"]
    if not intervals:
        raise ValueError("empty native ledger")
    if intervals[0]["start_index"] != start:
        raise ValueError("native ledger start mismatch")
    for index, row in enumerate(intervals, start=start):
        if row["start_index"] != index or row["next_index"] != index + 1:
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
