"""Append diagnostics and retain ordinary attempt state plus exact file hashes."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from math import fsum
from pathlib import Path
from typing import Any

from tier_d_protocol import CAPITAL, file_sha256


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_log(root: Path, event: str, **values: Any) -> None:
    row = {"utc": datetime.now(UTC).isoformat(), "event": event, **values}
    with (root / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)


def initialize_attempt(
    root: Path, protocol: dict[str, Any], *, resume: bool = False
) -> dict[str, Any]:
    attempt_path = root / "attempt.json"
    if attempt_path.exists():
        if not resume:
            raise FileExistsError(
                "retain prior attempt; use a new attempt directory for technical retries"
            )
        state = json.loads(attempt_path.read_text(encoding="utf-8"))
        append_log(
            root,
            "attempt_resumed",
            scope=protocol["scope"],
            completed_arms=sum(r["state"] == "COMPLETED" for r in state["arms"]),
        )
        return state
    root.mkdir(parents=True, exist_ok=True)
    state = {
        "scope": protocol["scope"],
        "status": "PREPARING",
        "arms": [
            {
                "account": account,
                "scenario": scenario,
                "role": role,
                "state": "UNATTEMPTED",
                "receipt": None,
                "failure": None,
            }
            for account in protocol["accounts"]
            for scenario in protocol["scenarios"]
            for role in protocol["roles"]
        ],
    }
    write_json(root / "protocol.json", protocol)
    write_json(attempt_path, state)
    append_log(
        root, "attempt_started", scope=protocol["scope"], arms=len(state["arms"])
    )
    return state


def save_gzip(root: Path, relative: str, payload: Any) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    # json.dump writes incrementally; avoid another full raw JSON bytes buffer.
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(
            payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    return {"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size}


def load_gzip(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def arm_state(
    state: dict[str, Any], account: str, scenario: str, role: str
) -> dict[str, Any]:
    return next(
        row
        for row in state["arms"]
        if (row["account"], row["scenario"], row["role"]) == (account, scenario, role)
    )


def native_closure(raw: dict[str, Any]) -> dict[str, Any]:
    native = raw["native_receipt"]
    rows = native["rows"]
    fills = [
        row["native"]["execution"]["next_index"]
        for row in rows
        if row["native"] is not None and row["native"]["execution"]["fill_count"] > 0
    ]
    return {
        "reset_count": native["reset_count"],
        "actor_rows": len(rows),
        "closing_rows": len(raw["closing_rows"]),
        "settlement_complete": raw["settlement_complete"],
        "final_quantities": raw["final_book"]["quantities"],
        "active_orders": len(raw["final_active_orders"]),
        "ledger_reward_sum": raw["fixed_capital_reward_sum"],
        "first_fill_processing_index": min(fills) if fills else None,
    }


def combined_rows(
    groups: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    """Combine exactly two independent symbol books, never all 48 capital books."""
    rows = []
    for config in protocol["configurations"]:
        for scenario in protocol["scenarios"]:
            peers = [
                group
                for group in groups
                if group["configuration"] == config["id"]
                and group["scenario"] == scenario
            ]
            complete = len(peers) == 2 and all(
                group["comparison_complete"] for group in peers
            )
            denominator = 2 * CAPITAL
            row = {
                "configuration": config["id"],
                "scenario": scenario,
                "symbol_accounts_observed": len(peers),
                "denominator": denominator,
                "comparison_complete": complete,
                "nonrl_minus_cash": None,
                "nonrl_profit_rate": None,
                "cash_profit_rate": None,
            }
            if complete:
                nonrl = fsum(
                    group["roles"]["nonrl"]["final_equity"] - CAPITAL for group in peers
                )
                cash = fsum(
                    group["roles"]["cash"]["final_equity"] - CAPITAL for group in peers
                )
                row.update(
                    nonrl_minus_cash=(nonrl - cash) / denominator,
                    nonrl_profit_rate=nonrl / denominator,
                    cash_profit_rate=cash / denominator,
                )
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["nonrl_minus_cash"] is None,
            -row["nonrl_minus_cash"] if row["nonrl_minus_cash"] is not None else 0,
            row["configuration"],
            row["scenario"],
        ),
    )


def output_manifest(root: Path) -> dict[str, Any]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "output-manifest.json"
    }
