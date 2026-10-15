"""Summarize observed software timings without extrapolation or profit ranking."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tools.tier_d_nonrl.tier_d_records import load_gzip, read_json
from trade_rl.artifacts import canonical_json_bytes, content_digest

from .context import _check_manifest
from .spec import CASES
from .supervision import LIMITS


def _matches_limits(value: object) -> bool:
    return (
        type(value) is dict
        and value.keys() == LIMITS.keys()
        and all(type(value[key]) is int and value[key] == LIMITS[key] for key in LIMITS)
    )


def _supervision_qualifies(root: Path, measurement: dict[str, Any]) -> bool:
    protocol = measurement.get("protocol")
    protocol_measurement = (
        protocol.get("measurement") if type(protocol) is dict else None
    )
    protocol_limits = (
        protocol_measurement.get("limits")
        if type(protocol_measurement) is dict
        else None
    )
    if not _matches_limits(protocol_limits):
        return False

    supervision_path = root.parent / "supervision.json"
    if not supervision_path.is_file():
        return False
    try:
        record = read_json(supervision_path)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    if type(record) is not dict:
        return False

    command = record.get("command")
    if type(command) is not list or any(type(item) is not str for item in command):
        return False
    root_flags = [index for index, item in enumerate(command) if item == "--root"]
    if len(root_flags) != 1 or root_flags[0] + 1 >= len(command):
        return False
    try:
        command_root = Path(command[root_flags[0] + 1]).resolve()
        measurement_root = root.resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False

    return (
        record.get("status") == "COMPLETED"
        and type(record.get("returncode")) is int
        and record["returncode"] == 0
        and record.get("worker_reaped") is True
        and type(record.get("owned_members_remaining")) is int
        and record["owned_members_remaining"] == 0
        and "failure" in record
        and record["failure"] is None
        and "cleanup_error" in record
        and record["cleanup_error"] is None
        and record.get("final_violation") == []
        and "final_sample_error" in record
        and record["final_sample_error"] is None
        and _matches_limits(record.get("limits"))
        and record["limits"] == protocol_limits
        and command_root == measurement_root
    )


def summarize(roots: Sequence[Path]) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}
    binding = None
    for root in roots:
        _check_manifest(root)
        row = read_json(root / "measurement.json")
        identity = (row["case"]["name"], row["case"]["target"], row["case"]["profile"])
        name = identity[0]
        if (
            identity not in CASES
            or name in cases
            or row["financial_authority"] is not False
            or row["protocol"]["scope"] != "ARTIFICIAL_SOFTWARE_PROFILE_ONLY"
        ):
            raise ValueError("foreign or repeated artificial measurement")
        current = content_digest(
            {
                k: row[k]
                for k in ("protocol", "provenance", "plan_digest", "prepared_root")
            }
        )
        if binding is not None and current != binding:
            raise ValueError(
                "measurements must share the exact protocol, source and full plan"
            )
        binding = current
        if row.get("receipt_canonical_sha256"):
            digest = hashlib.sha256(
                canonical_json_bytes(load_gzip(root / "receipt.partial.json.gz"))
            ).hexdigest()
            if digest != row["receipt_canonical_sha256"]:
                raise ValueError("partial receipt differs from canonical byte pin")
        row["qualified_measurement"] = (
            row["status"] == "PREFIX_OBSERVED"
            and row.get("expected_stop") is True
            and type(row.get("count")) is int
            and row["count"] == identity[1]
            and row.get("readback_verified") is True
            and bool(row.get("receipt_canonical_sha256"))
            and type(row.get("elapsed_seconds")) in (int, float)
            and row["elapsed_seconds"] > 0
            and _supervision_qualifies(root, row)
        )
        cases[name] = row
    timing_names = ("n100", "n400", "n1600")
    curve = all(
        cases.get(name, {}).get("qualified_measurement") for name in timing_names
    )
    oracle = "NOT_ESTABLISHED"
    if all(
        cases.get(name, {}).get("qualified_measurement")
        for name in ("n1600", "n1600-profile")
    ):
        oracle = (
            "MATCH"
            if cases["n1600"]["receipt_canonical_sha256"]
            == cases["n1600-profile"]["receipt_canonical_sha256"]
            else "MISMATCH"
        )
    complete = len(cases) == len(CASES) and all(
        row["qualified_measurement"] for row in cases.values()
    )
    return {
        "status": "MEASUREMENTS_COMPLETE"
        if complete and oracle == "MATCH"
        else "INCOMPLETE",
        "financial_authority": False,
        "receipt_byte_oracle": oracle,
        "historical_release_equivalence": "NOT_ESTABLISHED",
        "time_ratios": [
            cases[b]["elapsed_seconds"] / cases[a]["elapsed_seconds"]
            for a, b in zip(timing_names, timing_names[1:])
        ]
        if curve
        else [],
        "measurements": [
            {
                "name": name,
                "status": row["status"],
                "qualified_measurement": row["qualified_measurement"],
                "count": row.get("count"),
                "elapsed_seconds": row.get("elapsed_seconds"),
            }
            for name, row in sorted(cases.items())
        ],
        "limitations": "one observation per case; profiled timing separate; no complexity-order, reserve/vintage, real-data or full48 bound",
    }
