"""Offline replay and the fixed ninety-day prospective paper screen."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from trade_rl._validation import require_aware_datetime
from trade_rl.evaluation.paper.account import PaperSettings, timestamp
from trade_rl.evaluation.paper.engine import PaperEngine
from trade_rl.evaluation.paper.operations import screen_plan
from trade_rl.evaluation.runs import build_candidate_run_provenance


@dataclass(frozen=True)
class Measurements:
    net_profit: float
    recorded_fees: float
    block_returns: tuple[float, ...]
    block_funding_counts: tuple[int, ...]
    maximum_drawdown: float
    first_delay_seconds: float
    final_grace_seconds: float
    maximum_gap_seconds: float
    terminal_flat: bool
    pending: bool
    stop_reason: str | None
    quality_failures: tuple[str, ...]
    unpaid_funding: tuple[dict[str, str], ...]
    filled_symbols: tuple[int, ...]


def screen_reasons(measured: Measurements) -> list[str]:
    m = measured
    checks = {
        "nonfinite_metrics": not all(
            math.isfinite(value)
            for value in (
                m.net_profit,
                m.recorded_fees,
                m.maximum_drawdown,
                m.first_delay_seconds,
                m.final_grace_seconds,
                m.maximum_gap_seconds,
                *m.block_returns,
            )
        ),
        "nonpositive_net_profit": m.net_profit <= 0,
        "insufficient_fee_headroom": m.net_profit <= m.recorded_fees,
        "nonpositive_block_return": len(m.block_returns) != 3
        or any(value <= 0 for value in m.block_returns),
        "missing_block_funding": len(m.block_funding_counts) != 3
        or any(count <= 0 for count in m.block_funding_counts),
        "drawdown_limit": not 0 <= m.maximum_drawdown < 0.1,
        "initial_coverage_gap": not 0 <= m.first_delay_seconds <= 180,
        "observation_coverage_gap": not 0 <= m.maximum_gap_seconds <= 180,
        "terminal_coverage_gap": not 120 <= m.final_grace_seconds < 180,
        "not_flat": not m.terminal_flat,
        "pending_intent": m.pending,
        "nonterminal_stop": m.stop_reason != "terminal_close",
        "quality_failure": bool(m.quality_failures),
        "unpaid_funding": bool(m.unpaid_funding),
        "missing_instrument_fills": m.filled_symbols != (0, 1, 2, 3),
    }
    return [name for name, failed in checks.items() if failed]


def measure_observations(
    records: list[dict[str, Any]],
    status: dict[str, Any],
    *,
    start_at: datetime,
    close_at: datetime,
    unpaid_funding: list[dict[str, str]],
) -> Measurements:
    points: list[tuple[datetime, float]] = []
    counts = [0, 0, 0]
    filled: set[int] = set()
    for record in records:
        event = record["event"]
        if event["kind"] == "gap":
            continue
        payload = event["payload"]
        result = payload["result"]
        at = timestamp(payload["request"]["at"])
        points.append((at, result["account"]["equity"]))
        filled.update(
            row["symbol_index"] for row in result["fills"] if row["filled_lots"]
        )
        for payment in result["funding"]:
            known = timestamp(payment["known_at"])
            if payment["amount"] and known >= start_at:
                index = min(2, int((known - start_at).total_seconds() // (30 * 86400)))
                counts[index] += 1
    account = status["account"]
    equities = [10000.0]
    for days in (30, 60):
        boundary = start_at + timedelta(days=days)
        eligible = [(at, equity) for at, equity in points if at <= boundary]
        if eligible and (boundary - eligible[-1][0]).total_seconds() <= 180:
            equities.append(eligible[-1][1])
        else:
            equities.append(0.0)
    equities.append(account["equity"])
    returns = tuple(
        right / left - 1 if left > 0 else -1.0
        for left, right in zip(equities, equities[1:])
    )
    times = [at for at, _ in points]
    gaps = [
        (right - left).total_seconds() for left, right in zip([start_at, *times], times)
    ]
    return Measurements(
        net_profit=account["equity"] - 10000,
        recorded_fees=account["total_cost"],
        block_returns=returns,
        block_funding_counts=tuple(counts),
        maximum_drawdown=account["maximum_drawdown"],
        first_delay_seconds=(times[0] - start_at).total_seconds() if times else -1,
        final_grace_seconds=(times[-1] - close_at).total_seconds() if times else -1,
        maximum_gap_seconds=max(gaps, default=-1),
        terminal_flat=status["terminal_flat"],
        pending=status["pending"],
        stop_reason=status["stop_reason"],
        quality_failures=tuple(status["quality_failures"]),
        unpaid_funding=tuple(unpaid_funding),
        filled_symbols=tuple(sorted(filled)),
    )


def collection_evidence_reasons(
    root: Path,
    digest: str,
    records: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
) -> list[str]:
    """Audit control acknowledgements and all source directories, including unused ones."""
    reasons = []
    if (root / "collection-failure.json").exists() or (
        root / "collection-failure.json"
    ).is_symlink():
        reasons.append("collection_failure")
    path = root / "collection-control.sqlite"
    if not path.is_file() or path.is_symlink():
        return [*reasons, "missing_collection_control"]
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
        if connection.execute("SELECT digest FROM identity").fetchall() != [(digest,)]:
            raise ValueError("collection control protocol mismatch")
        cycles = connection.execute(
            "SELECT id,started_at,complete FROM cycles ORDER BY id"
        ).fetchall()
    if not cycles or [row[0] for row in cycles] != list(range(1, len(cycles) + 1)):
        reasons.append("invalid_cycle_sequence")
    if any(row[2] != 1 for row in cycles):
        reasons.append("incomplete_collection_cycle")
    starts = [timestamp(row[1]) for row in cycles]
    if any(not start <= at < end for at in starts) or any(
        left >= right for left, right in zip(starts, starts[1:])
    ):
        reasons.append("invalid_cycle_times")
    grouped: list[list[str]] = [[] for _ in cycles]
    references: set[str] = set()
    for record in records:
        event = record["event"]
        request = event["payload"]["request"]
        at = timestamp(request["at"])
        index = bisect_right(starts, at) - 1
        if index < 0 or not start < at < end:
            reasons.append("untracked_account_command")
        else:
            grouped[index].append(event["kind"])
        references.update(
            request[key]["path"] for key in ("market", "rules") if key in request
        )
    if any(kinds not in (["decision"], ["decision", "execution"]) for kinds in grouped):
        reasons.append("cycle_command_mismatch")
    source_root = root / "sources"
    sources: set[str] = set()
    if not source_root.is_dir() or source_root.is_symlink():
        reasons.append("missing_collection_sources")
    else:
        for day in source_root.iterdir():
            if not day.is_dir() or day.is_symlink():
                reasons.append("invalid_source_roster")
                continue
            for capture in day.iterdir():
                if not capture.is_dir() or capture.is_symlink():
                    reasons.append("invalid_source_roster")
                sources.add(capture.relative_to(root).as_posix())
                if (capture / "failure.json").exists():
                    reasons.append("source_capture_failure")
    if sources != references:
        reasons.append("unconsumed_or_missing_sources")
    return list(dict.fromkeys(reasons))


def evaluate_paper_study(
    root: str | Path,
    *,
    expected_protocol_sha256: str,
    expected_tip: str,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    destination = Path(root).absolute()
    path = destination / "protocol.json"
    if destination.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("paper protocol must be a regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_protocol_sha256:
        raise ValueError("paper protocol digest mismatch")
    protocol = json.loads(raw)["protocol"]
    study = protocol["study"]
    start = timestamp(protocol["settings"]["start_at"])
    close = start + timedelta(days=90)
    if protocol["settings"] != PaperSettings(start_at=start, close_at=close).to_dict():
        raise ValueError("paper study differs from fixed prospective settings")
    if (
        study.get("schema") != "paper_public_collection_v1"
        or study.get("research_plan") != screen_plan()
        or study.get("interval_seconds") != 60
        or study.get("terminal_grace_seconds") != 180
        or timestamp(study["created_at"]) > start - timedelta(minutes=5)
    ):
        raise ValueError("paper study differs from the frozen economic protocol")
    now = require_aware_datetime(clock(), field="paper assessment")
    end = close + timedelta(seconds=180)
    if now < end:
        raise ValueError("cannot evaluate before the fixed terminal deadline")
    if build_candidate_run_provenance() != study["provenance"]:
        raise ValueError("paper study source/runtime provenance changed")
    lock = destination / "collector.sqlite"
    if lock.is_symlink() or not lock.is_file():
        raise ValueError("paper collection has no regular collector lock")
    connection = sqlite3.connect(
        f"{lock.as_uri()}?mode=rw", uri=True, isolation_level=None, timeout=0
    )
    try:
        try:
            connection.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as error:
            raise RuntimeError(
                "stop the active collector before economic evaluation"
            ) from error
        engine = PaperEngine(
            destination, expected_protocol_sha256=expected_protocol_sha256
        )
        state = engine.status()
        if state["tip"] != expected_tip:
            raise ValueError("paper final event tip differs from external anchor")
        records = engine.journal.events()
        measured = measure_observations(
            records,
            state,
            start_at=start,
            close_at=close,
            unpaid_funding=engine.unsettled_funding(),
        )
        reasons = collection_evidence_reasons(
            destination, expected_protocol_sha256, records, start=start, end=end
        )
        reasons.extend(screen_reasons(measured))
        if (
            path.is_symlink()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_protocol_sha256
        ):
            raise ValueError("paper protocol digest changed during assessment")
        if build_candidate_run_provenance() != study["provenance"]:
            raise ValueError(
                "paper study source/runtime provenance changed during assessment"
            )
        return dict(
            schema="carry_paper_assessment_v1",
            protocol_sha256=expected_protocol_sha256,
            final_tip=expected_tip,
            assessed_at=now.isoformat(),
            decision="PAPER_SCREEN_PASSED" if not reasons else "PAPER_SCREEN_REJECTED",
            reasons=reasons,
            measurements=asdict(measured),
            status=state,
            production_eligible=False,
            fee_headroom_is_dynamic_stress_replay=False,
        )
    finally:
        connection.rollback()
        connection.close()
