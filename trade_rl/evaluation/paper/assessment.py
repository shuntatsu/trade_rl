"""Offline replay and the fixed ninety-day prospective paper screen."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from trade_rl._validation import require_aware_datetime
from trade_rl.evaluation.paper.account import PaperSettings, timestamp
from trade_rl.evaluation.paper.collection_audit import (
    Registry,
    allowed_empty_cycles,
    audit_registered_collection,
    cycle_slot,
    read_registry,
)
from trade_rl.evaluation.paper.engine import PaperEngine
from trade_rl.evaluation.paper.operations import (
    REQUIRED_FILL_SYMBOL_INDICES,
    STORAGE_PROBE_PLAN_SCHEMA,
    screen_plan,
    screen_plan_v3,
    validate_attempt_lineage,
    validate_attempt_lineage_v4,
)
from trade_rl.evaluation.paper.predecessor import (
    ATTEMPT_ONE_PROTOCOL_SHA256,
    SCREEN_V3_SCHEMA,
    same_canonical_record,
)
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
    post_close_funding: bool


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
        "post_close_funding": m.post_close_funding,
        "drawdown_limit": not 0 <= m.maximum_drawdown < 0.1,
        "initial_coverage_gap": not 0 <= m.first_delay_seconds <= 180,
        "observation_coverage_gap": not 0 <= m.maximum_gap_seconds <= 180,
        "terminal_coverage_gap": not 120 <= m.final_grace_seconds < 180,
        "not_flat": not m.terminal_flat,
        "pending_intent": m.pending,
        "nonterminal_stop": m.stop_reason != "terminal_close",
        "quality_failure": bool(m.quality_failures),
        "unpaid_funding": bool(m.unpaid_funding),
        "missing_instrument_fills": m.filled_symbols != REQUIRED_FILL_SYMBOL_INDICES,
    }
    return [name for name, failed in checks.items() if failed]


class _ObservationAccumulator:
    def __init__(self, *, start_at: datetime, close_at: datetime) -> None:
        self.start_at = start_at
        self.close_at = close_at
        self.points: list[tuple[datetime, float]] = []
        self.counts = [0, 0, 0]
        self.filled: set[int] = set()
        self.post_close_funding = False

    def add(self, record: dict[str, Any]) -> None:
        event = record["event"]
        if event["kind"] == "gap":
            return
        payload = event["payload"]
        result = payload["result"]
        at = timestamp(payload["request"]["at"])
        self.points.append((at, result["account"]["equity"]))
        self.filled.update(
            row["symbol_index"] for row in result["fills"] if row["filled_lots"]
        )
        for payment in result["funding"]:
            settled = timestamp(payment["funding_at"])
            if not payment["amount"]:
                continue
            if settled >= self.close_at:
                self.post_close_funding = True
            if self.start_at <= settled < self.close_at:
                index = int((settled - self.start_at).total_seconds() // (30 * 86400))
                self.counts[index] += 1

    def finish(
        self, status: dict[str, Any], unpaid_funding: list[dict[str, str]]
    ) -> Measurements:
        account = status["account"]
        equities = [10000.0]
        for days in (30, 60):
            boundary = self.start_at + timedelta(days=days)
            eligible = [(at, equity) for at, equity in self.points if at <= boundary]
            if eligible and (boundary - eligible[-1][0]).total_seconds() <= 180:
                equities.append(eligible[-1][1])
            else:
                equities.append(0.0)
        equities.append(account["equity"])
        returns = tuple(
            right / left - 1 if left > 0 else -1.0
            for left, right in zip(equities, equities[1:])
        )
        times = [at for at, _ in self.points]
        gaps = [
            (right - left).total_seconds()
            for left, right in zip([self.start_at, *times], times)
        ]
        return Measurements(
            net_profit=account["equity"] - 10000,
            recorded_fees=account["total_cost"],
            block_returns=returns,
            block_funding_counts=tuple(self.counts),
            maximum_drawdown=account["maximum_drawdown"],
            first_delay_seconds=(times[0] - self.start_at).total_seconds()
            if times
            else -1,
            final_grace_seconds=(times[-1] - self.close_at).total_seconds()
            if times
            else -1,
            maximum_gap_seconds=max(gaps, default=-1),
            terminal_flat=status["terminal_flat"],
            pending=status["pending"],
            stop_reason=status["stop_reason"],
            quality_failures=tuple(status["quality_failures"]),
            unpaid_funding=tuple(unpaid_funding),
            filled_symbols=tuple(sorted(self.filled)),
            post_close_funding=self.post_close_funding,
        )


def measure_observations(
    records: list[dict[str, Any]],
    status: dict[str, Any],
    *,
    start_at: datetime,
    close_at: datetime,
    unpaid_funding: list[dict[str, str]],
) -> Measurements:
    accumulator = _ObservationAccumulator(start_at=start_at, close_at=close_at)
    for record in records:
        accumulator.add(record)
    return accumulator.finish(status, unpaid_funding)


def _collection_context(
    root: Path,
    digest: str,
    *,
    start: datetime,
    end: datetime,
    registered: bool = False,
) -> tuple[
    list[str],
    list[tuple[int, str, int]] | None,
    list[datetime] | None,
    Registry | None,
]:
    reasons = []
    if (root / "collection-failure.json").exists() or (
        root / "collection-failure.json"
    ).is_symlink():
        reasons.append("collection_failure")
    path = root / "collection-control.sqlite"
    if not path.is_file() or path.is_symlink():
        return [*reasons, "missing_collection_control"], None, None, None
    registry = None
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
        if connection.execute("SELECT digest FROM identity").fetchall() != [(digest,)]:
            raise ValueError("collection control protocol mismatch")
        cycles = connection.execute(
            "SELECT id,started_at,complete FROM cycles ORDER BY id"
        ).fetchall()
        if registered:
            registry = read_registry(connection)
    if not cycles or [row[0] for row in cycles] != list(range(1, len(cycles) + 1)):
        reasons.append("invalid_cycle_sequence")
    if any(row[2] != 1 for row in cycles):
        reasons.append("incomplete_collection_cycle")
    starts = [timestamp(row[1]) for row in cycles]
    if any(not start <= at < end for at in starts) or any(
        left >= right for left, right in zip(starts, starts[1:])
    ):
        reasons.append("invalid_cycle_times")
    return reasons, cycles, starts, registry


def _finish_collection_evidence(
    root: Path,
    reasons: list[str],
    grouped: list[list[str]],
    references: set[str],
    *,
    allowed_empty: frozenset[int] = frozenset(),
    registered_paths: frozenset[str] = frozenset(),
    disk: dict[str, bool] | None = None,
) -> list[str]:
    """The screen v2 audit; the keyword arguments carry what screen v3 accepts.

    ``allowed_empty`` are positions in ``grouped`` that may hold no command and
    ``registered_paths`` the capture directories the registry accounts for. With both
    empty this is exactly the v2 audit. ``disk`` collects every capture directory and
    whether it holds ``failure.json``.
    """
    if any(
        kinds not in (["decision"], ["decision", "execution"])
        and not (not kinds and index in allowed_empty)
        for index, kinds in enumerate(grouped)
    ):
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
                relative = capture.relative_to(root).as_posix()
                sources.add(relative)
                failed = (capture / "failure.json").exists()
                if disk is not None:
                    disk[relative] = failed
                if failed and relative not in registered_paths:
                    reasons.append("source_capture_failure")
    if sources - registered_paths != references:
        reasons.append("unconsumed_or_missing_sources")
    return list(dict.fromkeys(reasons))


def collection_evidence_reasons(
    root: Path,
    digest: str,
    records: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
) -> list[str]:
    """Audit control acknowledgements and all source directories, including unused ones."""
    return _collection_evidence_reasons_from_compact_events(
        root,
        digest,
        _iter_compact_audit_events(records),
        start=start,
        end=end,
    )


def _iter_compact_audit_events(
    records: Iterable[dict[str, Any]],
) -> Iterable[tuple[str, dict[str, Any]]]:
    for record in records:
        event = record["event"]
        yield event["kind"], event["payload"]["request"]


def _collection_evidence_reasons_from_compact_events(
    root: Path,
    digest: str,
    events: Iterable[tuple[str, dict[str, Any]]],
    *,
    start: datetime,
    end: datetime,
) -> list[str]:
    return _audit_compact_events(
        root, digest, events, start=start, end=end, registered=False
    )[0]


def registered_collection_evidence(
    root: Path,
    digest: str,
    records: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
) -> tuple[list[str], dict[str, Any] | None]:
    """The screen v3 collection audit: registered failures inside the budgets pass.

    Returns the rejection reasons and the skip report (``None`` when the control record
    or the registry is missing). Everything screen v2 rejected is still rejected unless
    the collector registered it as a failed capture or a skipped slot of a registered
    class inside the pre-registered budgets (``collection_audit``).
    """
    return _registered_collection_evidence_from_compact_events(
        root, digest, _iter_compact_audit_events(records), start=start, end=end
    )


def _registered_collection_evidence_from_compact_events(
    root: Path,
    digest: str,
    events: Iterable[tuple[str, dict[str, Any]]],
    *,
    start: datetime,
    end: datetime,
) -> tuple[list[str], dict[str, Any] | None]:
    return _audit_compact_events(
        root, digest, events, start=start, end=end, registered=True
    )


def _audit_compact_events(
    root: Path,
    digest: str,
    events: Iterable[tuple[str, dict[str, Any]]],
    *,
    start: datetime,
    end: datetime,
    registered: bool,
) -> tuple[list[str], dict[str, Any] | None]:
    reasons, cycles, starts, registry = _collection_context(
        root, digest, start=start, end=end, registered=registered
    )
    if cycles is None or starts is None:
        return reasons, None
    if registered and registry is None:
        return [*reasons, "missing_collection_registry"], None
    grouped: list[list[str]] = [[] for _ in cycles]
    references: set[str] = set()
    for kind, request in events:
        at = timestamp(request["at"])
        index = bisect_right(starts, at) - 1
        if index < 0 or not start < at < end:
            reasons.append("untracked_account_command")
        else:
            grouped[index].append(kind)
        references.update(
            request[key]["path"] for key in ("market", "rules") if key in request
        )
    if registry is None:
        return _finish_collection_evidence(root, reasons, grouped, references), None
    identifiers = [row[0] for row in cycles]
    empty = allowed_empty_cycles(registry, identifiers)
    disk: dict[str, bool] = {}
    reasons = _finish_collection_evidence(
        root,
        reasons,
        grouped,
        references,
        allowed_empty=frozenset(
            index for index, cycle in enumerate(identifiers) if cycle in empty
        ),
        registered_paths=frozenset(row.path for row in registry.attempts),
        disk=disk,
    )
    audit_reasons, report = audit_registered_collection(
        registry,
        cycles=[
            (cycle, cycle_slot(at, start))
            for cycle, at in zip(identifiers, starts, strict=True)
        ],
        command_cycles={
            cycle for cycle, kinds in zip(identifiers, grouped, strict=True) if kinds
        },
        gap_cycles={
            cycle
            for cycle, kinds in zip(identifiers, grouped, strict=True)
            if "gap" in kinds
        },
        consumed=references,
        disk=disk,
        start_at=start,
    )
    return list(dict.fromkeys([*reasons, *audit_reasons])), report


def reused_seal_reasons(protocol_sha256: str, lineage: dict[str, Any]) -> list[str]:
    """A successor screen never reuses the protocol digest of an earlier attempt."""
    earlier = {ATTEMPT_ONE_PROTOCOL_SHA256, lineage["predecessor"]["protocol_sha256"]}
    return ["predecessor_seal_reused"] if protocol_sha256 in earlier else []


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
    probe_plan = study.get("research_plan")
    if (
        isinstance(probe_plan, dict)
        and probe_plan.get("schema") == STORAGE_PROBE_PLAN_SCHEMA
    ):
        raise ValueError("a storage probe root is never an evaluable screen")
    start = timestamp(protocol["settings"]["start_at"])
    close = start + timedelta(days=90)
    if protocol["settings"] != PaperSettings(start_at=start, close_at=close).to_dict():
        raise ValueError("paper study differs from fixed prospective settings")
    research_plan = study.get("research_plan")
    registered_screen = (
        isinstance(research_plan, dict)
        and research_plan.get("schema") == SCREEN_V3_SCHEMA
    )
    try:
        if not isinstance(research_plan, dict):
            raise ValueError("paper research plan must be an object")
        if registered_screen:
            lineage = validate_attempt_lineage_v4(
                research_plan.get("attempt_lineage"),
                sealed_at=timestamp(study["created_at"]),
            )
            expected_plan = screen_plan_v3(attempt_lineage=lineage)
        else:
            lineage = validate_attempt_lineage(
                research_plan.get("attempt_lineage"),
                sealed_at=timestamp(study["created_at"]),
            )
            expected_plan = screen_plan(attempt_lineage=lineage)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("paper attempt lineage is invalid") from error
    if (
        study.get("schema") != "paper_public_collection_v1"
        or not same_canonical_record(research_plan, expected_plan)
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
        observations = _ObservationAccumulator(start_at=start, close_at=close)
        audit_events: list[tuple[str, dict[str, Any]]] = []
        for record in engine.journal.iter_events():
            observations.add(record)
            event = record["event"]
            audit_events.append((event["kind"], event["payload"]["request"]))
        measured = observations.finish(state, engine.unsettled_funding())
        collection_report: dict[str, Any] | None = None
        if registered_screen:
            reasons, collection_report = (
                _registered_collection_evidence_from_compact_events(
                    destination,
                    expected_protocol_sha256,
                    audit_events,
                    start=start,
                    end=end,
                )
            )
            reasons.extend(reused_seal_reasons(expected_protocol_sha256, lineage))
        else:
            reasons = _collection_evidence_reasons_from_compact_events(
                destination,
                expected_protocol_sha256,
                audit_events,
                start=start,
                end=end,
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
        if registered_screen:
            return dict(
                schema="carry_paper_assessment_v3",
                screen_schema=SCREEN_V3_SCHEMA,
                protocol_sha256=expected_protocol_sha256,
                attempt_lineage=lineage,
                final_tip=expected_tip,
                assessed_at=now.isoformat(),
                decision="PAPER_SCREEN_PASSED"
                if not reasons
                else "PAPER_SCREEN_REJECTED",
                reasons=reasons,
                measurements=asdict(measured),
                collection_report=collection_report,
                status=state,
                production_eligible=False,
                fee_headroom_is_dynamic_stress_replay=False,
            )
        return dict(
            schema="carry_paper_assessment_v2",
            protocol_sha256=expected_protocol_sha256,
            attempt_lineage=lineage,
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
