"""Evaluator audit of the registered collection failures (``carry_paper_screen_v3``).

The collector (``supervisor``) registers every failed or unused capture directory and
every skipped slot in ``collection-control.sqlite`` (tables ``attempts`` and ``skips``,
see ``control``). Screen v2 rejected any root containing a failed capture or a skipped
slot. Screen v3 accepts them only when they are registered, use a registered class and
stay inside the pre-registered budgets of ``collection_policy``; everything else v2
rejected stays rejected.

This module is pure: it reads no files except through the read-only connection the
caller passes, never writes, and does not look at prices, equity or fills. The audit
re-derives the skipped slots from the cycle roster instead of trusting the ``skips``
table alone, so a slot that never ran cannot hide behind a missing registration.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from trade_rl.evaluation.paper.collection_policy import (
    ATTEMPT_PURPOSES,
    CADENCE_SECONDS,
    MAX_FAILED_CAPTURES_PER_SLOT,
    SKIP_REASONS,
    first_budget_breach,
)

REPORT_SCHEMA = "carry_paper_collection_report_v1"
MISSED_SLOT = "missed_slot"
# Funding settles at these UTC hours; the report positions every skipped slot, and every
# slot of the screen for comparison, relative to the nearest settlement.
FUNDING_SETTLEMENT_HOURS_UTC = (0, 8, 16)
PROXIMITY_BUCKETS = (
    ("within_180s", 180),
    ("within_900s", 900),
    ("within_3600s", 3600),
    ("beyond_3600s", None),
)
_DAY_SECONDS = 86_400
_ATTEMPT_OUTCOMES = ("failed", "unused")


@dataclass(frozen=True)
class AttemptRow:
    cycle_id: int
    slot: int
    ordinal: int
    purpose: str
    path: str
    outcome: str


@dataclass(frozen=True)
class SkipRow:
    slot: int
    cycle_id: int
    reason: str


@dataclass(frozen=True)
class Registry:
    attempts: tuple[AttemptRow, ...]
    skips: tuple[SkipRow, ...]


def read_registry(connection: sqlite3.Connection) -> Registry | None:
    """The registered attempts and skips, or ``None`` if the tables do not exist."""
    names = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if not {"attempts", "skips"} <= names:
        return None
    attempts = tuple(
        AttemptRow(*row)
        for row in connection.execute(
            "SELECT cycle_id,slot,ordinal,purpose,path,outcome FROM attempts "
            "ORDER BY id"
        )
    )
    skips = tuple(
        SkipRow(*row)
        for row in connection.execute(
            "SELECT slot,cycle_id,reason FROM skips ORDER BY slot"
        )
    )
    return Registry(attempts=attempts, skips=skips)


def cycle_slot(started_at: datetime, start_at: datetime) -> int:
    """The frozen sixty-second slot a cycle started in (as the collector computes it)."""
    return int((started_at - start_at).total_seconds() // CADENCE_SECONDS)


def allowed_empty_cycles(registry: Registry, cycle_ids: list[int]) -> set[int]:
    """Cycles that may carry no command: those that registered a skip of their own slot."""
    known = set(cycle_ids)
    return {
        row.cycle_id
        for row in registry.skips
        if row.reason != MISSED_SLOT and row.cycle_id in known
    }


def funding_proximity(slot_start: datetime) -> tuple[str, str, int]:
    """Nearest settlement of a slot start: (bucket, settlement hour label, signed seconds).

    The signed offset is the slot start minus the nearest settlement, so a negative
    value is before the settlement. Distances use the three daily settlements.
    """
    at = slot_start.astimezone(UTC)
    second_of_day = at.hour * 3600 + at.minute * 60 + at.second
    candidates = [hour * 3600 for hour in FUNDING_SETTLEMENT_HOURS_UTC] + [_DAY_SECONDS]
    nearest = min(candidates, key=lambda moment: abs(second_of_day - moment))
    offset = second_of_day - nearest
    label = f"{(nearest // 3600) % 24:02d}"
    distance = abs(offset)
    for name, limit in PROXIMITY_BUCKETS:
        if limit is None or distance <= limit:
            return name, label, offset
    raise AssertionError("unreachable proximity bucket")


def skip_report(
    skipped: list[int],
    reasons: dict[int, str],
    *,
    start_at: datetime,
    last_slot: int,
) -> dict[str, Any]:
    """Skip count and its funding-proximity distribution (the G3 claim boundary).

    ``skipped`` are the effective skipped slots; ``last_slot`` is the last slot that ran
    a cycle, so ``all_slots_by_proximity`` is the comparison population.
    """
    names = [name for name, _ in PROXIMITY_BUCKETS]
    by_bucket = dict.fromkeys(names, 0)
    by_hour = {f"{hour:02d}": 0 for hour in FUNDING_SETTLEMENT_HOURS_UTC}
    before = after = 0
    for slot in skipped:
        bucket, label, offset = funding_proximity(
            start_at + timedelta(seconds=slot * CADENCE_SECONDS)
        )
        by_bucket[bucket] += 1
        by_hour[label] += 1
        if offset < 0:
            before += 1
        else:
            after += 1
    everything = dict.fromkeys(names, 0)
    for slot in range(0, last_slot + 1):
        bucket, _, _ = funding_proximity(
            start_at + timedelta(seconds=slot * CADENCE_SECONDS)
        )
        everything[bucket] += 1
    return {
        "schema": REPORT_SCHEMA,
        "skipped_slot_count": len(skipped),
        "skipped_slots": list(skipped),
        "skip_reasons": dict(sorted(Counter(reasons.values()).items())),
        "slots_observed": last_slot + 1,
        "funding_settlement_hours_utc": list(FUNDING_SETTLEMENT_HOURS_UTC),
        "proximity_buckets_seconds": {name: limit for name, limit in PROXIMITY_BUCKETS},
        "skipped_by_proximity": by_bucket,
        "skipped_by_nearest_settlement_utc_hour": by_hour,
        "skipped_before_nearest_settlement": before,
        "skipped_at_or_after_nearest_settlement": after,
        "all_slots_by_proximity": everything,
    }


def audit_registered_collection(
    registry: Registry,
    *,
    cycles: list[tuple[int, int]],
    command_cycles: set[int],
    gap_cycles: set[int],
    consumed: set[str],
    disk: dict[str, bool],
    start_at: datetime,
) -> tuple[list[str], dict[str, Any]]:
    """Audit the registry against the cycle roster and the capture directories.

    ``cycles`` are ``(cycle id, slot)`` in cycle order; ``command_cycles`` the cycles
    that appended a decision or execution; ``gap_cycles`` the cycles that recorded a
    non-trading gap (already a permanent quality failure, so the slots they missed are
    not counted again, exactly as the collector does); ``consumed`` the capture
    directories referenced by a command; ``disk`` maps every capture directory to
    whether it holds ``failure.json``. Returns the reasons and the skip report.
    """
    reasons: list[str] = []
    slot_of = dict(cycles)
    skip_slots = {row.slot for row in registry.skips}
    command_slots = {slot_of[cycle] for cycle in command_cycles if cycle in slot_of}

    for row in registry.skips:
        owner = slot_of.get(row.cycle_id)
        if row.reason not in SKIP_REASONS:
            reasons.append("unregistered_skip_reason")
        if owner is None or (
            row.slot >= owner if row.reason == MISSED_SLOT else row.slot != owner
        ):
            reasons.append("inconsistent_skip_registry")
        if row.slot < 0:
            reasons.append("inconsistent_skip_registry")

    failed_per_slot: Counter[int] = Counter()
    for attempt in registry.attempts:
        if (
            attempt.outcome not in _ATTEMPT_OUTCOMES
            or attempt.purpose not in ATTEMPT_PURPOSES
            or slot_of.get(attempt.cycle_id) != attempt.slot
        ):
            reasons.append("inconsistent_attempt_registry")
        if attempt.path not in disk:
            reasons.append("registered_capture_missing")
        elif attempt.path in consumed:
            reasons.append("registered_capture_consumed")
        elif disk[attempt.path] != (attempt.outcome == "failed"):
            # A failed capture holds ``failure.json``; an unused one is a complete
            # capture that no command consumed.
            reasons.append("registered_capture_outcome_mismatch")
        if attempt.outcome == "unused" and attempt.slot not in skip_slots:
            reasons.append("inconsistent_attempt_registry")
        if attempt.outcome == "failed":
            failed_per_slot[attempt.slot] += 1
    if any(count > MAX_FAILED_CAPTURES_PER_SLOT for count in failed_per_slot.values()):
        reasons.append("retry_allowance_exceeded")

    # Independent derivation of the skipped slots: any slot from slot 0 up to the last
    # cycle that no command-bearing cycle covers never produced an observation, whether
    # or not the collector registered it (the engine anchors the last command at the
    # screen start, so slot 0 is not exempt). A cycle that recorded the frozen gap is
    # excluded: the gap is already a permanent quality failure. No cycle, no slot.
    last_slot = max(slot_of.values(), default=-1)
    required = set(range(0, last_slot + 1)) - command_slots
    previous_slot = -1
    for cycle_id, slot in cycles:
        if cycle_id in gap_cycles:
            required -= set(range(previous_slot + 1, slot))
        previous_slot = max(previous_slot, slot)
    if required - skip_slots:
        reasons.append("unregistered_skipped_slot")
    effective = sorted(skip_slots | required)
    breach = first_budget_breach(effective)
    if breach is not None:
        reasons.append(f"skip_budget_{breach[1]}")

    reason_of = {row.slot: row.reason for row in registry.skips}
    for slot in required - skip_slots:
        reason_of[slot] = "unregistered"
    report = skip_report(
        effective,
        reason_of,
        start_at=start_at,
        last_slot=last_slot,
    )
    return list(dict.fromkeys(reasons)), report
