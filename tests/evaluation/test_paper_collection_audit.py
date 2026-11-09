"""Evaluator audit of registered collection failures (``carry_paper_screen_v3``).

Every root is produced by the real ``PaperCollector`` over a fake venue (tmp dirs, no
network) and then audited; defects are injected into the durable registry or the source
directories. The oracle is the pre-registered policy: registered failures inside the
budgets are accepted, everything else screen v2 rejected stays rejected, and a root with
no skipped slot audits exactly like v2.
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation.test_paper_collection_retry import (
    at,
    build,
    http_failure,
    rows,
    running,
)
from tests.evaluation.test_paper_supervisor import NOW
from trade_rl.evaluation.paper import assessment, collection_audit
from trade_rl.evaluation.paper import collection_policy as policy
from trade_rl.evaluation.paper.engine import PaperEngine


def records_of(root: Path, digest: str) -> list[dict[str, Any]]:
    return PaperEngine(root, expected_protocol_sha256=digest).journal.events()


def audit(root: Path, digest: str, *, slots: int = 8):
    return assessment.registered_collection_evidence(
        root,
        digest,
        records_of(root, digest),
        start=at(0, 0),
        end=at(slots, 0),
    )


def v2(root: Path, digest: str, *, slots: int = 8) -> list[str]:
    return assessment.collection_evidence_reasons(
        root, digest, records_of(root, digest), start=at(0, 0), end=at(slots, 0)
    )


def one_skipped_slot(tmp_path, monkeypatch):
    """Slot 0 trades, slot 1 fails its snapshot three times (skip), slot 2 observes."""
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        feed.snapshot_error = http_failure(503)
        clock.value = at(1)
        assert active.cycle()["phase"] == "skipped"
        feed.snapshot_error = None
        clock.value = at(2)
        active.cycle()
    return root, digest


def retried_slot(tmp_path, monkeypatch):
    """Slot 1: one transient failure, then the retry commits (no skipped slot)."""
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        feed.script = {feed.captures + 1: http_failure(503)}
        clock.value = at(1)
        assert active.cycle()["phase"] == "observed"
        clock.value = at(2)
        active.cycle()
    return root, digest


def clean_root(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        for slot in (1, 2, 3):
            clock.value = at(slot)
            active.cycle()
    return root, digest


def edit(root: Path, *statements: tuple[str, tuple[Any, ...]]) -> None:
    with sqlite3.connect(root / "collection-control.sqlite") as connection:
        for statement, parameters in statements:
            connection.execute(statement, parameters)


def test_registered_skip_and_failed_captures_within_budget_are_accepted(
    tmp_path, monkeypatch
):
    root, digest = one_skipped_slot(tmp_path, monkeypatch)

    reasons, report = audit(root, digest)

    assert reasons == []
    assert report["skipped_slot_count"] == 1
    assert report["skipped_slots"] == [1]
    assert report["skip_reasons"] == {"snapshot_unavailable": 1}
    assert len(rows(root, "attempts")) == 3
    # The v2 audit still rejects exactly this root.
    assert {"source_capture_failure", "cycle_command_mismatch"} <= set(v2(root, digest))


def test_a_retried_failure_without_a_skip_is_accepted_and_reports_zero_skips(
    tmp_path, monkeypatch
):
    root, digest = retried_slot(tmp_path, monkeypatch)

    reasons, report = audit(root, digest)

    assert reasons == []
    assert report["skipped_slot_count"] == 0
    assert len(rows(root, "attempts")) == 1
    assert "source_capture_failure" in v2(root, digest)


def test_the_worst_case_slot_of_four_failed_captures_is_accepted(tmp_path, monkeypatch):
    """A held rules refresh fails once (no retry) and the snapshot fails three times."""
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        captures = feed.captures
        feed.script = {
            captures + 1: http_failure(503),
            captures + 2: http_failure(503),
            captures + 3: http_failure(503),
            captures + 4: http_failure(503),
        }
        clock.value = at(55, 5.0)  # rules due for a refresh (3300 s) but still held
        assert active.cycle()["phase"] == "skipped"
    assert len(rows(root, "attempts")) == policy.MAX_FAILED_CAPTURES_PER_SLOT
    reasons, report = assessment.registered_collection_evidence(
        root,
        digest,
        records_of(root, digest),
        start=at(0, 0),
        end=at(60, 0),
    )
    # Slots 1-54 never ran, so this synthetic root breaches the budgets by itself; the
    # point here is that the per-slot allowance is not among the reasons.
    assert "retry_allowance_exceeded" not in reasons
    assert report["skipped_slot_count"] >= 1


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("incomplete", "incomplete_collection_cycle"),
        ("unused_source", "unconsumed_or_missing_sources"),
        ("failure_marker", "collection_failure"),
        ("empty_cycle", "cycle_command_mismatch"),
    ],
)
def test_screen_v3_equals_v2_on_a_root_without_skipped_slots(
    tmp_path, monkeypatch, defect, reason
):
    root, digest = clean_root(tmp_path, monkeypatch)
    assert audit(root, digest)[0] == v2(root, digest) == []
    if defect == "incomplete":
        edit(root, ("UPDATE cycles SET complete=0 WHERE id=3", ()))
    elif defect == "unused_source":
        (root / "sources/2026-09-18/unconsumed").mkdir(parents=True)
    elif defect == "failure_marker":
        (root / "collection-failure.json").write_text("{}")
    else:
        edit(
            root,
            (
                "INSERT INTO cycles VALUES(5,?,1)",
                ((at(3, 40)).isoformat(),),
            ),
        )

    registered_reasons, report = audit(root, digest)

    assert reason in v2(root, digest)
    assert registered_reasons == v2(root, digest)
    assert report is not None and report["skipped_slot_count"] == 0


def test_an_unregistered_failed_capture_is_rejected_as_in_v2(tmp_path, monkeypatch):
    root, digest = one_skipped_slot(tmp_path, monkeypatch)
    edit(root, ("DELETE FROM attempts", ()))

    reasons, _ = audit(root, digest)

    assert "source_capture_failure" in reasons
    assert "unconsumed_or_missing_sources" in reasons


def test_a_skipped_slot_without_a_registration_is_rejected(tmp_path, monkeypatch):
    root, digest = one_skipped_slot(tmp_path, monkeypatch)
    edit(root, ("DELETE FROM skips", ()))

    reasons, report = audit(root, digest)

    assert "unregistered_skipped_slot" in reasons
    assert "cycle_command_mismatch" in reasons
    # The audit re-derives the skip from the roster, so it is still counted.
    assert report["skipped_slots"] == [1]
    assert report["skip_reasons"] == {"unregistered": 1}


def test_a_slot_that_never_ran_without_registration_is_rejected(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        clock.value = at(2)  # slot 1 never ran
        active.cycle()
    assert [row["slot"] for row in rows(root, "skips")] == [1]
    assert audit(root, digest)[0] == []
    edit(root, ("DELETE FROM skips", ()))

    reasons, _ = audit(root, digest)

    assert "unregistered_skipped_slot" in reasons


def test_an_unregistered_skip_reason_is_rejected(tmp_path, monkeypatch):
    root, digest = one_skipped_slot(tmp_path, monkeypatch)
    edit(root, ("UPDATE skips SET reason='calm_market'", ()))

    assert "unregistered_skip_reason" in audit(root, digest)[0]


def test_every_registered_reason_and_attempt_purpose_is_accepted_by_the_audit(
    tmp_path, monkeypatch
):
    """The audit's accepted classes are exactly the collector's registered literals."""
    assert set(policy.SKIP_REASONS) == {
        "execution_capture_failed",
        "missed_slot",
        "rules_expired",
        "rules_unavailable",
        "snapshot_unavailable",
    }
    assert set(policy.ATTEMPT_PURPOSES) == {"execution", "rules", "snapshot"}


@pytest.mark.parametrize(
    "tamper",
    [
        "skip_in_the_future_cycle",
        "skip_owned_by_another_slot",
        "attempt_wrong_slot",
        "attempt_unknown_cycle",
        "attempt_bad_purpose",
    ],
)
def test_inconsistent_registry_rows_are_rejected(tmp_path, monkeypatch, tamper):
    root, digest = one_skipped_slot(tmp_path, monkeypatch)
    if tamper == "skip_in_the_future_cycle":
        edit(root, ("UPDATE skips SET reason='missed_slot'", ()))
    elif tamper == "skip_owned_by_another_slot":
        edit(root, ("UPDATE skips SET cycle_id=1", ()))
    elif tamper == "attempt_wrong_slot":
        edit(root, ("UPDATE attempts SET slot=slot+1", ()))
    elif tamper == "attempt_unknown_cycle":
        edit(root, ("UPDATE attempts SET cycle_id=77", ()))
    else:
        edit(root, ("UPDATE attempts SET purpose='telemetry'", ()))

    reasons = audit(root, digest)[0]

    assert any(
        reason in reasons
        for reason in ("inconsistent_skip_registry", "inconsistent_attempt_registry")
    )


def test_a_registered_capture_that_is_missing_or_consumed_is_rejected(
    tmp_path, monkeypatch
):
    root, digest = one_skipped_slot(tmp_path, monkeypatch)
    path = rows(root, "attempts")[0]["path"]
    shutil.rmtree(root / path)
    assert "registered_capture_missing" in audit(root, digest)[0]

    root, digest = one_skipped_slot(tmp_path / "b", monkeypatch)
    consumed = records_of(root, digest)[0]["event"]["payload"]["request"]["market"]
    edit(
        root,
        (
            "UPDATE attempts SET path=? WHERE ordinal=1",
            (consumed["path"],),
        ),
    )
    assert "registered_capture_consumed" in audit(root, digest)[0]


def test_a_failed_attempt_without_failure_evidence_is_rejected(tmp_path, monkeypatch):
    root, digest = one_skipped_slot(tmp_path, monkeypatch)
    path = rows(root, "attempts")[0]["path"]
    (root / path / "failure.json").unlink()

    assert "registered_capture_outcome_mismatch" in audit(root, digest)[0]


def test_an_unused_capture_must_belong_to_a_skipped_slot(tmp_path, monkeypatch):
    root, digest = one_skipped_slot(tmp_path, monkeypatch)
    path = rows(root, "attempts")[0]["path"]
    (root / path / "failure.json").unlink()
    edit(root, ("UPDATE attempts SET outcome='unused' WHERE ordinal=1", ()))
    assert audit(root, digest)[0] == []
    edit(root, ("DELETE FROM skips", ()))

    assert "inconsistent_attempt_registry" in audit(root, digest)[0]


def test_more_failed_captures_than_the_allowance_in_one_slot_is_rejected(
    tmp_path, monkeypatch
):
    root, digest = one_skipped_slot(tmp_path, monkeypatch)
    template = rows(root, "attempts")[0]
    donor = root / template["path"]
    for index in range(policy.MAX_FAILED_CAPTURES_PER_SLOT):
        clone = root / "sources" / "clone" / f"extra{index}"
        shutil.copytree(donor, clone)
        edit(
            root,
            (
                "INSERT INTO attempts(cycle_id,slot,ordinal,purpose,path,outcome,"
                "error_type,error) VALUES(?,?,?,?,?,?,?,?)",
                (
                    template["cycle_id"],
                    template["slot"],
                    10 + index,
                    "snapshot",
                    clone.relative_to(root).as_posix(),
                    "failed",
                    "X",
                    "x",
                ),
            ),
        )

    assert "retry_allowance_exceeded" in audit(root, digest)[0]


def test_a_missing_registry_is_rejected_not_treated_as_no_skips(tmp_path, monkeypatch):
    root, digest = clean_root(tmp_path, monkeypatch)
    edit(root, ("DROP TABLE attempts", ()), ("DROP TABLE skips", ()))

    reasons, report = audit(root, digest)

    assert reasons == ["missing_collection_registry"]
    assert report is None


def test_each_budget_breach_is_rejected_from_the_roster_alone(tmp_path, monkeypatch):
    """Delete the marker the collector wrote: the audit still finds the breach."""
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        feed.snapshot_error = http_failure(503)
        clock.value = at(1)
        assert active.cycle()["phase"] == "skipped"
        clock.value = at(2)
        with pytest.raises(policy.SkipBudgetExceeded):
            active.cycle()
    (root / "collection-failure.json").unlink()

    reasons, report = audit(root, digest)

    assert "skip_budget_consecutive" in reasons
    assert report["skipped_slots"] == [1, 2]


@pytest.mark.parametrize(
    "slots,budget",
    [
        ([10, 20, 30], "per_hour"),
        # Thirty slots apart: never more than two in sixty, so only the total breaches.
        (
            [1 + 30 * index for index in range(policy.MAX_SKIPPED_SLOTS_TOTAL + 1)],
            "total",
        ),
    ],
)
def test_per_hour_and_total_budget_breaches_are_rejected(slots, budget):
    last = max(slots) + 1
    cycles = [(slot + 1, slot) for slot in range(0, last + 1)]
    registry = collection_audit.Registry(
        attempts=(),
        skips=tuple(
            collection_audit.SkipRow(slot, slot + 1, "snapshot_unavailable")
            for slot in slots
        ),
    )
    reasons, report = collection_audit.audit_registered_collection(
        registry,
        cycles=cycles,
        command_cycles={cycle for cycle, slot in cycles if slot not in slots},
        gap_cycles=set(),
        consumed=set(),
        disk={},
        start_at=datetime(2026, 11, 9, tzinfo=UTC),
    )
    assert f"skip_budget_{budget}" in reasons
    assert report["skipped_slot_count"] == len(slots)


def test_the_129th_skip_is_inside_the_budget_and_the_130th_is_not():
    spread = [1 + 30 * index for index in range(130)]
    cycles = [(slot + 1, slot) for slot in range(0, spread[-1] + 2)]
    for count, expected in ((policy.MAX_SKIPPED_SLOTS_TOTAL, False), (130, True)):
        slots = spread[:count]
        registry = collection_audit.Registry(
            attempts=(),
            skips=tuple(
                collection_audit.SkipRow(slot, slot + 1, "snapshot_unavailable")
                for slot in slots
            ),
        )
        reasons, _ = collection_audit.audit_registered_collection(
            registry,
            cycles=cycles,
            command_cycles={c for c, slot in cycles if slot not in slots},
            gap_cycles=set(),
            consumed=set(),
            disk={},
            start_at=datetime(2026, 11, 9, tzinfo=UTC),
        )
        assert ("skip_budget_total" in reasons) is expected, count
        assert "skip_budget_per_hour" not in reasons


def test_the_report_places_skipped_slots_relative_to_funding_settlements():
    start = datetime(2026, 11, 9, 0, 0, tzinfo=UTC)
    # Slot starts: 07:59 (before 08:00), 08:00:00 (at), 08:02 (after, within 180 s),
    # 12:30 (far), 15:30 (between buckets), 23:59 (before the 00:00 settlement).
    minutes = [
        7 * 60 + 59,
        8 * 60,
        8 * 60 + 2,
        12 * 60 + 30,
        15 * 60 + 30,
        23 * 60 + 59,
    ]
    report = collection_audit.skip_report(
        minutes,
        dict.fromkeys(minutes, "snapshot_unavailable"),
        start_at=start,
        last_slot=24 * 60,
    )

    assert report["skipped_slot_count"] == 6
    assert report["skipped_by_proximity"] == {
        "within_180s": 4,
        "within_900s": 0,
        "within_3600s": 1,
        "beyond_3600s": 1,
    }
    assert report["skipped_by_nearest_settlement_utc_hour"] == {
        "00": 1,
        "08": 3,
        "16": 2,
    }
    assert report["skipped_before_nearest_settlement"] == 4
    assert report["skipped_at_or_after_nearest_settlement"] == 2
    assert sum(report["all_slots_by_proximity"].values()) == 24 * 60 + 1
    assert report["funding_settlement_hours_utc"] == [0, 8, 16]
    assert report["skip_reasons"] == {"snapshot_unavailable": 6}


def test_a_recorded_gap_cycle_does_not_count_its_missed_slots_twice():
    registry = collection_audit.Registry(attempts=(), skips=())
    cycles = [(1, 0), (2, 5)]
    reasons, report = collection_audit.audit_registered_collection(
        registry,
        cycles=cycles,
        command_cycles={1, 2},
        gap_cycles={2},
        consumed=set(),
        disk={},
        start_at=datetime(2026, 11, 9, tzinfo=UTC),
    )
    assert reasons == []
    assert report["skipped_slot_count"] == 0


def test_the_collection_clock_helpers_use_the_frozen_cadence():
    start = NOW + timedelta(seconds=1)
    assert collection_audit.cycle_slot(start + timedelta(seconds=59.9), start) == 0
    assert collection_audit.cycle_slot(start + timedelta(seconds=60), start) == 1
