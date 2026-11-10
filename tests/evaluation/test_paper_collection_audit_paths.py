"""Registry paths of the real collector that the evaluator audit must accept.

An unused capture of a stale-rules skip and a failed execution capture are produced by
the real ``PaperCollector`` over a fake venue; the audit must find their registry rows
consistent, while the permanent quality failures they cause still reject the root.
"""

from __future__ import annotations

from datetime import UTC, datetime

from tests.evaluation.test_paper_collection_audit import edit, records_of
from tests.evaluation.test_paper_collection_retry import (
    at,
    build,
    collector,
    http_failure,
    rows,
    running,
)
from trade_rl.evaluation.paper import assessment, collection_audit

REGISTRY_REASONS = {
    "unregistered_skipped_slot",
    "unregistered_skip_reason",
    "inconsistent_skip_registry",
    "inconsistent_attempt_registry",
    "registered_capture_missing",
    "registered_capture_consumed",
    "registered_capture_outcome_mismatch",
    "retry_allowance_exceeded",
    "missing_collection_registry",
    "skip_budget_consecutive",
    "skip_budget_per_hour",
    "skip_budget_total",
}


def test_a_snapshot_finished_after_the_rules_hold_is_an_unused_capture_of_a_skip(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        feed.script = {feed.captures + 1: http_failure(503)}  # the refresh fails
        feed.delay = 0.5
        clock.value = at(59, 58.5)  # the capture outlasts the one-hour hold
        assert active.cycle()["phase"] == "skipped"
    assert [(r["purpose"], r["outcome"]) for r in rows(root, "attempts")] == [
        ("rules", "failed"),
        ("snapshot", "unused"),
    ]

    reasons, report = assessment.registered_collection_evidence(
        root,
        digest,
        records_of(root, digest),
        start=at(0, 0),
        end=at(61, 0),
    )

    # The recorded observation gap and the missing hour of data still reject the root;
    # the registry itself (an unused capture of a registered skip) is consistent.
    assert not set(reasons) & REGISTRY_REASONS
    assert "cycle_command_mismatch" in reasons
    assert report["skipped_slots"] == [59]
    assert report["skip_reasons"] == {"rules_expired": 1}


def test_a_failed_execution_capture_is_a_registered_skip_and_the_decision_expires(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    feed.script = {3: http_failure(503)}  # rules, decision snapshot, execution snapshot
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0)
        assert active.cycle()["phase"] == "skipped"
        clock.value = at(1)
        assert active.cycle()["phase"] == "cancelled"

    reasons, report = assessment.registered_collection_evidence(
        root,
        digest,
        records_of(root, digest),
        start=at(0, 0),
        end=at(4, 0),
    )

    assert not set(reasons) & REGISTRY_REASONS
    assert report["skip_reasons"] == {"execution_capture_failed": 1}
    # The permanent expired-decision gap is not rescued by the skip budget: the cycle
    # that carries it is still rejected (and so is the account quality, in the replay).
    assert "cycle_command_mismatch" in reasons


def late_first_cycle(tmp_path, monkeypatch):
    """The collector starts late, inside the frozen 180 s limit: slot 0 never ran."""
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(1)
        active.cycle()
        clock.value = at(2)
        active.cycle()
    return root, digest


def test_a_slot_zero_that_never_ran_is_a_registered_skip_the_audit_counts(
    tmp_path, monkeypatch
):
    root, digest = late_first_cycle(tmp_path, monkeypatch)

    reasons, report = assessment.registered_collection_evidence(
        root, digest, records_of(root, digest), start=at(0, 0), end=at(5, 0)
    )

    assert not set(reasons) & REGISTRY_REASONS
    assert reasons == []
    assert report["skipped_slots"] == [0]
    assert report["skip_reasons"] == {"missed_slot": 1}


def test_an_unregistered_missing_slot_zero_is_flagged_and_still_counted(
    tmp_path, monkeypatch
):
    root, digest = late_first_cycle(tmp_path, monkeypatch)
    edit(root, ("DELETE FROM skips", ()))

    reasons, report = assessment.registered_collection_evidence(
        root, digest, records_of(root, digest), start=at(0, 0), end=at(5, 0)
    )

    assert "unregistered_skipped_slot" in reasons
    assert report["skipped_slots"] == [0]
    assert report["skip_reasons"] == {"unregistered": 1}


def test_a_first_cycle_in_slot_zero_leaves_nothing_to_register(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0)
        active.cycle()
        clock.value = at(1)
        active.cycle()

    reasons, report = assessment.registered_collection_evidence(
        root, digest, records_of(root, digest), start=at(0, 0), end=at(5, 0)
    )

    assert reasons == [] and report["skipped_slots"] == []


def test_slot_zero_is_required_from_the_roster_alone():
    start = datetime(2026, 11, 9, tzinfo=UTC)
    empty = collection_audit.Registry(attempts=(), skips=())
    reasons, report = collection_audit.audit_registered_collection(
        empty,
        cycles=[(1, 1), (2, 2)],
        command_cycles={1, 2},
        gap_cycles=set(),
        consumed=set(),
        disk={},
        start_at=start,
    )
    assert reasons == ["unregistered_skipped_slot"]
    assert report["skipped_slots"] == [0]
    # Slots 0 and 1 together are two consecutive skips: the budget is breached even
    # though no row was registered.
    reasons, report = collection_audit.audit_registered_collection(
        empty,
        cycles=[(1, 2)],
        command_cycles={1},
        gap_cycles=set(),
        consumed=set(),
        disk={},
        start_at=start,
    )
    assert "skip_budget_consecutive" in reasons
    assert report["skipped_slots"] == [0, 1]


def test_a_first_cycle_that_recorded_the_gap_condemns_slot_zero_only_once():
    start = datetime(2026, 11, 9, tzinfo=UTC)
    empty = collection_audit.Registry(attempts=(), skips=())
    reasons, report = collection_audit.audit_registered_collection(
        empty,
        cycles=[(1, 5)],
        command_cycles={1},
        gap_cycles={1},
        consumed=set(),
        disk={},
        start_at=start,
    )
    assert reasons == [] and report["skipped_slots"] == []


def test_a_root_with_no_cycle_has_no_required_slot():
    start = datetime(2026, 11, 9, tzinfo=UTC)
    empty = collection_audit.Registry(attempts=(), skips=())
    reasons, report = collection_audit.audit_registered_collection(
        empty,
        cycles=[],
        command_cycles=set(),
        gap_cycles=set(),
        consumed=set(),
        disk={},
        start_at=start,
    )
    assert reasons == [] and report["skipped_slots"] == []
