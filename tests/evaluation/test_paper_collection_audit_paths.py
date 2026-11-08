"""Registry paths of the real collector that the evaluator audit must accept.

An unused capture of a stale-rules skip and a failed execution capture are produced by
the real ``PaperCollector`` over a fake venue; the audit must find their registry rows
consistent, while the permanent quality failures they cause still reject the root.
"""

from __future__ import annotations

from tests.evaluation.test_paper_collection_audit import records_of
from tests.evaluation.test_paper_collection_retry import (
    at,
    build,
    collector,
    http_failure,
    rows,
    running,
)
from trade_rl.evaluation.paper import assessment

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
