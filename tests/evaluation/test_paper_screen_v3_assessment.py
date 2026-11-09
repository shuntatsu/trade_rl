"""``evaluate_paper_study`` under screen v3: dispatch, report and v2 compatibility.

Synthetic lineage identities and tmp roots only. The registered-failure audit itself is
tested in ``test_paper_collection_audit.py``; here the evaluator is shown to select the
audit by the sealed plan's schema, to keep screen v2 byte-for-byte as it was and never
to turn a v2 or a v4-shaped mismatch into a qualification.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.integrations.test_binance_forward import NOW
from trade_rl.evaluation.paper import assessment, supervisor
from trade_rl.evaluation.paper import predecessor as pred
from trade_rl.evaluation.paper.engine import PaperSettings
from trade_rl.evaluation.paper.operations import (
    screen_plan,
    screen_plan_v3,
    seal_paper_study,
)
from trade_rl.evaluation.paper.store import PaperJournal
from trade_rl.evaluation.paper.supervisor import PaperCollector
from trade_rl.evaluation.runs import build_candidate_run_provenance

ATTEMPT2_DIGEST = hashlib.sha256(b"synthetic attempt-2 protocol").hexdigest()
FIRST_ATTEMPT_LINEAGE = {
    "schema": "carry_paper_attempt_lineage_v2",
    "attempt_number": 1,
    "predecessor": None,
}


def v4(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = dict(
        protocol_sha256=ATTEMPT2_DIGEST,
        final_tip_sha256=hashlib.sha256(b"synthetic attempt-2 tip").hexdigest(),
        last_observed_at="2026-09-18T00:00:00+00:00",
        event_count=12,
        root_path="C:\\synthetic\\attempt2",
    )
    values.update(overrides)
    return pred.canonical_lineage_v4(**values)


def seal_v3(root: Path) -> tuple[str, Any]:
    start = NOW + timedelta(minutes=5)
    digest = seal_paper_study(
        root,
        start_at=start,
        attempt_lineage=v4(),
        lineage_seal_token=pred._ATTEMPT3_SEAL_TOKEN,
        clock=lambda: NOW,
    )
    return digest, start


def create_collection_files(root: Path, digest: str, monkeypatch: Any) -> None:
    """Open and close a collector so the lock and control files exist.

    The collector re-reads the predecessor root on every entry (tested elsewhere with
    a fake reader); this test has no attempt-2 root, so that one gate is stubbed.
    """
    monkeypatch.setattr(supervisor, "collection_lineage_problems", lambda *a, **k: [])
    with PaperCollector(root, expected_protocol_sha256=digest):
        pass


def arguments(digest: str, start: Any) -> dict[str, Any]:
    return dict(
        expected_protocol_sha256=digest,
        expected_tip=digest,
        clock=lambda: start + timedelta(days=90, seconds=180),
    )


def test_a_screen_v3_root_is_assessed_with_the_registered_audit_and_a_skip_report(
    tmp_path, monkeypatch
):
    root = tmp_path / "study"
    digest, start = seal_v3(root)
    create_collection_files(root, digest, monkeypatch)

    report = assessment.evaluate_paper_study(root, **arguments(digest, start))

    assert report["schema"] == "carry_paper_assessment_v3"
    assert report["screen_schema"] == "carry_paper_screen_v3"
    assert report["decision"] == "PAPER_SCREEN_REJECTED"
    assert report["production_eligible"] is False
    assert "initial_coverage_gap" in report["reasons"]
    assert "missing_instrument_fills" in report["reasons"]
    assert report["attempt_lineage"] == v4()
    collection = report["collection_report"]
    assert collection["schema"] == "carry_paper_collection_report_v1"
    assert collection["skipped_slot_count"] == 0
    assert collection["funding_settlement_hours_utc"] == [0, 8, 16]
    assert "predecessor_seal_reused" not in report["reasons"]


def test_screen_v2_assessment_keeps_its_schema_and_has_no_collection_report(tmp_path):
    root = tmp_path / "study"
    start = NOW + timedelta(minutes=5)
    digest = seal_paper_study(
        root, start_at=start, attempt_lineage=FIRST_ATTEMPT_LINEAGE, clock=lambda: NOW
    )
    with PaperCollector(root, expected_protocol_sha256=digest):
        pass

    report = assessment.evaluate_paper_study(root, **arguments(digest, start))

    assert report["schema"] == "carry_paper_assessment_v2"
    assert "collection_report" not in report and "screen_schema" not in report


def test_a_registered_failed_capture_does_not_reject_screen_v3_but_does_reject_v2(
    tmp_path, monkeypatch
):
    """The same registry rows: v3 accepts a registered failure, v2 still rejects it."""
    import json
    import sqlite3

    root = tmp_path / "study"
    digest, start = seal_v3(root)
    create_collection_files(root, digest, monkeypatch)
    # One registered failed capture directory (no cycle needed: it is an attempt row).
    capture = root / "sources" / "2026-09-18" / "failed"
    capture.mkdir(parents=True)
    (capture / "failure.json").write_text(json.dumps({"eligible": False}))
    with sqlite3.connect(root / "collection-control.sqlite") as connection:
        connection.execute(
            "INSERT INTO cycles VALUES(1,?,1)",
            ((start + timedelta(seconds=1)).isoformat(),),
        )
        connection.execute(
            "INSERT INTO attempts(cycle_id,slot,ordinal,purpose,path,outcome,"
            "error_type,error) VALUES(1,0,1,'snapshot',?,'failed','X','x')",
            ("sources/2026-09-18/failed",),
        )
        connection.execute("INSERT INTO skips VALUES(0,1,'snapshot_unavailable')")

    v3 = assessment.evaluate_paper_study(root, **arguments(digest, start))
    assert "source_capture_failure" not in v3["reasons"]
    assert "unconsumed_or_missing_sources" not in v3["reasons"]
    assert "cycle_command_mismatch" not in v3["reasons"]
    assert v3["collection_report"]["skipped_slots"] == [0]

    # The v2 audit over the very same root rejects both the failure and the empty cycle.
    reasons = assessment.collection_evidence_reasons(
        root, digest, [], start=start, end=start + timedelta(days=91)
    )
    assert "source_capture_failure" in reasons
    assert "cycle_command_mismatch" in reasons


def test_a_failure_marker_of_any_class_rejects_screen_v3(tmp_path, monkeypatch):
    root = tmp_path / "study"
    digest, start = seal_v3(root)
    create_collection_files(root, digest, monkeypatch)
    for failure_class in ("D", "I", "B"):
        marker = root / "collection-failure.json"
        marker.write_text('{"failure_class": "%s"}' % failure_class)
        report = assessment.evaluate_paper_study(root, **arguments(digest, start))
        assert "collection_failure" in report["reasons"], failure_class
        assert report["decision"] == "PAPER_SCREEN_REJECTED"
        marker.unlink()


def test_a_v3_protocol_whose_lineage_is_not_v4_is_refused(tmp_path):
    start = NOW + timedelta(minutes=5)
    plan = screen_plan_v3(attempt_lineage=v4())
    plan["attempt_lineage"] = pred.canonical_lineage("unavailable")
    settings = PaperSettings(start_at=start, close_at=start + timedelta(days=90))
    digest = PaperJournal.initialize(
        tmp_path / "mixed",
        dict(
            engine="funding_carry_paper_engine_v1",
            settings=settings.to_dict(),
            study=dict(
                schema="paper_public_collection_v1",
                created_at=NOW.isoformat(),
                interval_seconds=60,
                terminal_grace_seconds=180,
                provenance=build_candidate_run_provenance(),
                research_plan=plan,
            ),
        ),
    )
    with pytest.raises(ValueError, match="lineage is invalid"):
        assessment.evaluate_paper_study(tmp_path / "mixed", **arguments(digest, start))


def test_a_v2_plan_carrying_a_v4_lineage_is_refused(tmp_path):
    start = NOW + timedelta(minutes=5)
    plan = screen_plan(attempt_lineage=FIRST_ATTEMPT_LINEAGE)
    plan["attempt_lineage"] = v4()
    settings = PaperSettings(start_at=start, close_at=start + timedelta(days=90))
    digest = PaperJournal.initialize(
        tmp_path / "mixed2",
        dict(
            engine="funding_carry_paper_engine_v1",
            settings=settings.to_dict(),
            study=dict(
                schema="paper_public_collection_v1",
                created_at=NOW.isoformat(),
                interval_seconds=60,
                terminal_grace_seconds=180,
                provenance=build_candidate_run_provenance(),
                research_plan=plan,
            ),
        ),
    )
    with pytest.raises(ValueError, match="lineage is invalid"):
        assessment.evaluate_paper_study(tmp_path / "mixed2", **arguments(digest, start))


@pytest.mark.parametrize(
    "name,value",
    [
        ("MAX_SKIPPED_SLOTS_TOTAL", 500),
        ("MAX_SKIPPED_SLOTS_PER_HOUR", 5),
        ("MAX_CONSECUTIVE_SKIPPED_SLOTS", 2),
        ("RETRIES_PER_SLOT", 5),
    ],
)
def test_a_budget_changed_after_the_seal_is_not_the_sealed_screen(
    tmp_path, monkeypatch, name, value
):
    """The sealed plan carries the budgets, so a later change cannot qualify a root."""
    from trade_rl.evaluation.paper import operations

    root = tmp_path / "study"
    digest, start = seal_v3(root)
    create_collection_files(root, digest, monkeypatch)
    monkeypatch.setattr(operations, name, value)

    with pytest.raises(ValueError, match="differs from the frozen economic protocol"):
        assessment.evaluate_paper_study(root, **arguments(digest, start))


def test_a_reused_predecessor_seal_rejects_the_screen(tmp_path):
    """A protocol digest equal to the predecessor's is a reused seal."""
    from trade_rl.evaluation.paper.assessment import reused_seal_reasons

    lineage = v4()
    assert reused_seal_reasons(ATTEMPT2_DIGEST, lineage) == ["predecessor_seal_reused"]
    assert reused_seal_reasons(pred.ATTEMPT_ONE_PROTOCOL_SHA256, lineage) == [
        "predecessor_seal_reused"
    ]
    assert reused_seal_reasons("e" * 64, lineage) == []
