"""Attempt-2 predecessor-root status: the lost-root rule and its refusals.

Offline only. The v1 journal reader is replaced by a fake except in one test that
runs the real reader on a small synthetic root.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.evaluation.test_paper_attempt2 import (  # noqa: F401  (autouse fixture)
    AUTH,
    CLEAN,
    NOW,
    REPO,
    START,
    Recorder,
    clock,
    launch,
    offline,
    plan,
    refusal_of,
    roomy,
    seal,
)
from trade_rl.evaluation.paper import attempt2, operations
from trade_rl.evaluation.paper.attempt2 import (
    ATTEMPT_ONE_FINAL_TIP_SHA256,
    ATTEMPT_ONE_LAST_OBSERVED_AT,
    ATTEMPT_ONE_PROTOCOL_SHA256,
    attempt2_lineage,
    launch_attempt2,
    seal_attempt2,
)


def v1_report(**overrides):
    """What `collection_status` returns for an intact attempt-1 root."""
    report = dict(
        events=attempt2.ATTEMPT_ONE_EVENT_COUNT,
        tip=ATTEMPT_ONE_FINAL_TIP_SHA256,
        status=dict(observed_at=ATTEMPT_ONE_LAST_OBSERVED_AT),
    )
    report.update(overrides)
    return report


class FakeReader:
    def __init__(self, report=None, error=None):
        self.report = v1_report() if report is None else report
        self.error, self.calls = error, []

    def __call__(self, root, *, expected_protocol_sha256):
        self.calls.append((Path(root), expected_protocol_sha256))
        if self.error is not None:
            raise self.error
        return self.report


def available(tmp_path, reader=None, **overrides):
    root = tmp_path / "attempt1-root"
    root.mkdir(exist_ok=True)
    options = dict(
        predecessor_root_status="available",
        predecessor_root=root,
        read_predecessor=reader or FakeReader(),
    )
    options.update(overrides)
    return options


# ------------------------------------------------------------------ records


def test_unavailable_record_is_honest_and_quotes_the_user_verbatim():
    record = attempt2_lineage("unavailable")["predecessor_root"]
    assert record["status"] == "unavailable"
    assert record["re_verified_at_seal"] is False
    assert record["lineage_source"] == {
        "kind": "prior_result_blind_review_record",
        "document": "docs/research/current-status.md",
        "commit": "b9f1ca65b26dcff7acbb089c7643e95702d474c9",
    }
    assert record["user_attestation"] == {
        "quote": "そのデータはしらない。結果保存もどこにあるわからないから、新しくとっていいよ",
        "date": "2026-10-30",
        "channel": "chat",
    }
    rule = record["review_rule"]
    assert "evidence root is lost" in rule
    assert "prior recorded review" in rule
    assert "not re-verified" in rule
    assert "lineage documentation and the pull request" in rule
    for status in ("available", "unavailable"):
        predecessor = attempt2_lineage(status)["predecessor"]
        assert predecessor["disposition"] == "invalidated"
        assert predecessor["reason_codes"] == ["observation_gap"]
    status_document = (REPO / "docs/research/current-status.md").read_text("utf-8")
    assert "Fresh result-blind review independently verified" in status_document


def test_available_record_carries_the_verification_it_was_built_from():
    assert attempt2_lineage("available")["predecessor_root"] == {
        "status": "available",
        "verification": {
            "reader": "paper_journal_protocol_v1",
            "event_count": 188,
            "protocol_sha256": ATTEMPT_ONE_PROTOCOL_SHA256,
            "final_tip_sha256": ATTEMPT_ONE_FINAL_TIP_SHA256,
            "last_observed_at": ATTEMPT_ONE_LAST_OBSERVED_AT,
        },
    }
    with pytest.raises(ValueError):
        attempt2_lineage("maybe")


def test_both_lineages_validate_and_v2_is_unchanged():
    for status in ("available", "unavailable"):
        operations.validate_attempt_lineage(attempt2_lineage(status), sealed_at=NOW)
    legacy = attempt2_lineage("unavailable")
    legacy.pop("predecessor_root")
    legacy["schema"] = "carry_paper_attempt_lineage_v2"
    assert operations.validate_attempt_lineage(legacy, sealed_at=NOW) == legacy


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v.pop("predecessor_root"),
        lambda v: v.update(schema="carry_paper_attempt_lineage_v2"),
        lambda v: v.update(attempt_number=1, predecessor=None),
        lambda v: v["predecessor_root"].update(status="verified"),
        lambda v: v["predecessor_root"].pop("verification"),
        lambda v: v["predecessor_root"]["verification"].update(reader="other"),
        lambda v: v["predecessor_root"]["verification"].update(
            final_tip_sha256="0" * 64
        ),
        lambda v: v["predecessor_root"]["verification"].update(event_count=0),
    ],
)
def test_a_hand_made_available_claim_without_matching_verification_is_invalid(mutate):
    lineage = attempt2_lineage("available")
    mutate(lineage)
    with pytest.raises(ValueError):
        operations.validate_attempt_lineage(lineage, sealed_at=NOW)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(re_verified_at_seal=True),
        lambda r: r.pop("user_attestation"),
        lambda r: r["user_attestation"].update(quote=""),
        lambda r: r["user_attestation"].update(date="2026-13-40"),
        lambda r: r["user_attestation"].update(date="2999-01-01"),
        lambda r: r["lineage_source"].update(commit="abc"),
        lambda r: r["lineage_source"].update(kind="root_read"),
        lambda r: r.update(review_rule=""),
        lambda r: r.update(extra=1),
    ],
)
def test_an_incomplete_or_overclaiming_unavailable_record_is_invalid(mutate):
    lineage = attempt2_lineage("unavailable")
    mutate(lineage["predecessor_root"])
    with pytest.raises(ValueError):
        operations.validate_attempt_lineage(lineage, sealed_at=NOW)


# ------------------------------------------------------------- plan and seal


def test_plan_and_seal_require_an_explicit_known_status(tmp_path):
    for call in (plan, seal):
        assert refusal_of(call, tmp_path=tmp_path, predecessor_root_status=None) == [
            "predecessor_root_status_missing"
        ]
        assert refusal_of(call, tmp_path=tmp_path, predecessor_root_status="lost") == [
            "predecessor_root_status_invalid"
        ]
    assert not (tmp_path / "attempt2").exists()


def test_unavailable_plan_records_the_rule_and_never_claims_reverification(tmp_path):
    draft = plan(tmp_path)
    assert draft["predecessor_root_status"] == "unavailable"
    assert draft["predecessor_root_reverified_at_seal"] is False
    assert draft["attempt_lineage"]["predecessor_root"]["status"] == "unavailable"
    prerequisite = " ".join(draft["human_prerequisites"])
    assert "lineage documentation and the pull request" in prerequisite
    assert "G0 review of the preserved attempt-1 root" not in prerequisite
    seal_step = next(s for s in draft["steps"] if s["step"] == "seal")
    assert "--predecessor-root-status unavailable" in seal_step["command"]
    assert "--predecessor-root " not in seal_step["command"]


def test_unavailable_cannot_be_combined_with_a_root_path(tmp_path):
    options = available(tmp_path, predecessor_root_status="unavailable")
    for call in (plan, seal):
        assert refusal_of(call, tmp_path=tmp_path, **options) == [
            "predecessor_root_given_but_declared_unavailable"
        ]
    assert options["read_predecessor"].calls == []


def test_available_without_a_root_path_is_refused(tmp_path):
    for call in (plan, seal):
        assert refusal_of(
            call,
            tmp_path=tmp_path,
            predecessor_root_status="available",
            predecessor_root=None,
        ) == ["predecessor_root_required_for_available"]
    assert not (tmp_path / "attempt2").exists()


def test_available_plan_reads_the_root_with_the_v1_reader_and_records_it(tmp_path):
    reader = FakeReader()
    draft = plan(tmp_path, **available(tmp_path, reader))
    assert reader.calls == [(tmp_path / "attempt1-root", ATTEMPT_ONE_PROTOCOL_SHA256)]
    assert draft["attempt_lineage"] == attempt2_lineage("available")
    assert draft["predecessor_root_reverified_at_seal"] is True
    seal_step = next(s for s in draft["steps"] if s["step"] == "seal")
    assert "--predecessor-root-status available" in seal_step["command"]
    assert "--predecessor-root <ATTEMPT_1_ROOT>" in seal_step["command"]


def test_available_seal_reads_the_root_and_binds_the_verification(tmp_path):
    reader = FakeReader()
    result = seal(tmp_path, **available(tmp_path, reader))
    assert reader.calls == [(tmp_path / "attempt1-root", ATTEMPT_ONE_PROTOCOL_SHA256)]
    assert result["predecessor_root_status"] == "available"
    assert result["predecessor_root_reverified_at_seal"] is True
    protocol = json.loads((tmp_path / "attempt2" / "protocol.json").read_bytes())
    lineage = protocol["protocol"]["study"]["research_plan"]["attempt_lineage"]
    assert lineage == attempt2_lineage("available")


@pytest.mark.parametrize(
    ("reader", "reason"),
    [
        (FakeReader(v1_report(events=187)), "predecessor_root_event_count_differs"),
        (FakeReader(v1_report(tip="0" * 64)), "predecessor_root_tip_differs"),
        (
            FakeReader(v1_report(status=dict(observed_at="2026-09-17T23:11:02+00:00"))),
            "predecessor_root_last_observation_differs",
        ),
        (
            FakeReader(v1_report(status=dict(observed_at=None))),
            "predecessor_root_last_observation_differs",
        ),
        (
            FakeReader(error=ValueError("paper protocol digest mismatch")),
            "predecessor_root_unreadable:ValueError",
        ),
        (FakeReader(report={}), "predecessor_root_unreadable:KeyError"),
    ],
)
def test_available_is_refused_when_the_root_does_not_match_the_record(
    tmp_path, reader, reason
):
    for call in (plan, seal):
        assert refusal_of(call, tmp_path=tmp_path, **available(tmp_path, reader)) == [
            reason
        ]
    assert not (tmp_path / "attempt2").exists()


def test_available_is_refused_for_a_missing_root(tmp_path):
    reader = FakeReader()
    options = available(tmp_path, reader, predecessor_root=tmp_path / "nowhere")
    assert refusal_of(seal, tmp_path=tmp_path, **options) == [
        "predecessor_root_not_a_directory"
    ]
    assert reader.calls == []


def test_the_real_v1_reader_verifies_a_synthetic_root(tmp_path, monkeypatch):
    from trade_rl.evaluation.paper.store import PaperJournal

    root = tmp_path / "old"
    digest = PaperJournal.initialize(root, {"engine": "synthetic"})
    journal = PaperJournal(root, expected_protocol_sha256=digest)
    parent = digest
    for index in range(3):
        record = journal.append(
            f"k{index}",
            "decide",
            {"request": {}, "result": {"observed_at": ATTEMPT_ONE_LAST_OBSERVED_AT}},
            expected_parent=parent,
        )
        parent = record["sha256"]
    monkeypatch.setattr(attempt2, "ATTEMPT_ONE_PROTOCOL_SHA256", digest)
    monkeypatch.setattr(attempt2, "ATTEMPT_ONE_FINAL_TIP_SHA256", parent)
    monkeypatch.setattr(attempt2, "ATTEMPT_ONE_EVENT_COUNT", 3)
    assert attempt2.verify_predecessor_root(root) == []
    monkeypatch.setattr(attempt2, "ATTEMPT_ONE_EVENT_COUNT", 188)
    assert attempt2.verify_predecessor_root(root) == [
        "predecessor_root_event_count_differs"
    ]
    monkeypatch.setattr(attempt2, "ATTEMPT_ONE_EVENT_COUNT", 3)
    monkeypatch.setattr(attempt2, "ATTEMPT_ONE_PROTOCOL_SHA256", "f" * 64)
    assert attempt2.verify_predecessor_root(root) == [
        "predecessor_root_unreadable:ValueError"
    ]


def test_unavailable_seal_records_the_status_in_the_protocol(tmp_path):
    result = seal(tmp_path)
    assert result["predecessor_root_status"] == "unavailable"
    assert result["predecessor_root_reverified_at_seal"] is False
    protocol = json.loads((tmp_path / "attempt2" / "protocol.json").read_bytes())
    lineage = protocol["protocol"]["study"]["research_plan"]["attempt_lineage"]
    assert lineage == attempt2_lineage("unavailable")


# ------------------------------------------------------------------- launch


def test_launch_accepts_either_recorded_status(tmp_path):
    for status in ("unavailable", "available"):
        base = tmp_path / status
        base.mkdir()
        options = {} if status == "unavailable" else available(base)
        result = seal_attempt2(
            base / "attempt2",
            start_at=START,
            authorization=AUTH,
            bytes_per_observation=2_000_000,
            predecessor_root_status=status,
            clock=clock,
            code_identity=lambda: CLEAN,
            disk_usage=roomy,
            **{k: v for k, v in options.items() if k != "predecessor_root_status"},
        )
        recorder = Recorder()
        outcome = launch_attempt2(
            base / "attempt2",
            protocol_sha256=result["protocol_sha256"],
            start_at=START,
            authorization=AUTH,
            allow_network=True,
            log_file=base / "child.log",
            clock=clock,
            code_identity=lambda: CLEAN,
            launcher=recorder.launcher,
            supervise=recorder.supervise,
        )
        assert outcome["exit_code"] == 0
        assert recorder.calls[-1][0] == "supervise"


def test_launch_refuses_a_tampered_unavailable_record(tmp_path):
    lineage = attempt2_lineage("unavailable")
    lineage["predecessor_root"]["user_attestation"]["quote"] = "different words"
    digest = operations.seal_paper_study(
        tmp_path / "attempt2", start_at=START, attempt_lineage=lineage, clock=clock
    )
    recorder = Recorder()
    assert refusal_of(lambda **kw: launch(tmp_path, digest, recorder, **kw)) == [
        "protocol_not_attempt2_lineage"
    ]
    assert recorder.calls == []


def test_cli_plan_refuses_available_without_a_root(tmp_path, capsys):
    start = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    code = attempt2.main(
        [
            "plan",
            "--root",
            str(tmp_path / "attempt2"),
            "--start-at",
            start,
            "--authorization",
            AUTH,
            "--predecessor-root-status",
            "available",
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().err)["refused"] == [
        "predecessor_root_required_for_available"
    ]
    assert not (tmp_path / "attempt2").exists()
