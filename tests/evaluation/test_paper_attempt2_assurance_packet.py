"""The attempt-2 research-assurance author packet stays UNREVIEWED and bound to the frozen plan.

The packet restates the existing ``carry_paper_screen_v2`` plan and the attempt-2
lineage v3 (``predecessor_root`` unavailable). It must never claim a review, a
pass, or any economic/live authorization, and any edit must change its digest.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance
from trade_rl.evaluation.paper.operations import screen_plan
from trade_rl.evaluation.paper.predecessor import (
    ATTEMPT_ONE_FINAL_TIP_SHA256,
    ATTEMPT_ONE_LAST_OBSERVED_AT,
    ATTEMPT_ONE_PROTOCOL_SHA256,
    LINEAGE_RECORD_COMMIT,
    UNAVAILABLE_NOT_ESTABLISHED,
    canonical_lineage,
)

ROOT = Path(__file__).resolve().parents[2]
RELATIVE_PACKET = "report/issue810-carry-attempt2-research-assurance-20261105.json"
PACKET = ROOT / RELATIVE_PACKET
PINNED_DIGEST = "f8faa1e083a3c15361d8834387e41c3ed4cc1ea600aec16b0a8a33eb338b9ca3"
AUTHORIZATION_URL = (
    "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-6659653178"
)
BOUNDARIES = {
    "unit",
    "time",
    "state",
    "sign",
    "risk",
    "execution",
    "accounting",
    "evidence",
}


def _packet() -> Any:
    return json.loads(PACKET.read_text(encoding="utf-8"))


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.agent_repo", "assurance", *args, RELATIVE_PACKET],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_packet_is_unreviewed_valid_and_pins_its_digest() -> None:
    packet = _packet()

    result = evaluate_assurance(packet)

    assert result.status == "UNREVIEWED"
    assert result.errors == ()
    assert result.economic_execution_authorized is False
    assert result.record_digest == assurance_digest(packet) == PINNED_DIGEST
    assert b"\r" not in PACKET.read_bytes()


def test_cli_check_inspect_and_digest_agree_and_report_unreviewed() -> None:
    check = _cli("check")
    inspect = _cli("inspect")
    digest = _cli("digest")

    assert check.returncode == 1, check.stderr
    assert json.loads(check.stdout)["status"] == "UNREVIEWED"
    assert inspect.returncode == 0, inspect.stderr
    inspected = json.loads(inspect.stdout)
    assert inspected["status"] == "UNREVIEWED"
    assert inspected["record_digest"] == PINNED_DIGEST
    assert inspected["economic_execution_authorized"] is False
    assert digest.returncode == 0, digest.stderr
    assert json.loads(digest.stdout) == {"record_digest": PINNED_DIGEST}


def test_any_edit_changes_the_digest_and_a_self_authored_review_stays_blocked() -> None:
    packet = _packet()
    edited = copy.deepcopy(packet)
    edited["thesis"]["stop_rule"] += " "
    assert assurance_digest(edited) != PINNED_DIGEST

    identity = packet["identity"]
    review = {
        "schema_version": "research_assurance_review_v1",
        "reviewed_record_digest": PINNED_DIGEST,
        "protocol_head": identity["protocol_head"],
        "implementation_head": identity["implementation_head"],
        "reviewer_identity": "self",
        "reviewer_execution_identity": "self-run",
        "reviewer_independence": "ESTABLISHED",
        "result_blind": True,
        "g0": "PASS",
        "g1": "PASS",
        "g2": "PASS",
        "status": "pass",
        "disposition": "ASSURANCE_PASS",
        "blocking_findings": [],
        "missing_evidence": [],
        "rationale": "author supplied",
        "limitations": ["none"],
    }
    result = evaluate_assurance(packet, review=review)
    assert result.status == "BLOCKED"
    assert result.economic_execution_authorized is False


def test_packet_grants_no_authorization_and_names_the_required_gates() -> None:
    packet = _packet()
    claims = packet["claims"]
    thesis = packet["thesis"]

    assert claims["production_eligible"] is False
    assert claims["live_trading_authorized"] is False
    assert claims["next_action_class"] == ("request_separate_economic_authorization")
    assert claims["claim_level"] == "software_validity"
    assert thesis["stage"] == "prospective_paper"
    text = json.dumps(packet)
    assert "Issue #810" in text
    assert AUTHORIZATION_URL in text
    assert "UNREVIEWED" in text
    assert "owner-accepted deviation" in text
    assert "different authenticated GitHub principal" in text
    assert "PASS" not in claims["next_authorized_action"]
    # Every boundary is covered by an oracle that cites existing test paths.
    mechanism = packet["mechanism"]
    assert set(mechanism["changed_boundaries"]) == BOUNDARIES
    cited = set(re.findall(r"tests/[\w/]+\.py", text))
    assert cited
    missing = sorted(path for path in cited if not (ROOT / path).is_file())
    assert missing == []


def test_packet_restates_the_frozen_screen_and_lineage_without_changing_them() -> None:
    lineage = canonical_lineage("unavailable")
    plan = screen_plan(attempt_lineage=lineage)
    packet = _packet()
    text = json.dumps(packet)
    rule = packet["claims"]["decision_rule"]

    assert plan["schema"] == "carry_paper_screen_v2"
    assert "carry_paper_screen_v2" in text
    assert plan["duration_days"] == 90 and "ninety-day" in text
    assert plan["block_count"] == 3 and "three thirty-day blocks" in rule
    assert plan["maximum_drawdown_exclusive"] == 0.1 and "below 10%" in rule
    assert plan["comparison"] == "cash_zero_interest"
    assert "cash_zero_interest" in packet["evidence"]["controls"]
    assert plan["production_eligible"] is False
    assert plan["minimum_seal_notice_seconds"] == 300 and "five minutes" in text

    assert lineage["schema"] == "carry_paper_attempt_lineage_v3"
    assert lineage["predecessor_root"]["status"] == "unavailable"
    assert lineage["predecessor_root"]["re_verified_at_seal"] is False
    for value in (
        ATTEMPT_ONE_PROTOCOL_SHA256,
        ATTEMPT_ONE_FINAL_TIP_SHA256,
        ATTEMPT_ONE_LAST_OBSERVED_AT,
        LINEAGE_RECORD_COMMIT,
        *UNAVAILABLE_NOT_ESTABLISHED,
    ):
        assert value in text
    assert "observation_gap" in text and "invalidated" in text


def test_docs_point_to_the_packet_digest_and_unreviewed_status() -> None:
    for relative in (
        "docs/specs/funding-carry-paper.md",
        "docs/architecture/research-assurance.md",
        "docs/research/current-status.md",
    ):
        document = (ROOT / relative).read_text(encoding="utf-8")
        assert RELATIVE_PACKET in document, relative
        assert PINNED_DIGEST in document, relative
        assert "UNREVIEWED" in document, relative
        assert AUTHORIZATION_URL in document, relative
    spec = (ROOT / "docs/specs/funding-carry-paper.md").read_text(encoding="utf-8")
    assert "owner-accepted deviation" in spec.lower().replace("**", "")
    assert "different authenticated GitHub principal" in " ".join(spec.split())
