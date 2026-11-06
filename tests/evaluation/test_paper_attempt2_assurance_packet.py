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
from trade_rl.evaluation.paper.attempt2 import (
    CAPACITY_HEADROOM,
    SCREEN_DAYS,
    SCREEN_SLOTS,
    TERMINAL_GRACE_SECONDS,
)
from trade_rl.evaluation.paper.operations import screen_plan
from trade_rl.evaluation.paper.predecessor import (
    ATTEMPT_ONE_FINAL_TIP_SHA256,
    ATTEMPT_ONE_LAST_OBSERVED_AT,
    ATTEMPT_ONE_PROTOCOL_SHA256,
    LINEAGE_RECORD_COMMIT,
    UNAVAILABLE_NOT_ESTABLISHED,
    UNAVAILABLE_REVIEW_RULE,
    USER_STATEMENT_DATE,
    USER_STATEMENT_QUOTE,
    canonical_lineage,
)

ROOT = Path(__file__).resolve().parents[2]
RELATIVE_PACKET = "report/issue810-carry-attempt2-research-assurance-20261105.json"
PACKET = ROOT / RELATIVE_PACKET
PINNED_DIGEST = "9a35fcd888f85166e2abb34000dd35580c8b94e216de0c1591be003c676df2d0"
AUTHORIZATION_URL = (
    "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-6659653178"
)
# The owner's verbatim words of 2026-11-05 (Issue #810 comment 6659653178). They say the
# reviewing account does not matter and that an agy review is acceptable; they do not
# name the same GitHub account.
OWNER_REVIEW_WORDS = (
    "別アカウントによるレビュー　別になんでもいいよ。agyのレビューでいいよ"
)
DOCUMENTS = (
    "docs/specs/funding-carry-paper.md",
    "docs/architecture/research-assurance.md",
    "docs/research/current-status.md",
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


def _text(packet: Any) -> str:
    return json.dumps(packet, ensure_ascii=False)


def _doc(relative: str) -> str:
    return " ".join((ROOT / relative).read_text(encoding="utf-8").split())


def _raw(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _added_section(relative: str) -> str:
    """The part of a document that records the attempt-2 packet."""
    document = _raw(relative)
    marker, end = {
        "docs/specs/funding-carry-paper.md": (
            "### Attempt 2 research-assurance author packet",
            None,
        ),
        "docs/architecture/research-assurance.md": (
            "#### Predecessor root unavailable: G0 exception pending owner authorization",
            "**Known limitations:**",
        ),
        "docs/research/current-status.md": (
            "2026-11-05: a Research Assurance author packet",
            "\n\n",
        ),
    }[relative]
    start = document.index(marker)
    stop = document.index(end, start) if end else len(document)
    return " ".join(document[start:stop].split())


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
    assert claims["next_action_class"] == "request_separate_economic_authorization"
    assert claims["claim_level"] == "software_validity"
    assert packet["evidence"]["evidence_level"] == "software_validity"
    assert thesis["stage"] == "prospective_paper"
    assert "Issue #810" in thesis["stop_rule"]
    assert AUTHORIZATION_URL in thesis["stop_rule"]
    assert AUTHORIZATION_URL in claims["decision_rule"]
    assert "different authenticated GitHub principal" in thesis["stop_rule"]
    assert "PASS" not in claims["next_authorized_action"]
    # Every boundary is covered by an oracle that cites existing test paths.
    mechanism = packet["mechanism"]
    assert set(mechanism["changed_boundaries"]) == BOUNDARIES
    assert set(mechanism["independent_oracles"]) == BOUNDARIES
    cited = set(re.findall(r"tests/[\w/]+\.py", _text(packet)))
    assert cited
    missing = sorted(path for path in cited if not (ROOT / path).is_file())
    assert missing == []


def test_packet_records_no_development_result_label() -> None:
    """A result-blind packet may not carry a development economic outcome label."""
    packet = _packet()
    text = _text(packet)
    hypothesis = packet["thesis"]["hypothesis"]

    forbidden = (
        re.compile(
            r"\bpass(?:ed|es)\b[^.;]{0,60}\b(?:screens?|criteria|preregistered)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bpast development profit", re.IGNORECASE),
        re.compile(
            r"\bdevelopment (?:replay|run|result)s?[^.;]{0,40}\b(?:profit|passed|positive)",
            re.IGNORECASE,
        ),
    )
    for pattern in forbidden:
        assert pattern.search(text) is None, pattern.pattern
        for relative in DOCUMENTS:
            assert pattern.search(_added_section(relative)) is None, (
                relative,
                pattern.pattern,
            )
    assert "reused-development evidence only" in hypothesis
    assert "no development economic outcome is recorded" in hypothesis


def test_review_route_is_a_same_principal_deviation_not_accepted_and_not_a_pass() -> (
    None
):
    packet = _packet()
    limitations = packet["thesis"]["limitations"]
    permitted = " ".join(packet["claims"]["permitted"])
    next_action = packet["claims"]["next_authorized_action"]

    assert OWNER_REVIEW_WORDS in limitations
    assert "do not name the same GitHub account" in limitations
    assert "planner's reading" in limitations
    assert "same-principal deviation" in limitations
    assert "beyond those quoted words" in limitations
    for field in (limitations, permitted, next_action):
        assert "owner-recorded" not in field
    for field in (limitations, permitted, next_action):
        assert "owner-accepted" not in field
        assert "explicitly accepted" not in field
    assert (
        "tools.research_assurance_github rejects a same-principal review" in limitations
    )
    assert "makes no PASS claim" in limitations
    for relative in DOCUMENTS:
        document = _doc(relative)
        assert "owner-accepted" not in document, relative
        assert "explicitly accepted" not in document, relative
        assert "owner-recorded" not in document, relative
        assert OWNER_REVIEW_WORDS in _raw(relative).replace("\n", " "), relative
        assert "planner's reading" in document, relative
        assert "tools.research_assurance_github" in document, relative
        assert "rejects" in document, relative


def test_predecessor_unavailable_case_is_not_a_waiver_and_needs_owner_authorization() -> (
    None
):
    packet = _packet()
    limitations = packet["thesis"]["limitations"]
    stop_rule = packet["thesis"]["stop_rule"]
    claims = packet["claims"]
    lineage = canonical_lineage("unavailable")
    contract = _doc("docs/architecture/research-assurance.md")
    heading = "Predecessor root unavailable: G0 exception pending owner authorization"

    assert heading in contract
    section = contract[
        contract.index(heading) : contract.index("The attempt-2 author packet")
    ]
    # The general rule stands unchanged: no exception is described as authorized,
    # and the section never calls itself an existing bounded exception or a waiver.
    assert (
        "G0 review must verify the complete predecessor chain before a successor is "
        "sealed. No G0 exception has been authorized"
    ) in contract
    assert "one bounded exception" not in contract
    assert "bounded exception" not in contract.lower()
    assert "is **not waived**" in section
    assert "may not be sealed with `--predecessor-root-status unavailable`" in section
    assert "the planner's text, not the owner's words" in section
    assert "does not say that the root is lost or unreadable" in section
    assert "attestation alone" in contract
    for required in (
        "a separate explicit owner authorization exists on Issue #810 that names the G0",
        "predecessor-verification exception",
        "The comment of 2026-11-05 does not say this and is not that authorization",
        "does not apply to any other family",
        "relaxes no screen criterion",
        "invalidated",
        "observation_gap",
        "re_verified_at_seal",
        "no attempt-1 economic or status readout was used",
        "global attempt uniqueness stays unestablished",
        "does **not** attest the lineage values",
        "**Does not replace:**",
        AUTHORIZATION_URL.rsplit("-", 1)[-1],
        *UNAVAILABLE_NOT_ESTABLISHED,
    ):
        assert required in section, required
    # The owner's verbatim words are quoted, not paraphrased into a stronger claim.
    assert USER_STATEMENT_QUOTE.replace("\n", "") in section
    # The implementer search record states its scope, its result and what it omits.
    for required in (
        "Implementer search record (read-only, 2026-11-06)",
        ATTEMPT_ONE_PROTOCOL_SHA256,
        "627",
        "none matched",
        "Not searched:",
        "does not prove that the root does not exist elsewhere",
    ):
        assert required in section, required
    # The packet states the same position and never claims the authorization exists.
    assert USER_STATEMENT_QUOTE.replace("\n", "") in limitations
    assert "not waived" in limitations
    assert "do not mention G0" in limitations
    assert "requires a separate explicit owner authorization" in limitations
    assert (
        "does not exist as of 2026-11-06 and this packet does not claim it"
        in limitations
    )
    assert "a seal with unavailable is not permitted until it does" in limitations
    assert "No screen criterion is relaxed" in limitations
    assert "does not apply generally" not in limitations
    assert "replaced by a fresh result-blind review" not in limitations
    assert "no authorization of the (c) kind exists as of 2026-11-06" in stop_rule
    assert any(
        "predecessor-verification exception" in item for item in claims["forbidden"]
    )
    assert "predecessor-verification exception" in claims["decision_rule"]
    assert "predecessor-verification exception" in claims["next_authorized_action"]
    # The sealed record, if it is ever used, carries what the contract says.
    record = lineage["predecessor_root"]
    assert record["status"] == "unavailable"
    assert record["re_verified_at_seal"] is False
    assert record["not_established"] == list(UNAVAILABLE_NOT_ESTABLISHED)
    assert record["user_statement"]["quote"] == USER_STATEMENT_QUOTE
    assert record["user_statement"]["date"] == USER_STATEMENT_DATE
    assert record["review_rule"] == UNAVAILABLE_REVIEW_RULE
    assert USER_STATEMENT_DATE in limitations
    assert lineage["predecessor"]["disposition"] == "invalidated"
    assert lineage["predecessor"]["reason_codes"] == ["observation_gap"]


def test_packet_restates_the_frozen_screen_and_lineage_without_changing_them() -> None:
    lineage = canonical_lineage("unavailable")
    plan = screen_plan(attempt_lineage=lineage)
    packet = _packet()
    text = _text(packet)
    rule = packet["claims"]["decision_rule"]
    evidence_oracle = packet["mechanism"]["independent_oracles"]["evidence"][
        "statement"
    ]

    assert plan["schema"] == "carry_paper_screen_v2"
    assert "carry_paper_screen_v2" in text
    assert plan["duration_days"] == SCREEN_DAYS == 90 and "ninety-day" in text
    assert plan["block_days"] == 30 and plan["block_count"] == 3
    assert "three thirty-day blocks" in rule
    assert plan["maximum_drawdown_exclusive"] == 0.1 and "below 10%" in rule
    assert plan["comparison"] == "cash_zero_interest"
    assert packet["evidence"]["controls"] == [plan["comparison"]]
    assert plan["production_eligible"] is False
    assert plan["minimum_seal_notice_seconds"] == 300 and "five minutes" in text
    assert TERMINAL_GRACE_SECONDS == 180 and "close plus 180 seconds" in rule
    assert SCREEN_SLOTS == 129604 and f"{SCREEN_SLOTS:,}-slot" in evidence_oracle
    assert CAPACITY_HEADROOM == 1.25 and "1.25 headroom" in evidence_oracle
    fills = {(row["symbol"], row["venue"]) for row in plan["required_fill_coverage"]}
    assert fills == {
        ("BTCUSDT", "spot"),
        ("BTCUSDT", "perpetual"),
        ("ETHUSDT", "spot"),
        ("ETHUSDT", "perpetual"),
    }
    assert (
        "one fill in each of BTC spot, BTC perpetual, ETH spot and ETH perpetual"
        in evidence_oracle
    )
    for flag in (
        "positive_net_profit",
        "positive_each_block",
        "net_profit_exceeds_recorded_fees",
        "nonzero_funding_each_block",
        "complete_coverage_and_actual_flat",
    ):
        assert plan[flag] is True, flag

    assert lineage["schema"] == "carry_paper_attempt_lineage_v3"
    assert lineage["attempt_number"] == 2
    assert lineage["predecessor_root"]["status"] == "unavailable"
    assert lineage["predecessor_root"]["re_verified_at_seal"] is False
    for value in (
        ATTEMPT_ONE_PROTOCOL_SHA256,
        ATTEMPT_ONE_FINAL_TIP_SHA256,
        ATTEMPT_ONE_LAST_OBSERVED_AT,
        LINEAGE_RECORD_COMMIT,
        *UNAVAILABLE_NOT_ESTABLISHED,
    ):
        assert value in packet["thesis"]["limitations"], value
    assert lineage["predecessor"] == {
        "attempt_number": 1,
        "protocol_sha256": ATTEMPT_ONE_PROTOCOL_SHA256,
        "final_tip_sha256": ATTEMPT_ONE_FINAL_TIP_SHA256,
        "disposition": "invalidated",
        "reason_codes": ["observation_gap"],
        "last_observed_at": ATTEMPT_ONE_LAST_OBSERVED_AT,
    }


def test_evidence_booleans_are_bound_to_what_the_source_establishes() -> None:
    """Booleans say what is established now; a paper design is not prospective evidence."""
    packet = _packet()
    evidence = packet["evidence"]
    plan = screen_plan(attempt_lineage=canonical_lineage("unavailable"))
    limitations = packet["thesis"]["limitations"]

    expected = {
        # Enforced in software and tests by the frozen plan and canonical ledger.
        "common_accounting": True,
        "hard_risk": plan["maximum_drawdown_exclusive"] == 0.1,
        "terminal_state": plan["complete_coverage_and_actual_flat"] is True,
        "multi_symbol": len({row["symbol"] for row in plan["required_fill_coverage"]})
        > 1,
        "no_development_rescue": True,
        # Need prospective observation or are paper assumptions: not established.
        "point_in_time": False,
        "realistic_costs": False,
        "fit_development_unused_separated": False,
        "multi_period": False,
    }
    assert {name: evidence[name] for name in expected} == expected
    oracles = packet["mechanism"]["independent_oracles"]
    assert (
        "paper assumptions, not measured venue costs"
        in (oracles["execution"]["known_limitations"])
    )
    assert (
        "Spot REST depth has no exchange timestamp"
        in (oracles["time"]["known_limitations"])
    )
    assert "Paper trading establishes no queue position, latency" in limitations
    assert (
        "No prospective observation has been collected"
        in (evidence["independent_reconstruction"])
    )
    assert evidence["evidence_level"] == "software_validity"


def test_docs_point_to_the_packet_digest_and_unreviewed_status() -> None:
    for relative in DOCUMENTS:
        document = (ROOT / relative).read_text(encoding="utf-8")
        assert RELATIVE_PACKET in document, relative
        assert PINNED_DIGEST in document, relative
        assert "UNREVIEWED" in document, relative
        assert AUTHORIZATION_URL in document, relative
    spec = _doc("docs/specs/funding-carry-paper.md").replace("**", "")
    assert "not attributed to the owner beyond those quoted words" in spec
    assert "different authenticated GitHub principal" in spec
