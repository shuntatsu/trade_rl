from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.agent_repo.research_assurance import (
    TrustedReviewEvidence,
    assurance_digest,
    evaluate_assurance,
)


def author_record() -> dict:
    return json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "tools/agent_repo/examples/research_assurance.json"
        ).read_text(encoding="utf-8")
    )


def review_for(record: dict, **overrides: object) -> dict:
    review = {
        "schema_version": "research_assurance_review_v1",
        "reviewed_record_digest": assurance_digest(record),
        "protocol_head": record["identity"]["protocol_head"],
        "implementation_head": record["identity"]["implementation_head"],
        "reviewer_identity": "independent-reviewer",
        "reviewer_execution_identity": "trusted-run-1",
        "reviewer_independence": "ESTABLISHED",
        "result_blind": True,
        "g0": "PASS",
        "g1": "PASS",
        "g2": "PASS",
        "status": "pass",
        "disposition": "ASSURANCE_PASS",
        "blocking_findings": [],
        "missing_evidence": [],
        "rationale": "The causal question and mechanism have an independent control and counterexample.",
        "limitations": [
            "No real economic result, unused-data study or live authorization."
        ],
    }
    review.update(overrides)
    return review


def trusted_for(
    record: dict, review: dict, **overrides: object
) -> TrustedReviewEvidence:
    attributes = {
        "review_body_sha256": hashlib.sha256(
            json.dumps(
                review,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "reviewer_identity": "independent-reviewer",
        "author_identity": "research-author",
        "reviewed_commit_sha": record["identity"]["implementation_head"],
        "reviewer_run_identity": "trusted-run-1",
        "source_kind": "authenticated_pr_review",
        "authenticated": True,
    }
    attributes.update(overrides)
    return TrustedReviewEvidence(**attributes)


def evaluate_review(record: dict, **overrides: object):
    review = review_for(record, **overrides)
    return evaluate_assurance(
        record, review=review, authority=trusted_for(record, review)
    )


def test_complete_unreviewed_never_authorizes_economic_execution() -> None:
    result = evaluate_assurance(author_record())
    assert result.status == "UNREVIEWED"
    assert result.economic_execution_authorized is False
    assert result.errors == ()


def test_reviewed_bounded_claim_can_pass_assurance_but_never_authorize_economics() -> (
    None
):
    result = evaluate_review(author_record())
    assert result.status == "PASS"
    assert result.economic_execution_authorized is False
    assert result.errors == ()


def test_review_without_external_authenticated_authority_is_blocked() -> None:
    record = author_record()
    review = review_for(record)
    result = evaluate_assurance(record, review=review)
    assert result.status == "BLOCKED"
    assert any("authority" in error for error in result.errors)
    assert result.economic_execution_authorized is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reviewer_independence", "NOT_ESTABLISHED"),
        ("result_blind", False),
        ("g0", "FAIL"),
        ("g1", "NOT_ESTABLISHED"),
        ("g2", "FAIL"),
        ("disposition", "READY_FOR_MACHINE_VERIFICATION"),
        ("blocking_findings", ["the chosen metric answers a proxy question"]),
        ("missing_evidence", ["independent clock oracle"]),
    ],
)
def test_review_body_semantics_are_structural_fail_closed(
    field: str, value: object
) -> None:
    result = evaluate_review(author_record(), **{field: value})
    assert result.status == "BLOCKED"
    assert result.economic_execution_authorized is False


def test_self_authored_review_even_with_all_pass_fields_is_blocked() -> None:
    record = author_record()
    review = review_for(record)
    witness = trusted_for(record, review, reviewer_identity="research-author")
    result = evaluate_assurance(record, review=review, authority=witness)
    assert result.status == "BLOCKED"
    assert any("independence" in error or "author" in error for error in result.errors)


def test_wrong_question_review_blocks_even_when_software_tests_green() -> None:
    record = author_record()
    review = review_for(
        record,
        status="blocked",
        disposition="BLOCKED",
        g0="FAIL",
        blocking_findings=[
            "Wrong question: constant exposure beta explains positive net return."
        ],
    )
    result = evaluate_assurance(
        record, review=review, authority=trusted_for(record, review)
    )
    assert result.status == "BLOCKED"
    assert result.economic_execution_authorized is False


def test_review_becomes_stale_after_changes_to_thesis_or_head() -> None:
    for kind in ("hypothesis", "head"):
        record = author_record()
        review = review_for(record)
        witness = trusted_for(record, review)
        changed = copy.deepcopy(record)
        if kind == "hypothesis":
            changed["thesis"]["hypothesis"] = "A different question"
        else:
            changed["identity"]["implementation_head"] = "c" * 40
        result = evaluate_assurance(changed, review=review, authority=witness)
        assert result.status == "BLOCKED"
        assert result.economic_execution_authorized is False
        assert result.errors


def test_provenance_mismatch_and_replay_are_blocked() -> None:
    record = author_record()
    review = review_for(record)
    for override in (
        {"review_body_sha256": "e" * 64},
        {"reviewed_commit_sha": "c" * 40},
        {"reviewer_run_identity": "another-run"},
        {"authenticated": False},
        {"source_kind": "self_asserted"},
    ):
        result = evaluate_assurance(
            record, review=review, authority=trusted_for(record, review, **override)
        )
        assert result.status == "BLOCKED"
        assert result.errors


@pytest.mark.parametrize(
    "field", ["counter_hypothesis", "cheapest_falsifier", "stop_rule", "limitations"]
)
def test_required_question_falsifier_stop_and_limitations_fail_closed(
    field: str,
) -> None:
    record = author_record()
    record["thesis"][field] = ""
    assert evaluate_assurance(record).status == "BLOCKED"


@pytest.mark.parametrize(
    ("evidence_level", "claim_level"),
    [
        ("software_validity", "mechanism_validity"),
        ("software_validity", "relative_improvement"),
        ("relative_improvement", "development_profitability"),
        ("development_profitability", "unused_validation"),
    ],
)
def test_evidence_cannot_be_promoted_to_stronger_claim(
    evidence_level: str, claim_level: str
) -> None:
    record = author_record()
    record["evidence"]["evidence_level"] = evidence_level
    record["claims"]["claim_level"] = claim_level
    result = evaluate_assurance(record)
    assert result.status == "BLOCKED"
    assert any("claim_level" in value for value in result.errors)


@pytest.mark.parametrize("field", ["production_eligible", "live_trading_authorized"])
def test_development_stage_cannot_self_authorize_production(field: str) -> None:
    record = author_record()
    record["claims"][field] = True
    assert evaluate_assurance(record).status == "BLOCKED"


def test_author_packet_cannot_contain_self_authored_review() -> None:
    record = author_record()
    record["review"] = review_for(record)
    result = evaluate_assurance(record)
    assert result.status == "BLOCKED"


def test_review_status_unknown_and_incomplete_are_blocked() -> None:
    record = author_record()
    assert (
        evaluate_review(record, status="pass", g2="EVIDENCE_BOUND").status == "BLOCKED"
    )
    assert (
        evaluate_review(record, status="pass", reviewer_execution_identity="").status
        == "BLOCKED"
    )
    assert evaluate_review(record, status="pass", rationale="").status == "BLOCKED"


def test_malformed_and_noncanonical_documents_fail_closed() -> None:
    for broken in ({}, {"schema_version": "wrong"}, {"a": float("nan")}):
        assert evaluate_assurance(broken).status == "BLOCKED"
    record = author_record()
    record["mechanism"]["chain"] = ["source", "decision"]
    assert evaluate_assurance(record).status == "BLOCKED"
    record = author_record()
    record["mechanism"]["independent_oracles"] = []
    assert evaluate_assurance(record).status == "BLOCKED"


def test_reviewer_cannot_downgrade_claim_scope_silently() -> None:
    record = author_record()
    review = review_for(record)
    review["limitations"] = []
    assert (
        evaluate_assurance(
            record, review=review, authority=trusted_for(record, review)
        ).status
        == "BLOCKED"
    )


def test_packet_digest_is_order_independent_and_review_separate() -> None:
    source = author_record()
    reordered = dict(reversed(list(source.items())))
    assert assurance_digest(source) == assurance_digest(reordered)
    assert len(assurance_digest(source)) == 64


def test_every_changed_economic_boundary_requires_independent_oracle() -> None:
    record = author_record()
    mechanism = record["mechanism"]
    changed = mechanism["changed_boundaries"]
    assert changed
    oracles = mechanism["independent_oracles"]
    for boundary in changed:
        assert boundary in oracles
        missing = copy.deepcopy(record)
        del missing["mechanism"]["independent_oracles"][boundary]
        result = evaluate_assurance(missing)
        assert result.status == "BLOCKED"
        assert any("oracle" in error for error in result.errors)


def test_duplicate_or_undeclared_mechanism_boundaries_are_blocked() -> None:
    record = author_record()
    record["mechanism"]["changed_boundaries"].append(
        record["mechanism"]["changed_boundaries"][0]
    )
    assert evaluate_assurance(record).status == "BLOCKED"

    record = author_record()
    record["mechanism"]["changed_boundaries"] = ["not_a_contract"]
    assert evaluate_assurance(record).status == "BLOCKED"


def test_oracle_requires_counterexample_scope_and_limitations() -> None:
    record = author_record()
    first_boundary = record["mechanism"]["changed_boundaries"][0]
    for key in ("statement", "counterexample", "oracle", "known_limitations"):
        broken = copy.deepcopy(record)
        broken["mechanism"]["independent_oracles"][first_boundary][key] = ""
        assert evaluate_assurance(broken).status == "BLOCKED"


@pytest.mark.parametrize("field", ["decision_rule", "next_action_class"])
def test_reviewed_claim_requires_explicit_machine_checked_decision_rule(
    field: str,
) -> None:
    record = author_record()
    record["claims"].pop(field, None)
    result = evaluate_assurance(record)
    assert result.status == "BLOCKED"
    assert any(field in error for error in result.errors)


def test_action_class_cannot_self_grant_economic_or_live_execution() -> None:
    record = author_record()
    record["claims"]["next_action_class"] = "live_trading"
    assert evaluate_assurance(record).status == "BLOCKED"


@pytest.mark.parametrize("flag", ["production_eligible", "live_trading_authorized"])
def test_even_production_stage_record_cannot_authorize_live_or_production(
    flag: str,
) -> None:
    record = author_record()
    record["thesis"]["stage"] = "production"
    record["evidence"]["evidence_level"] = "production"
    record["claims"]["claim_level"] = "production"
    record["claims"][flag] = True
    assert evaluate_assurance(record).status == "BLOCKED"
