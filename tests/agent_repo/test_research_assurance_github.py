from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.agent_repo.research_assurance import assurance_digest
from tools.research_assurance_github import verify_github_review

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "tools/agent_repo/examples/research_assurance.json"
)


def samples() -> tuple[dict, dict, dict, dict]:
    author = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    review = {
        "schema_version": "research_assurance_review_v1",
        "reviewed_record_digest": assurance_digest(author),
        "protocol_head": author["identity"]["protocol_head"],
        "implementation_head": author["identity"]["implementation_head"],
        "reviewer_identity": "external-reviewer",
        "reviewer_execution_identity": "separate-read-only-review-session",
        "reviewer_independence": "ESTABLISHED",
        "result_blind": True,
        "g0": "PASS",
        "g1": "PASS",
        "g2": "PASS",
        "status": "pass",
        "disposition": "ASSURANCE_PASS",
        "blocking_findings": [],
        "missing_evidence": [],
        "rationale": "Independent causal counterexample and test oracle were inspected.",
        "limitations": [
            "Economic outputs, prospective profitability and live trading unverified."
        ],
    }
    review_body = json.dumps(
        review, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    pr = {
        "number": 432,
        "state": "open",
        "base": {"repo": {"full_name": "shuntatsu/trade_rl"}},
        "head": {
            "sha": author["identity"]["implementation_head"],
            "repo": {"full_name": "shuntatsu/trade_rl"},
        },
        "user": {"id": 10, "login": "author"},
    }
    gh_review = {
        "id": 9815,
        "state": "COMMENTED",
        "commit_id": author["identity"]["implementation_head"],
        "user": {"id": 55, "login": "external-reviewer"},
        "submitted_at": "2026-11-02T00:00:00Z",
        "body": review_body,
    }
    return author, review, pr, gh_review


def run_readback(author: dict, pr: dict, review: dict):
    calls = []

    def read(path: str):
        calls.append(path)
        if path == "repos/shuntatsu/trade_rl/pulls/432":
            return copy.deepcopy(pr)
        if path == "repos/shuntatsu/trade_rl/pulls/432/reviews/9815":
            return copy.deepcopy(review)
        raise AssertionError(path)

    return verify_github_review(
        author,
        repository="shuntatsu/trade_rl",
        pr_number=432,
        review_id=9815,
        api_read=read,
    ), calls


def test_pr_review_is_authenticated_at_exact_head_with_distinct_principal() -> None:
    author, _, pr, review = samples()
    result, calls = run_readback(author, pr, review)
    assert result.status == "PASS"
    assert result.economic_execution_authorized is False
    assert result.errors == ()
    assert calls.count("repos/shuntatsu/trade_rl/pulls/432") == 2
    assert calls.count("repos/shuntatsu/trade_rl/pulls/432/reviews/9815") == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "review_head",
        "pr_head",
        "author_principal",
        "not_submitted",
        "dismissed",
        "wrong_repository",
        "changed_number",
        "edited_status",
        "noncanonical_body",
    ],
)
def test_authenticated_review_mismatches_fail_closed(mutation: str) -> None:
    author, _, pr, review = samples()
    if mutation == "review_head":
        review["commit_id"] = "c" * 40
    elif mutation == "pr_head":
        pr["head"]["sha"] = "c" * 40
    elif mutation == "author_principal":
        review["user"]["id"] = pr["user"]["id"]
    elif mutation == "not_submitted":
        review["submitted_at"] = None
    elif mutation == "dismissed":
        review["state"] = "DISMISSED"
    elif mutation == "wrong_repository":
        pr["head"]["repo"]["full_name"] = "attacker/fork"
    elif mutation == "changed_number":
        pr["number"] = 999
    elif mutation == "edited_status":
        review["body"] = review["body"].replace('"g0":"PASS"', '"g0":"FAIL"')
    else:
        review["body"] = "  " + review["body"] + "  "
    result, _ = run_readback(author, pr, review)
    assert result.status == "BLOCKED"
    assert result.economic_execution_authorized is False


def test_second_pr_readback_detects_head_change_race() -> None:
    author, _, pr, review = samples()
    reads = {"count": 0}

    def fake(path: str):
        if path.endswith("/reviews/9815"):
            return review
        reads["count"] += 1
        copy_pr = copy.deepcopy(pr)
        if reads["count"] == 2:
            copy_pr["head"]["sha"] = "c" * 40
        return copy_pr

    result = verify_github_review(
        author,
        repository="shuntatsu/trade_rl",
        pr_number=432,
        review_id=9815,
        api_read=fake,
    )
    assert result.status == "BLOCKED"


def test_untrusted_pr_review_body_cannot_force_approval() -> None:
    author, _, pr, review = samples()
    review["body"] = review["body"].replace(
        '"result_blind":true', '"result_blind":false'
    )
    result, _ = run_readback(author, pr, review)
    assert result.status == "BLOCKED"


def test_invalid_repository_path_rejected_before_network_call() -> None:
    author, _, _, _ = samples()

    def no_network(_path: str):
        raise AssertionError("should not fetch")

    with pytest.raises(ValueError):
        verify_github_review(
            author,
            repository="../other",
            pr_number=432,
            review_id=9815,
            api_read=no_network,
        )


def test_review_body_must_be_canonical_exact_bytes() -> None:
    author, _, pr, review = samples()
    body = json.loads(review["body"])
    review["body"] = json.dumps(body, ensure_ascii=False, indent=2)
    result, _ = run_readback(author, pr, review)
    assert result.status == "BLOCKED"


def test_reviewer_identity_is_not_inferred_from_author_claims() -> None:
    author, _, pr, review = samples()
    body = json.loads(review["body"])
    body["reviewer_identity"] = "another-identity"
    review["body"] = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    result, _ = run_readback(author, pr, review)
    assert result.status == "BLOCKED"
