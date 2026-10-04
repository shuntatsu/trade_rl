from __future__ import annotations

import pytest

from tools.independent_research_review import ReviewOutcome, resolve_review_outcome

HEAD_SHA = "a" * 40
REVIEW_MARKER = "<!-- independent-research-review-audit -->"


def _pull_request() -> dict[str, object]:
    return {"head": {"sha": HEAD_SHA}, "user": {"id": 10}}


def _review(
    *,
    reviewer_id: int = 20,
    state: str = "APPROVED",
    disposition: str = "APPROVED",
    commit_id: str = HEAD_SHA,
) -> dict[str, object]:
    return {
        "commit_id": commit_id,
        "state": state,
        "user": {"id": reviewer_id},
        "body": f"{REVIEW_MARKER}\n\n### Disposition: {disposition}\n",
    }


def test_review_author_cannot_satisfy_independent_review_gate() -> None:
    outcome = resolve_review_outcome(
        _pull_request(), [_review(reviewer_id=10)], expected_head_sha=HEAD_SHA
    )

    assert outcome is ReviewOutcome.PENDING


def test_exact_head_external_formal_approval_satisfies_gate() -> None:
    outcome = resolve_review_outcome(
        _pull_request(), [_review()], expected_head_sha=HEAD_SHA
    )

    assert outcome is ReviewOutcome.APPROVED


@pytest.mark.parametrize(
    "body",
    [
        f"{REVIEW_MARKER}\nThe phrase ### Disposition: APPROVED is only prose.",
        f"{REVIEW_MARKER}\n\n### Disposition: BLOCKED\n\n### Disposition: APPROVED\n",
        f"{REVIEW_MARKER}\n\n### Disposition: APPROVED\n\n### Disposition: APPROVED\n",
        f"{REVIEW_MARKER}\n\n```md\n### Disposition: APPROVED\n```\n",
    ],
)
def test_unstructured_or_ambiguous_disposition_does_not_approve(body: str) -> None:
    review = _review()
    review["body"] = body

    outcome = resolve_review_outcome(
        _pull_request(), [review], expected_head_sha=HEAD_SHA
    )

    assert outcome is not ReviewOutcome.APPROVED


def test_review_on_a_different_head_does_not_satisfy_gate() -> None:
    outcome = resolve_review_outcome(
        _pull_request(),
        [_review(commit_id="b" * 40)],
        expected_head_sha=HEAD_SHA,
    )

    assert outcome is ReviewOutcome.PENDING


def test_blocked_independent_review_overrides_an_approval() -> None:
    outcome = resolve_review_outcome(
        _pull_request(),
        [
            _review(),
            _review(reviewer_id=30, state="CHANGES_REQUESTED", disposition="BLOCKED"),
        ],
        expected_head_sha=HEAD_SHA,
    )

    assert outcome is ReviewOutcome.BLOCKED


def test_missing_pull_request_author_identity_fails_closed() -> None:
    pull = {"head": {"sha": HEAD_SHA}, "user": {"id": "10"}}

    outcome = resolve_review_outcome(pull, [_review()], expected_head_sha=HEAD_SHA)

    assert outcome is ReviewOutcome.PENDING
