from __future__ import annotations

from typing import Any

import pytest

from tools import ppo_4h_gemini_review as review

REVIEWED_SHA = "a" * 40
OTHER_SHA = "b" * 40


def _artifact(
    *,
    identity: str,
    run_id: int,
    attempt: int,
    expired: bool = False,
) -> dict[str, object]:
    return {
        "name": review.review_packet_artifact_name(
            identity,
            run_id=run_id,
            run_attempt=attempt,
        ),
        "expired": expired,
        "workflow_run": {"id": run_id},
    }


def _jobs(*, provider_conclusion: str | None) -> list[object]:
    steps: list[dict[str, object]] = []
    if provider_conclusion is not None:
        steps.append(
            {
                "name": review.PROVIDER_STEP_NAME,
                "status": "completed",
                "conclusion": provider_conclusion,
            }
        )
    return [
        {
            "name": review.REVIEW_JOB_NAME,
            "status": "completed",
            "conclusion": "failure",
            "steps": steps,
        }
    ]


def test_review_identity_is_sha_and_protocol_bound_but_tag_independent() -> None:
    first = review.review_identity_digest(
        repository="owner/repo",
        repository_id=99,
        pull_number=900,
        reviewed_code_sha=REVIEWED_SHA,
    )
    second = review.review_identity_digest(
        repository="owner/repo",
        repository_id=99,
        pull_number=900,
        reviewed_code_sha=REVIEWED_SHA,
    )
    changed = review.review_identity_digest(
        repository="owner/repo",
        repository_id=99,
        pull_number=900,
        reviewed_code_sha=OTHER_SHA,
    )

    assert len(first) == 64
    assert first == second
    assert first != changed
    assert review.REVIEW_PROTOCOL == "ppo_4h_gemini_semantic_review_v1"


def test_different_run_for_same_identity_is_not_a_retry() -> None:
    identity = review.review_identity_digest(
        repository="owner/repo",
        repository_id=99,
        pull_number=900,
        reviewed_code_sha=REVIEWED_SHA,
    )
    artifacts = [_artifact(identity=identity, run_id=100, attempt=1)]

    with pytest.raises(ValueError, match="review identity already has an attempt"):
        review.require_review_identity_retryable(
            identity,
            current_run_id=200,
            current_run_attempt=1,
            artifacts=artifacts,
            jobs_for_attempt=lambda _run, _attempt: _jobs(
                provider_conclusion="failure"
            ),
        )


def test_same_run_retry_is_allowed_only_before_terminal_provider_response() -> None:
    identity = review.review_identity_digest(
        repository="owner/repo",
        repository_id=99,
        pull_number=900,
        reviewed_code_sha=REVIEWED_SHA,
    )
    artifacts = [_artifact(identity=identity, run_id=100, attempt=1)]

    review.require_review_identity_retryable(
        identity,
        current_run_id=100,
        current_run_attempt=2,
        artifacts=artifacts,
        jobs_for_attempt=lambda _run, _attempt: _jobs(provider_conclusion="failure"),
    )

    with pytest.raises(ValueError, match="terminal Gemini review already exists"):
        review.require_review_identity_retryable(
            identity,
            current_run_id=100,
            current_run_attempt=2,
            artifacts=artifacts,
            jobs_for_attempt=lambda _run, _attempt: _jobs(
                provider_conclusion="success"
            ),
        )


def test_expired_or_unrelated_artifacts_do_not_create_false_identity() -> None:
    identity = review.review_identity_digest(
        repository="owner/repo",
        repository_id=99,
        pull_number=900,
        reviewed_code_sha=REVIEWED_SHA,
    )
    other = "c" * 64
    artifacts: list[object] = [
        _artifact(identity=identity, run_id=100, attempt=1, expired=True),
        _artifact(identity=other, run_id=101, attempt=1),
        {"name": "unrelated", "expired": False, "workflow_run": {"id": 102}},
    ]
    calls: list[tuple[int, int]] = []

    def jobs_for_attempt(run_id: int, attempt: int) -> list[object]:
        calls.append((run_id, attempt))
        return _jobs(provider_conclusion="success")

    review.require_review_identity_retryable(
        identity,
        current_run_id=200,
        current_run_attempt=1,
        artifacts=artifacts,
        jobs_for_attempt=jobs_for_attempt,
    )

    assert calls == []


def test_attestation_binds_review_identity_and_protocol() -> None:
    identity = review.review_identity_digest(
        repository="owner/repo",
        repository_id=99,
        pull_number=900,
        reviewed_code_sha=REVIEWED_SHA,
    )
    parsed: dict[str, Any] = {
        "reviewer_provider": "google_gemini",
        "reviewer_model": "gemini-model-version",
        "reviewer_response_id": "response-123",
        "g0": "PASS",
        "g1": "PASS",
        "g2": "EVIDENCE_BOUND",
        "disposition": "G0_G1_CLEAR_G2_EVIDENCE_BOUND",
        "blocking_findings": [],
        "strongest_counterexample": "counterexample",
        "missing_evidence": [],
        "claim_downgrade": "development only",
        "machine_oracles": ["oracle"],
        "what_this_cannot_prove": ["profitability"],
    }

    record = review.build_attestation(
        repository="owner/repo",
        repository_id=99,
        reviewed_code_sha=REVIEWED_SHA,
        review_tag="review/ppo-4h-indicator-smoke-v1",
        review_tag_object_sha="d" * 40,
        trusted_workflow_sha="e" * 40,
        trusted_workflow_ref=(
            "owner/repo/.github/workflows/ppo-4h-gemini-review.yml@refs/heads/main"
        ),
        trusted_workflow_file_sha256="1" * 64,
        trusted_runner_sha256="2" * 64,
        reviewer_run_id=456,
        reviewer_run_attempt=1,
        request_pull_number=900,
        request_comment_id=789,
        requester_login="author",
        requester_permission="write",
        request_body_sha256="3" * 64,
        trusted_verification_jobs={
            "core": 11,
            "ppo-runtime": 12,
            "guide": 13,
        },
        packet_sha256="4" * 64,
        gemini_request_sha256="5" * 64,
        raw_gemini_response_sha256="6" * 64,
        parsed_review=parsed,
    )

    assert record["review_identity_digest"] == identity
    assert record["review_protocol"] == review.REVIEW_PROTOCOL
