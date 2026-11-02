from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.independent_research_review import (
    ReviewOutcome,
    resolve_review_outcome,
)

HEAD_SHA = "a" * 40
REVIEW_MARKER = "<!-- independent-research-review-audit -->"
RECEIPT_FENCE = "```independent-review-receipt"


def _payload(*, publisher_id: int = 20) -> dict[str, object]:
    return {
        "schema": "independent_model_review_v1",
        "head_sha": HEAD_SHA,
        "publisher_user_id": publisher_id,
        "scope": "software-only",
        "disposition": "APPROVED",
        "implementation": [
            {"model_id": "gpt-6.1-sol", "task_id": "/root/implementation"}
        ],
        "reviewer": {"model_id": "gpt-6-astra", "task_id": "/root/review"},
        "report": "Reviewed source and contract tests. No blocking findings. Economic authorization remains separate.",
    }


def _receipt_body(payload: dict[str, object], *, digest: str | None = None) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    envelope = {
        "payload": payload,
        "sha256": digest or hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }
    return f"{RECEIPT_FENCE}\n{json.dumps(envelope)}\n```\n"


def _pull_request() -> dict[str, object]:
    return {"head": {"sha": HEAD_SHA}, "user": {"id": 10}}


def _review(
    *,
    reviewer_id: int = 20,
    state: str = "APPROVED",
    disposition: str = "APPROVED",
    commit_id: str = HEAD_SHA,
    receipt: bool = True,
    review_id: int = 100,
) -> dict[str, object]:
    body = f"{REVIEW_MARKER}\n\n### Disposition: {disposition}\n"
    if receipt:
        payload = _payload(publisher_id=reviewer_id)
        payload["disposition"] = disposition
        body += _receipt_body(payload)
    return {
        "id": review_id,
        "commit_id": commit_id,
        "state": state,
        "user": {"id": reviewer_id},
        "body": body,
    }


def _software_payload(*, publisher_id: int = 20) -> dict[str, object]:
    return {
        "schema": "software_source_review_v1",
        "head_sha": HEAD_SHA,
        "publisher_user_id": publisher_id,
        "scope": "software-only",
        "disposition": "APPROVED",
        "reviewer_model_id": "gemini-3.8-flash-high",
        "review_jobs": ["a" * 16, "b" * 16],
        "review_result": "PASS",
        "blocking_findings": 0,
        "diff_sha256": "f" * 64,
        "report": (
            "Reviewed the complete exact-head software diff with read-only AGY, "
            "including the related contract tests. No blocking findings; "
            "known limitations recorded; no economic authorization."
        ),
    }


def _software_review(
    *, publisher_id: int = 20, state: str = "APPROVED"
) -> dict[str, object]:
    review = _review(reviewer_id=publisher_id, state=state, receipt=False)
    review["body"] = f"{REVIEW_MARKER}\n\n### Disposition: APPROVED\n" + _receipt_body(
        _software_payload(publisher_id=publisher_id)
    )
    return review


def _software_v2_payload(*, publisher_id: int = 20) -> dict[str, object]:
    return {
        "schema": "software_source_review_v2",
        "head_sha": HEAD_SHA,
        "publisher_user_id": publisher_id,
        "scope": "software-only",
        "disposition": "APPROVED",
        "reviewer_model_id": "gpt-5.6-sol",
        "review_result": "PASS",
        "blocking_findings": 0,
        "report": (
            "GPT-5.6 Sol reviewed the exact-head source diff, adjacent tests, "
            "failure paths and authority boundaries. No blocking software findings "
            "remain; this review does not authorize economic or live execution."
        ),
    }


def _software_v2_review(
    *, publisher_id: int = 20, state: str = "APPROVED", review_id: int = 110
) -> dict[str, object]:
    review = _review(
        reviewer_id=publisher_id,
        state=state,
        receipt=False,
        review_id=review_id,
    )
    review["body"] = f"{REVIEW_MARKER}\n\n### Disposition: APPROVED\n" + _receipt_body(
        _software_v2_payload(publisher_id=publisher_id)
    )
    return review


def test_account_separation_alone_does_not_establish_model_independence() -> None:
    outcome = resolve_review_outcome(
        _pull_request(), [_review(receipt=False)], expected_head_sha=HEAD_SHA
    )

    assert outcome is ReviewOutcome.PENDING


def test_author_comment_without_model_receipt_does_not_approve() -> None:
    outcome = resolve_review_outcome(
        _pull_request(),
        [_review(reviewer_id=10, state="COMMENTED", receipt=False)],
        expected_head_sha=HEAD_SHA,
    )

    assert outcome is ReviewOutcome.PENDING


def test_author_commented_review_with_distinct_models_satisfies_gate() -> None:
    outcome = resolve_review_outcome(
        _pull_request(),
        [_review(reviewer_id=10, state="COMMENTED")],
        expected_head_sha=HEAD_SHA,
    )

    assert outcome is ReviewOutcome.APPROVED


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
        f"{REVIEW_MARKER}\n\n### Disposition: APPROVED\n\n### Disposition: PENDING\n",
    ],
)
def test_unstructured_or_ambiguous_disposition_does_not_approve(body: str) -> None:
    review = _review()
    review["body"] = body + _receipt_body(_payload())

    outcome = resolve_review_outcome(
        _pull_request(), [review], expected_head_sha=HEAD_SHA
    )

    assert outcome is not ReviewOutcome.APPROVED


@pytest.mark.parametrize(
    "body",
    [
        str(_review()["body"]).replace(
            "### Disposition: APPROVED", "    ### Disposition: APPROVED"
        ),
        str(_review()["body"]).replace(
            "### Disposition: APPROVED", "\t### Disposition: APPROVED"
        ),
        str(_review()["body"]).replace(REVIEW_MARKER, f"    {REVIEW_MARKER}"),
        str(_review()["body"]).replace(REVIEW_MARKER, f"  \t{REVIEW_MARKER}"),
        f"<pre>\n{_review()['body']}\n</pre>",
        f'<PRE class="literal">\n{_review()["body"]}\n</PRE>',
        f"<pre\nclass=literal>\n{_review()['body']}\n</pre>",
        str(_review()["body"]).replace(
            "### Disposition: APPROVED", "<pre>\n### Disposition: APPROVED\n</pre>"
        ),
        f"{REVIEW_MARKER}\n### Disposition: APPROVED\n<pre>\n{_receipt_body(_payload())}\n</pre>",
        str(_review()["body"]).replace(
            "### Disposition: APPROVED", "<!-- hidden\n### Disposition: APPROVED\n-->"
        ),
    ],
    ids=[
        "space-indented-heading",
        "tab-indented-heading",
        "space-indented-marker",
        "tab-indented-marker",
        "pre-wrapped-protocol",
        "pre-attributes",
        "pre-multiline-opener",
        "pre-wrapped-heading",
        "pre-wrapped-receipt",
        "comment-wrapped-heading",
    ],
)
def test_literal_markdown_and_html_cannot_supply_review_protocol(body: str) -> None:
    review = _review()
    review["body"] = body

    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


@pytest.mark.parametrize("tag", ["textarea", "div", "table", "details"])
def test_unsupported_html_wrapper_cannot_supply_review_disposition(tag: str) -> None:
    review = _review()
    review["body"] = str(review["body"]).replace(
        "### Disposition: APPROVED", f"<{tag}>\n### Disposition: APPROVED\n</{tag}>"
    )

    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


@pytest.mark.parametrize(
    "prefix",
    ["\u00a0", "\u2003", "Example\u2028", "Example\u2029", "Example\v", "Example\f"],
)
def test_non_markdown_space_and_line_separators_cannot_create_disposition(
    prefix: str,
) -> None:
    review = _review()
    review["body"] = str(review["body"]).replace(
        "### Disposition: APPROVED", f"{prefix}### Disposition: APPROVED"
    )

    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


def test_receipt_report_may_quote_html_without_hiding_visible_protocol() -> None:
    payload = _payload()
    payload["report"] = (
        "Reviewed the <pre> literal-block failure path; no blocking findings."
    )
    review = _review()
    review["body"] = f"{REVIEW_MARKER}\n### Disposition: APPROVED\n" + _receipt_body(
        payload
    )

    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.APPROVED
    )


def test_unsupported_raw_html_outside_receipt_fails_closed() -> None:
    review = _review()
    review["body"] = f"<pre>Unrelated literal example.</pre>\n{review['body']}"

    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


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
            _review(
                reviewer_id=10,
                state="COMMENTED",
                disposition="BLOCKED",
                receipt=False,
            ),
        ],
        expected_head_sha=HEAD_SHA,
    )

    assert outcome is ReviewOutcome.BLOCKED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("head_sha", "b" * 40),
        ("publisher_user_id", 30),
        ("publisher_user_id", True),
        ("publisher_user_id", "20"),
        ("scope", "economic-execution"),
        ("disposition", "BLOCKED"),
        ("disposition", ["APPROVED"]),
        ("schema", "independent_model_review_v0"),
        ("implementation", []),
        ("report", ""),
        ("report", "\u0000"),
        ("unexpected", "ignored fields must not authorize"),
        (
            "implementation",
            [{"model_id": "unknown", "task_id": "/root/implementation"}],
        ),
        (
            "implementation",
            [
                {
                    "model_id": "gpt-6.1-sol",
                    "task_id": "/root/implementation",
                    "extra": True,
                }
            ],
        ),
        (
            "implementation",
            [{"model_id": "gpt-6.1-sol", "task_id": "/root/implementation"}] * 2,
        ),
    ],
)
def test_receipt_bindings_are_verified(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value
    review = _review()
    review["body"] = f"{REVIEW_MARKER}\n### Disposition: APPROVED\n" + _receipt_body(
        payload
    )

    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


@pytest.mark.parametrize(
    ("model_id", "task_id"),
    [
        ("gpt-6.1-sol", "/root/review"),
        ("GPT-6.1-SOL", "/root/review"),
        ("gpt_6_1_sol", "/root/review"),
        ("gpt6.1sol", "/root/review"),
        (" gpt-6.1-sol ", "/root/review"),
        ("", "/root/review"),
        ("unknown", "/root/review"),
        ("UNKNOWN", "/root/review"),
        ("gpt-6-astra", "/root/implementation"),
        ("gpt-6-astra", "unknown"),
        ("gpt-6-astra", " /root/review "),
        ("gpt-6-astra", "/root/review\nforged"),
    ],
)
def test_same_or_unestablished_model_and_task_identity_does_not_approve(
    model_id: str, task_id: str
) -> None:
    payload = _payload()
    payload["reviewer"] = {"model_id": model_id, "task_id": task_id}
    review = _review()
    review["body"] = f"{REVIEW_MARKER}\n### Disposition: APPROVED\n" + _receipt_body(
        payload
    )

    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


def test_reviewer_model_must_differ_from_every_implementation_model() -> None:
    payload = _payload()
    payload["implementation"] = [
        {"model_id": "gpt-6.1-sol", "task_id": "/root/implementation"},
        {"model_id": "gpt-6-astra", "task_id": "/root/second_implementation"},
    ]
    review = _review()
    review["body"] = f"{REVIEW_MARKER}\n### Disposition: APPROVED\n" + _receipt_body(
        payload
    )

    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


@pytest.mark.parametrize(
    "receipt_body",
    [
        _receipt_body(_payload(), digest="0" * 64),
        _receipt_body(_payload()) + _receipt_body(_payload()),
        _receipt_body(_payload()).removesuffix("```\n"),
        f"```md\n{_receipt_body(_payload())}\n```\n",
        f'{RECEIPT_FENCE}\n{{"payload":null,"payload":{{}},"sha256":"{"0" * 64}"}}\n```\n',
        _receipt_body(_payload()).replace(
            '"schema":', '"scope":"software-only","schema":'
        ),
        _receipt_body(_payload()).replace('"sha256":', '"unexpected":true,"sha256":'),
    ],
)
def test_missing_corrupt_or_ambiguous_receipt_does_not_approve(
    receipt_body: str,
) -> None:
    review = _review(receipt=False)
    review["body"] = f"{review['body']}{receipt_body}"

    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


@pytest.mark.parametrize("state", ["DISMISSED", "PENDING", "CHANGES_REQUESTED"])
def test_review_state_must_be_an_active_approval_or_comment(state: str) -> None:
    assert (
        resolve_review_outcome(
            _pull_request(), [_review(state=state)], expected_head_sha=HEAD_SHA
        )
        is ReviewOutcome.PENDING
    )


@pytest.mark.parametrize("state", [["APPROVED"], {"state": "APPROVED"}])
def test_malformed_api_state_fails_closed(state: object) -> None:
    review = _review()
    review["state"] = state
    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


def test_later_approval_cannot_erase_active_blocked_from_same_publisher() -> None:
    assert (
        resolve_review_outcome(
            _pull_request(),
            [
                _review(disposition="BLOCKED", state="COMMENTED", review_id=100),
                _review(review_id=101),
            ],
            expected_head_sha=HEAD_SHA,
        )
        is ReviewOutcome.BLOCKED
    )


@pytest.mark.parametrize("state", ["DISMISSED", "PENDING", "CHANGES_REQUESTED"])
def test_newer_marked_review_revokes_earlier_approval_from_same_publisher(
    state: str,
) -> None:
    assert (
        resolve_review_outcome(
            _pull_request(),
            [_review(state=state, review_id=101), _review(review_id=100)],
            expected_head_sha=HEAD_SHA,
        )
        is ReviewOutcome.PENDING
    )


@pytest.mark.parametrize("field", ["id", "user"])
def test_github_api_identity_must_have_native_positive_integer_ids(field: str) -> None:
    review = _review()
    review[field] = "100" if field == "id" else {"id": True}

    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


@pytest.mark.parametrize("valid_first", [True, False])
def test_ambiguous_duplicate_github_review_id_fails_closed(valid_first: bool) -> None:
    reviews = [_review(), _review(receipt=False)]
    if not valid_first:
        reviews.reverse()

    assert (
        resolve_review_outcome(_pull_request(), reviews, expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


def test_receipt_renderer_preserves_actual_report_and_computes_payload_digest(
    tmp_path: Path,
) -> None:
    payload = _payload(publisher_id=10)
    payload["report"] = "G0/G1: NOT APPLICABLE.\nG2: PASS.\nBlocking findings: none."
    payload_path = tmp_path / "receipt-payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    tool = Path(__file__).resolve().parents[2] / "tools/independent_research_review.py"

    result = subprocess.run(
        [sys.executable, str(tool), "--render-payload-json", str(payload_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    envelope = json.loads(
        result.stdout.split(f"{RECEIPT_FENCE}\n", 1)[1].split("\n```", 1)[0]
    )
    assert envelope["payload"] == payload
    expected_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    assert envelope["sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    review = _review(reviewer_id=10, state="COMMENTED")
    review["body"] = result.stdout
    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.APPROVED
    )


def test_cli_validates_owner_commented_receipt_from_github_evidence(
    tmp_path: Path,
) -> None:
    pull_path = tmp_path / "pull.json"
    reviews_path = tmp_path / "reviews.json"
    pull_path.write_text(json.dumps(_pull_request()), encoding="utf-8")
    reviews_path.write_text(
        json.dumps([_review(reviewer_id=10, state="COMMENTED")]), encoding="utf-8"
    )
    tool = Path(__file__).resolve().parents[2] / "tools/independent_research_review.py"

    result = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--pull-json",
            str(pull_path),
            "--reviews-json",
            str(reviews_path),
            "--expected-head",
            HEAD_SHA,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "APPROVED"


@pytest.mark.parametrize("publisher_id,state", [(10, "COMMENTED"), (20, "APPROVED")])
def test_syntactically_valid_unverified_software_jobs_cannot_authorize_approval(
    publisher_id: int, state: str
) -> None:
    # Neither these managed-job-shaped IDs nor the claimed diff was obtained
    # from a trusted review provider. A self-consistent JSON digest proves
    # only that the publisher did not alter the *text* of their own claim.
    review = _software_review(publisher_id=publisher_id, state=state)
    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


@pytest.mark.parametrize("publisher_id,state", [(10, "COMMENTED"), (20, "APPROVED")])
def test_lightweight_v2_exact_head_model_review_satisfies_software_gate(
    publisher_id: int, state: str
) -> None:
    review = _software_v2_review(publisher_id=publisher_id, state=state)
    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.APPROVED
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reviewer_model_id", "UNKNOWN"),
        ("review_result", "BLOCKED"),
        ("blocking_findings", 1),
        ("report", "too short"),
    ],
)
def test_lightweight_v2_rejects_invalid_review_claim(field: str, value: object) -> None:
    payload = _software_v2_payload()
    payload[field] = value
    review = _software_v2_review()
    review["body"] = f"{REVIEW_MARKER}\n\n### Disposition: APPROVED\n" + _receipt_body(
        payload
    )
    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


def test_lightweight_v2_stale_head_does_not_approve() -> None:
    review = _software_v2_review()
    review["commit_id"] = "b" * 40
    assert (
        resolve_review_outcome(_pull_request(), [review], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.PENDING
    )


def test_lightweight_v2_renderer_emits_exact_receipt(tmp_path: Path) -> None:
    tool = Path(__file__).resolve().parents[2] / "tools/independent_research_review.py"
    payload_path = tmp_path / "model-source-review.json"
    payload_path.write_text(json.dumps(_software_v2_payload()), encoding="utf-8")
    rendered = subprocess.run(
        [sys.executable, str(tool), "--render-payload-json", str(payload_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert REVIEW_MARKER in rendered
    assert "### Disposition: APPROVED" in rendered
    assert '"schema":"software_source_review_v2"' in rendered
    assert '"reviewer_model_id":"gpt-5.6-sol"' in rendered


def test_unverified_software_receipt_does_not_override_v1_or_blocking_review() -> None:
    unverified = _software_review(publisher_id=20)
    verified_v1 = _review(reviewer_id=30, review_id=101)
    blocked = _review(
        reviewer_id=40,
        review_id=102,
        state="COMMENTED",
        disposition="BLOCKED",
        receipt=False,
    )
    assert (
        resolve_review_outcome(
            _pull_request(), [unverified, verified_v1], expected_head_sha=HEAD_SHA
        )
        is ReviewOutcome.APPROVED
    )
    assert (
        resolve_review_outcome(
            _pull_request(),
            [unverified, verified_v1, blocked],
            expected_head_sha=HEAD_SHA,
        )
        is ReviewOutcome.BLOCKED
    )


def test_unverified_software_receipt_renderer_refuses_authorizing_template(
    tmp_path: Path,
) -> None:
    tool = Path(__file__).resolve().parents[2] / "tools/independent_research_review.py"
    payload_path = tmp_path / "publisher-claimed-review.json"
    payload_path.write_text(json.dumps(_software_payload()), encoding="utf-8")
    attempt = subprocess.run(
        [sys.executable, str(tool), "--render-payload-json", str(payload_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert attempt.returncode == 2
    assert "Invalid" in attempt.stderr
    assert attempt.stdout == ""


def test_software_review_blocked_precedence_and_latest_actor_revocation() -> None:
    accepted = _software_v2_review(publisher_id=20, review_id=100)
    assert (
        resolve_review_outcome(_pull_request(), [accepted], expected_head_sha=HEAD_SHA)
        is ReviewOutcome.APPROVED
    )

    blocked = _review(
        reviewer_id=30,
        state="COMMENTED",
        disposition="BLOCKED",
        receipt=False,
        review_id=101,
    )
    assert (
        resolve_review_outcome(
            _pull_request(), [accepted, blocked], expected_head_sha=HEAD_SHA
        )
        is ReviewOutcome.BLOCKED
    )

    revoked = _review(
        reviewer_id=20,
        state="COMMENTED",
        disposition="PENDING",
        receipt=False,
        review_id=102,
    )
    assert (
        resolve_review_outcome(
            _pull_request(), [accepted, revoked], expected_head_sha=HEAD_SHA
        )
        is ReviewOutcome.PENDING
    )


def test_missing_pull_request_author_identity_fails_closed() -> None:
    pull = {"head": {"sha": HEAD_SHA}, "user": {"id": "10"}}

    outcome = resolve_review_outcome(pull, [_review()], expected_head_sha=HEAD_SHA)

    assert outcome is ReviewOutcome.PENDING
