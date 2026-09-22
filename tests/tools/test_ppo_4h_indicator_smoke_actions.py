from __future__ import annotations

import hashlib
import importlib
from copy import deepcopy

import pytest

from trade_rl.artifacts import canonical_json_bytes, content_digest

actions = importlib.import_module("tools.ppo_4h_indicator_smoke_actions")
smoke = importlib.import_module("trade_rl.evaluation.ppo_4h_indicator_smoke")


REVIEWED_SHA = "a" * 40
TRIGGER_SHA = "b" * 40
REVIEW_URL = "https://github.com/owner/repo/pull/758#pullrequestreview-12345"
SOURCE_REVIEW_SCHEMA = "ppo_4h_indicator_source_review_v2"
SOURCE_REVIEW_MARKER = "<!-- ppo-4h-indicator-source-review-v2 -->\n"


def _source_review_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SOURCE_REVIEW_SCHEMA,
        "reviewed_code_sha": REVIEWED_SHA,
        "static_contract_digest": content_digest(smoke.static_protocol_contract()),
        "reviewer_independence": "ESTABLISHED",
        "result_blind": True,
        "g0": "PASS",
        "g1": "PASS",
        "g2": "EVIDENCE_BOUND",
        "disposition": "G0_G1_CLEAR_G2_EVIDENCE_BOUND",
        "blocking_findings": [],
        "authorized_development_smoke": True,
        "unused_data_accessed": False,
        "final_data_accessed": False,
    }
    payload.update(updates)
    return payload


def _source_review_body(**updates: object) -> str:
    payload = _source_review_payload(**updates)
    return (
        "<!-- hourly-agent-review -->\n"
        "Result-blind structured review.\n"
        f"Disposition: {payload['disposition']}.\n"
        + SOURCE_REVIEW_MARKER
        + canonical_json_bytes(payload).decode("utf-8")
        + "\n"
    )


REVIEW_BODY = _source_review_body()


def _review() -> dict[str, object]:
    return {
        "schema": actions.REVIEW_SCHEMA,
        "reviewed_code_sha": REVIEWED_SHA,
        "static_contract_digest": content_digest(smoke.static_protocol_contract()),
        "source_review_url": REVIEW_URL,
        "source_review_body_sha256": hashlib.sha256(
            REVIEW_BODY.encode("utf-8")
        ).hexdigest(),
        "reviewer_surface": "github_pr_review_v2",
        "result_blind": True,
        "g0": "PASS",
        "g1": "PASS",
        "g2": "EVIDENCE_BOUND",
        "authorized_development_smoke": True,
        "unused_data_accessed": False,
        "final_data_accessed": False,
    }


def _fake_git(*args: str) -> str:
    if args == ("rev-parse", "HEAD"):
        return TRIGGER_SHA
    if args == ("rev-parse", "HEAD^"):
        return REVIEWED_SHA
    if args == ("log", "-1", "--pretty=%s"):
        return actions.TRIGGER_MESSAGE
    if args == ("diff", "--name-only", REVIEWED_SHA, TRIGGER_SHA):
        return actions.REVIEW_PATH.as_posix()
    raise AssertionError(args)


def _github_api(
    *,
    body: str = REVIEW_BODY,
    review_commit: str = REVIEWED_SHA,
    reviewer_id: int = 2,
    author_id: int = 1,
):
    def fake(url: str, **_kwargs: object) -> dict[str, object]:
        if url.endswith("/pulls/758/reviews/12345"):
            return {
                "html_url": REVIEW_URL,
                "body": body,
                "commit_id": review_commit,
                "user": {"id": reviewer_id, "login": "reviewer"},
                "state": "COMMENTED",
            }
        if url.endswith("/pulls/758"):
            return {"user": {"id": author_id, "login": "author"}}
        raise AssertionError(url)

    return fake


def test_review_gate_binds_exact_parent_contract_and_review_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(actions.transport, "_api_json", _github_api())

    actions.validate_review_gate(
        _review(),
        repository="owner/repo",
        token="token",
        deadline=999999999.0,
    )


@pytest.mark.parametrize(
    "field",
    (
        "reviewed_code_sha",
        "static_contract_digest",
        "result_blind",
        "authorized_development_smoke",
        "unused_data_accessed",
    ),
)
def test_review_gate_rejects_unbound_or_non_authorizing_evidence(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(actions.transport, "_api_json", _github_api())
    review = deepcopy(_review())
    if field == "reviewed_code_sha":
        review[field] = "c" * 40
    elif field == "static_contract_digest":
        review[field] = "d" * 64
    elif field in {"result_blind", "authorized_development_smoke"}:
        review[field] = False
    else:
        review[field] = True

    with pytest.raises(ValueError):
        actions.validate_review_gate(
            review,
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_rejects_edited_source_review_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(
        actions.transport,
        "_api_json",
        _github_api(body=REVIEW_BODY + " edited"),
    )

    with pytest.raises(ValueError, match="comment"):
        actions.validate_review_gate(
            _review(),
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_rejects_missing_structured_source_review_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (
        "<!-- hourly-agent-review -->\n"
        "Result-blind review.\n"
        "G0 PASS, G1 PASS, G2 PASS.\n"
        "Disposition: G0_G1_CLEAR_G2_EVIDENCE_BOUND.\n"
    )
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(actions.transport, "_api_json", _github_api(body=body))
    review = _review()
    review["source_review_body_sha256"] = hashlib.sha256(
        body.encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="structured"):
        actions.validate_review_gate(
            review,
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_rejects_noncanonical_trailing_source_review_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = REVIEW_BODY + "BLOCKING before economic execution.\n"
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(actions.transport, "_api_json", _github_api(body=body))
    review = _review()
    review["source_review_body_sha256"] = hashlib.sha256(
        body.encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="canonical"):
        actions.validate_review_gate(
            review,
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )

def test_review_gate_rejects_review_from_pr_author(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(
        actions.transport,
        "_api_json",
        _github_api(reviewer_id=1, author_id=1),
    )

    with pytest.raises(ValueError, match="independent"):
        actions.validate_review_gate(
            _review(),
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_rejects_review_bound_to_another_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(
        actions.transport,
        "_api_json",
        _github_api(review_commit="c" * 40),
    )

    with pytest.raises(ValueError, match="commit"):
        actions.validate_review_gate(
            _review(),
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


@pytest.mark.parametrize(
    ("updates", "match"),
    (
        ({"reviewer_independence": "NOT_ESTABLISHED"}, "independence"),
        ({"result_blind": False}, "result-blind"),
        ({"g0": "FAIL"}, "G0"),
        ({"g1": "NOT_ESTABLISHED"}, "G1"),
        ({"disposition": "READY_FOR_MACHINE_VERIFICATION"}, "disposition"),
        ({"blocking_findings": ["review provenance unknown"]}, "blocking"),
    ),
)
def test_review_gate_derives_authorization_from_source_review_payload(
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    match: str,
) -> None:
    body = _source_review_body(**updates)
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(actions.transport, "_api_json", _github_api(body=body))
    review = _review()
    review["source_review_body_sha256"] = hashlib.sha256(
        body.encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match=match):
        actions.validate_review_gate(
            review,
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_rejects_issue_comment_as_authorization_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    review = _review()
    review["source_review_url"] = (
        "https://github.com/owner/repo/pull/758#issuecomment-12345"
    )

    with pytest.raises(ValueError, match="review reference"):
        actions.validate_review_gate(
            review,
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_rejects_review_from_another_pull_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    review = _review()
    review["source_review_url"] = (
        "https://github.com/owner/repo/pull/759#pullrequestreview-12345"
    )

    with pytest.raises(ValueError, match="pull request"):
        actions.validate_review_gate(
            review,
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )
