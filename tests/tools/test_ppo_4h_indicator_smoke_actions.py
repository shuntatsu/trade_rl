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
REVIEW_URL = "https://github.com/owner/repo/pull/900#pullrequestreview-12345"
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
        if url.endswith("/pulls/900/reviews/12345"):
            return {
                "html_url": REVIEW_URL,
                "body": body,
                "commit_id": review_commit,
                "user": {"id": reviewer_id, "login": "reviewer"},
                "state": "COMMENTED",
            }
        if url.endswith("/pulls/900"):
            return {
                "number": 900,
                "state": "open",
                "draft": True,
                "user": {"id": author_id, "login": "author"},
                "head": {
                    "ref": actions.EXECUTION_BRANCH,
                    "sha": TRIGGER_SHA,
                    "repo": {"full_name": "owner/repo"},
                },
                "base": {"ref": "main"},
            }
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


def test_review_gate_rejects_execution_pr_head_other_than_trigger_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)

    def wrong_head(url: str, **_kwargs: object) -> dict[str, object]:
        record = _github_api()(url)
        if url.endswith("/pulls/900"):
            head = dict(record["head"])
            head["sha"] = "d" * 40
            record = dict(record)
            record["head"] = head
        return record

    monkeypatch.setattr(actions.transport, "_api_json", wrong_head)

    with pytest.raises(ValueError, match="head"):
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

    with pytest.raises(ValueError, match="identity or bytes"):
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

    with pytest.raises(ValueError, match="invalid JSON"):
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


def test_review_gate_rejects_review_from_nonexecution_pull_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    review = _review()
    review["source_review_url"] = (
        "https://github.com/owner/repo/pull/901#pullrequestreview-12345"
    )

    def wrong_pull_api(url: str, **_kwargs: object) -> dict[str, object]:
        if url.endswith("/pulls/901/reviews/12345"):
            return {
                "html_url": review["source_review_url"],
                "body": REVIEW_BODY,
                "commit_id": REVIEWED_SHA,
                "user": {"id": 2, "login": "reviewer"},
                "state": "COMMENTED",
            }
        if url.endswith("/pulls/901"):
            return {
                "number": 901,
                "state": "open",
                "draft": True,
                "user": {"id": 1, "login": "author"},
                "head": {
                    "ref": "other-branch",
                    "sha": REVIEWED_SHA,
                    "repo": {"full_name": "owner/repo"},
                },
                "base": {"ref": "main"},
            }
        raise AssertionError(url)

    monkeypatch.setattr(actions.transport, "_api_json", wrong_pull_api)

    with pytest.raises(ValueError, match="execution"):
        actions.validate_review_gate(
            review,
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


@pytest.mark.parametrize(
    ("pull_updates", "match"),
    (
        ({"state": "closed"}, "open"),
        ({"draft": False}, "draft"),
        (
            {
                "head": {
                    "ref": "wrong",
                    "sha": REVIEWED_SHA,
                    "repo": {"full_name": "owner/repo"},
                },
            },
            "execution",
        ),
        (
            {
                "head": {
                    "ref": "research/ppo-4h-indicator-smoke-execution",
                    "sha": "c" * 40,
                    "repo": {"full_name": "owner/repo"},
                },
            },
            "head",
        ),
        (
            {
                "head": {
                    "ref": "research/ppo-4h-indicator-smoke-execution",
                    "sha": REVIEWED_SHA,
                    "repo": {"full_name": "other/repo"},
                },
            },
            "repository",
        ),
        ({"base": {"ref": "develop"}}, "base"),
    ),
)
def test_independent_review_rejects_wrong_execution_pr_identity(
    pull_updates: dict[str, object],
    match: str,
) -> None:
    pull: dict[str, object] = {
        "number": 900,
        "state": "open",
        "draft": True,
        "user": {"id": 1, "login": "author"},
        "head": {
            "ref": actions.EXECUTION_BRANCH,
            "sha": REVIEWED_SHA,
            "repo": {"full_name": "owner/repo"},
        },
        "base": {"ref": "main"},
    }
    pull.update(pull_updates)

    with pytest.raises(ValueError, match=match):
        actions.validate_independent_review_status(
            _status_review(),
            pull,
            repository="owner/repo",
            pull_number=900,
            reviewed_code_sha=REVIEWED_SHA,
        )


def _status_review(
    *,
    body: str = REVIEW_BODY,
    commit_id: str = REVIEWED_SHA,
    reviewer_id: int = 2,
    state: str = "COMMENTED",
    review_id: int = 12345,
) -> dict[str, object]:
    return {
        "id": review_id,
        "html_url": REVIEW_URL,
        "body": body,
        "commit_id": commit_id,
        "user": {"id": reviewer_id, "login": "reviewer"},
        "state": state,
    }


def _review_inventory_api(
    *,
    reviews: list[dict[str, object]],
    author_id: int = 1,
):
    def fake_object(url: str, **_kwargs: object) -> dict[str, object]:
        if url.endswith("/pulls/900"):
            return {
                "number": 900,
                "state": "open",
                "draft": True,
                "user": {"id": author_id, "login": "author"},
                "head": {
                    "ref": actions.EXECUTION_BRANCH,
                    "sha": REVIEWED_SHA,
                    "repo": {"full_name": "owner/repo"},
                },
                "base": {"ref": "main"},
            }
        raise AssertionError(url)

    def fake_array(url: str, **_kwargs: object) -> list[object]:
        if "/pulls/900/reviews?" not in url:
            raise AssertionError(url)
        if "&page=1" in url:
            return list(reviews)
        return []

    return fake_object, fake_array


def test_independent_review_status_accepts_exact_result_blind_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_api, array_api = _review_inventory_api(reviews=[_status_review()])
    monkeypatch.setattr(actions.transport, "_api_json", object_api)
    monkeypatch.setattr(actions.transport, "_api_json_array", array_api)

    record = actions.find_authorizing_source_review(
        repository="owner/repo",
        pull_number=900,
        reviewed_code_sha=REVIEWED_SHA,
        token="token",
        deadline=999999999.0,
    )

    assert record["id"] == 12345


@pytest.mark.parametrize(
    "review",
    (
        _status_review(reviewer_id=1),
        _status_review(commit_id="c" * 40),
        _status_review(state="DISMISSED"),
        _status_review(
            body=_source_review_body(reviewer_independence="NOT_ESTABLISHED")
        ),
        _status_review(body=_source_review_body(result_blind=False)),
        _status_review(body=_source_review_body(g0="FAIL")),
        _status_review(body=_source_review_body(g1="NOT_ESTABLISHED")),
        _status_review(body=_source_review_body(g2="FAIL")),
        _status_review(body=_source_review_body(blocking_findings=["block"])),
        _status_review(body=_source_review_body(unused_data_accessed=True)),
        _status_review(body=_source_review_body(final_data_accessed=True)),
    ),
)
def test_independent_review_status_rejects_non_authorizing_reviews(
    monkeypatch: pytest.MonkeyPatch,
    review: dict[str, object],
) -> None:
    object_api, array_api = _review_inventory_api(reviews=[review])
    monkeypatch.setattr(actions.transport, "_api_json", object_api)
    monkeypatch.setattr(actions.transport, "_api_json_array", array_api)

    with pytest.raises(ValueError, match="independent research review is pending"):
        actions.find_authorizing_source_review(
            repository="owner/repo",
            pull_number=900,
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_independent_review_status_scans_later_review_inventory_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = [
        _status_review(reviewer_id=1, review_id=index + 1) for index in range(100)
    ]
    for index, review in enumerate(invalid, start=1):
        review["html_url"] = (
            "https://github.com/owner/repo/pull/900#pullrequestreview-" + str(index)
        )
    valid = _status_review(review_id=101)
    valid["html_url"] = "https://github.com/owner/repo/pull/900#pullrequestreview-101"

    def fake_object(url: str, **_kwargs: object) -> dict[str, object]:
        if url.endswith("/pulls/900"):
            return {
                "number": 900,
                "state": "open",
                "draft": True,
                "user": {"id": 1, "login": "author"},
                "head": {
                    "ref": actions.EXECUTION_BRANCH,
                    "sha": REVIEWED_SHA,
                    "repo": {"full_name": "owner/repo"},
                },
                "base": {"ref": "main"},
            }
        raise AssertionError(url)

    def fake_array(url: str, **_kwargs: object) -> list[object]:
        if "&page=1" in url:
            return list(invalid)
        if "&page=2" in url:
            return [valid]
        return []

    monkeypatch.setattr(actions.transport, "_api_json", fake_object)
    monkeypatch.setattr(actions.transport, "_api_json_array", fake_array)

    record = actions.find_authorizing_source_review(
        repository="owner/repo",
        pull_number=900,
        reviewed_code_sha=REVIEWED_SHA,
        token="token",
        deadline=999999999.0,
    )

    assert record["id"] == 101


def test_independent_review_status_keeps_valid_review_when_later_review_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = _status_review(review_id=12345)
    later_author_review = _status_review(review_id=12346, reviewer_id=1)
    later_author_review["html_url"] = (
        "https://github.com/owner/repo/pull/900#pullrequestreview-12346"
    )
    object_api, array_api = _review_inventory_api(reviews=[valid, later_author_review])
    monkeypatch.setattr(actions.transport, "_api_json", object_api)
    monkeypatch.setattr(actions.transport, "_api_json_array", array_api)

    record = actions.find_authorizing_source_review(
        repository="owner/repo",
        pull_number=900,
        reviewed_code_sha=REVIEWED_SHA,
        token="token",
        deadline=999999999.0,
    )

    assert record["id"] == 12345


def test_review_status_environment_is_pending_without_authorizing_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    event = tmp_path / "event.json"
    event.write_text(
        '{"pull_request":{"number":900,"head":{"sha":"' + REVIEWED_SHA + '"}}}',
        encoding="utf-8",
    )
    summary = tmp_path / "summary.md"
    object_api, array_api = _review_inventory_api(reviews=[])
    monkeypatch.setattr(actions.transport, "_api_json", object_api)
    monkeypatch.setattr(actions.transport, "_api_json_array", array_api)

    result = actions.execute_review_status_from_environment(
        {
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_TOKEN": "token",
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_STEP_SUMMARY": str(summary),
        }
    )

    assert result == 1
    assert "PENDING" in summary.read_text(encoding="utf-8")
    assert REVIEWED_SHA in summary.read_text(encoding="utf-8")


def test_review_status_environment_writes_ready_summary_after_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    event = tmp_path / "event.json"
    event.write_text(
        '{"pull_request":{"number":900,"head":{"sha":"' + REVIEWED_SHA + '"}}}',
        encoding="utf-8",
    )
    summary = tmp_path / "summary.md"
    object_api, array_api = _review_inventory_api(reviews=[_status_review()])
    monkeypatch.setattr(actions.transport, "_api_json", object_api)
    monkeypatch.setattr(actions.transport, "_api_json_array", array_api)

    result = actions.execute_review_status_from_environment(
        {
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_TOKEN": "token",
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_STEP_SUMMARY": str(summary),
        }
    )

    assert result == 0
    assert "READY" in summary.read_text(encoding="utf-8")
    assert REVIEWED_SHA in summary.read_text(encoding="utf-8")
