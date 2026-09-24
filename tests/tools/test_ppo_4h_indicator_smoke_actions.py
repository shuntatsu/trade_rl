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
CURRENT_MAIN_SHA = "d" * 40
REVIEW_TAG = "review/ppo-4h-indicator-smoke-v1"
REVIEW_TAG_OBJECT_SHA = "e" * 40
REVIEW_URL = "https://github.com/owner/repo/pull/900#pullrequestreview-12345"
SOURCE_REVIEW_SCHEMA = "ppo_4h_indicator_source_review_v3"
SOURCE_REVIEW_MARKER = "<!-- ppo-4h-indicator-source-review-v3 -->\n"


def _source_review_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SOURCE_REVIEW_SCHEMA,
        "reviewed_code_sha": REVIEWED_SHA,
        "static_contract_digest": content_digest(smoke.static_protocol_contract()),
        "review_tag": REVIEW_TAG,
        "review_tag_object_sha": REVIEW_TAG_OBJECT_SHA,
        "reviewer_independence": "ESTABLISHED",
        "reviewer_kind": "external_ai",
        "reviewer_model": "external-reviewer-v1",
        "reviewer_context": "fresh_read_only",
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
        "review_tag": REVIEW_TAG,
        "review_tag_object_sha": REVIEW_TAG_OBJECT_SHA,
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
    reviewer_login: str = "reviewer",
    author_id: int = 1,
    reviewer_permission: str = "write",
    tag_ref_type: str = "tag",
    tag_ref_sha: str = REVIEW_TAG_OBJECT_SHA,
    tag_object_name: str = REVIEW_TAG,
    tag_target_type: str = "commit",
    tag_target_sha: str = REVIEWED_SHA,
    tag_exists: bool = True,
):
    def fake(url: str, **_kwargs: object) -> dict[str, object]:
        if url.endswith("/pulls/900/reviews/12345"):
            return {
                "id": 12345,
                "html_url": REVIEW_URL,
                "body": body,
                "commit_id": review_commit,
                "user": {"id": reviewer_id, "login": reviewer_login},
                "state": "COMMENTED",
            }
        if "/collaborators/" in url and url.endswith("/permission"):
            return {"permission": reviewer_permission}
        if url.endswith(f"/git/ref/tags/{REVIEW_TAG}"):
            if not tag_exists:
                raise actions.transport.TransportError(
                    "GitHub API request failed with HTTP 404"
                )
            return {
                "ref": f"refs/tags/{REVIEW_TAG}",
                "object": {"type": tag_ref_type, "sha": tag_ref_sha},
            }
        if url.endswith(f"/git/tags/{REVIEW_TAG_OBJECT_SHA}"):
            return {
                "tag": tag_object_name,
                "object": {"type": tag_target_type, "sha": tag_target_sha},
            }
        if url.endswith("/branches/main"):
            return {"commit": {"sha": CURRENT_MAIN_SHA}}
        if f"/compare/{CURRENT_MAIN_SHA}...{REVIEWED_SHA}" in url:
            return {"status": "ahead", "behind_by": 0}
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
    api = _github_api()
    monkeypatch.setattr(actions.transport, "_api_json", api)

    def review_inventory(url: str, **_kwargs: object) -> list[object]:
        if "/pulls/900/reviews?" not in url:
            raise AssertionError(url)
        if "&page=1" in url:
            return [
                api("https://api.github.com/repos/owner/repo/pulls/900/reviews/12345")
            ]
        return []

    monkeypatch.setattr(actions.transport, "_api_json_array", review_inventory)

    actions.validate_review_gate(
        _review(),
        repository="owner/repo",
        token="token",
        deadline=999999999.0,
    )


@pytest.mark.parametrize(
    ("tag_ref_type", "tag_ref_sha", "tag_target_sha", "tag_exists", "match"),
    (
        ("commit", REVIEW_TAG_OBJECT_SHA, REVIEWED_SHA, True, "annotated"),
        ("tag", "c" * 40, REVIEWED_SHA, True, "object SHA"),
        ("tag", REVIEW_TAG_OBJECT_SHA, "c" * 40, True, "reviewed code"),
        (
            "tag",
            REVIEW_TAG_OBJECT_SHA,
            REVIEWED_SHA,
            False,
            "identity could not be verified",
        ),
    ),
)
def test_review_gate_rejects_invalid_review_tag_identity(
    monkeypatch: pytest.MonkeyPatch,
    tag_ref_type: str,
    tag_ref_sha: str,
    tag_target_sha: str,
    tag_exists: bool,
    match: str,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    api = _github_api(
        tag_ref_type=tag_ref_type,
        tag_ref_sha=tag_ref_sha,
        tag_target_sha=tag_target_sha,
        tag_exists=tag_exists,
    )
    monkeypatch.setattr(actions.transport, "_api_json", api)

    def review_inventory(url: str, **_kwargs: object) -> list[object]:
        if "&page=1" in url:
            return [
                api("https://api.github.com/repos/owner/repo/pulls/900/reviews/12345")
            ]
        return []

    monkeypatch.setattr(actions.transport, "_api_json_array", review_inventory)

    with pytest.raises(ValueError, match=match):
        actions.validate_review_gate(
            _review(),
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


@pytest.mark.parametrize(
    ("tag_object_name", "tag_target_type", "match"),
    (
        ("review/ppo-4h-indicator-smoke-v2", "commit", "object name"),
        (REVIEW_TAG, "tag", "does not point to a commit"),
    ),
)
def test_review_gate_rejects_malformed_annotated_tag_object(
    monkeypatch: pytest.MonkeyPatch,
    tag_object_name: str,
    tag_target_type: str,
    match: str,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    api = _github_api(
        tag_object_name=tag_object_name,
        tag_target_type=tag_target_type,
    )
    monkeypatch.setattr(actions.transport, "_api_json", api)

    def review_inventory(url: str, **_kwargs: object) -> list[object]:
        if "&page=1" in url:
            return [
                api("https://api.github.com/repos/owner/repo/pulls/900/reviews/12345")
            ]
        return []

    monkeypatch.setattr(actions.transport, "_api_json_array", review_inventory)

    with pytest.raises(ValueError, match=match):
        actions.validate_review_gate(
            _review(),
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_rejects_review_tag_ref_changed_during_identity_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    stable_api = _github_api()
    ref_reads = 0

    def moving_ref_api(url: str, **kwargs: object) -> dict[str, object]:
        nonlocal ref_reads
        if url.endswith(f"/git/ref/tags/{REVIEW_TAG}"):
            ref_reads += 1
            if ref_reads == 2:
                return {
                    "ref": f"refs/tags/{REVIEW_TAG}",
                    "object": {"type": "tag", "sha": "c" * 40},
                }
        return stable_api(url, **kwargs)

    monkeypatch.setattr(actions.transport, "_api_json", moving_ref_api)

    def review_inventory(url: str, **_kwargs: object) -> list[object]:
        if "&page=1" in url:
            return [
                stable_api(
                    "https://api.github.com/repos/owner/repo/pulls/900/reviews/12345"
                )
            ]
        return []

    monkeypatch.setattr(actions.transport, "_api_json_array", review_inventory)

    with pytest.raises(ValueError, match="changed during identity"):
        actions.validate_review_gate(
            _review(),
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("review_tag", "review/ppo-4h-indicator-smoke-v2"),
        ("review_tag_object_sha", "c" * 40),
    ),
)
def test_review_gate_rejects_trigger_tag_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    api = _github_api()
    monkeypatch.setattr(actions.transport, "_api_json", api)

    def review_inventory(url: str, **_kwargs: object) -> list[object]:
        if "&page=1" in url:
            return [
                api("https://api.github.com/repos/owner/repo/pulls/900/reviews/12345")
            ]
        return []

    monkeypatch.setattr(actions.transport, "_api_json_array", review_inventory)
    review = deepcopy(_review())
    review[field] = value

    with pytest.raises(ValueError, match="tag identity"):
        actions.validate_review_gate(
            review,
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_trigger_review_schema_versions_tag_identity_contract() -> None:
    assert actions.REVIEW_SCHEMA == "ppo_4h_indicator_smoke_review_v2"


def test_canonical_trigger_review_requires_review_tag_identity(tmp_path) -> None:
    path = tmp_path / "review.json"
    review = _review()
    path.write_bytes(canonical_json_bytes(review))

    assert actions._canonical_review(path) == review

    for field in ("review_tag", "review_tag_object_sha"):
        invalid = deepcopy(review)
        del invalid[field]
        path.write_bytes(canonical_json_bytes(invalid))
        with pytest.raises(ValueError, match="shape"):
            actions._canonical_review(path)


def test_canonical_trigger_review_rejects_legacy_v1_schema(tmp_path) -> None:
    path = tmp_path / "review.json"
    review = _review()
    review["schema"] = "ppo_4h_indicator_smoke_review_v1"
    path.write_bytes(canonical_json_bytes(review))

    with pytest.raises(ValueError, match="shape"):
        actions._canonical_review(path)


def test_execute_rechecks_review_gate_immediately_before_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        actions.transport,
        "available_memory_bytes",
        lambda: actions.MINIMUM_AVAILABLE_BYTES + 1,
    )
    monkeypatch.setattr(actions, "_canonical_review", lambda _path: _review())
    monkeypatch.setattr(
        actions,
        "validate_review_gate",
        lambda *_args, **_kwargs: events.append("gate"),
    )
    monkeypatch.setattr(
        actions,
        "_download_source",
        lambda **kwargs: kwargs["root"],
    )

    def prepare(_source, output) -> None:
        events.append("prepare")
        output.mkdir(parents=True, exist_ok=False)

    def run(_source, _output) -> None:
        events.append("run")

    monkeypatch.setattr(actions.smoke, "prepare_smoke", prepare)
    monkeypatch.setattr(actions.smoke, "run_smoke", run)

    result = actions.execute_from_environment(
        {
            "GITHUB_REPOSITORY": "owner/repo",
            "SMOKE_GITHUB_TOKEN": "token",
            "SMOKE_OUTPUT": str(tmp_path / "smoke"),
        }
    )

    assert result == 0
    assert events == ["gate", "prepare", "gate", "run"]


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


def test_review_gate_rejects_external_ai_review_posted_by_pr_author(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    api = _github_api(reviewer_id=1, author_id=1)
    monkeypatch.setattr(actions.transport, "_api_json", api)

    def review_inventory(url: str, **_kwargs: object) -> list[object]:
        if "/pulls/900/reviews?" not in url:
            raise AssertionError(url)
        if "&page=1" in url:
            return [
                api("https://api.github.com/repos/owner/repo/pulls/900/reviews/12345")
            ]
        return []

    monkeypatch.setattr(actions.transport, "_api_json_array", review_inventory)

    with pytest.raises(ValueError, match="not independent"):
        actions.validate_review_gate(
            _review(),
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_rejects_reviewer_without_write_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(
        actions.transport,
        "_api_json",
        _github_api(reviewer_permission="read"),
    )

    with pytest.raises(ValueError, match="write permission"):
        actions.validate_review_gate(
            _review(),
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_accepts_write_authorized_bot_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    api = _github_api(reviewer_login="research-reviewer[bot]")
    monkeypatch.setattr(actions.transport, "_api_json", api)

    def review_inventory(url: str, **_kwargs: object) -> list[object]:
        if "/pulls/900/reviews?" not in url:
            raise AssertionError(url)
        if "&page=1" in url:
            return [
                api("https://api.github.com/repos/owner/repo/pulls/900/reviews/12345")
            ]
        return []

    monkeypatch.setattr(actions.transport, "_api_json_array", review_inventory)
    actions.validate_review_gate(
        _review(),
        repository="owner/repo",
        token="token",
        deadline=999999999.0,
    )


def test_review_gate_rejects_superseded_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(actions.transport, "_api_json", _github_api())
    valid = _status_review(review_id=12345)
    later_blocking = _status_review(review_id=12346, state="CHANGES_REQUESTED")
    later_blocking["html_url"] = (
        "https://github.com/owner/repo/pull/900#pullrequestreview-12346"
    )

    def review_inventory(url: str, **_kwargs: object) -> list[object]:
        if "/pulls/900/reviews?" not in url:
            raise AssertionError(url)
        if "&page=1" in url:
            return [valid, later_blocking]
        return []

    monkeypatch.setattr(actions.transport, "_api_json_array", review_inventory)

    with pytest.raises(ValueError, match="superseded"):
        actions.validate_review_gate(
            _review(),
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("body", REVIEW_BODY + " edited"),
        ("commit_id", "c" * 40),
        ("state", "DISMISSED"),
        (
            "html_url",
            "https://github.com/owner/repo/pull/900#pullrequestreview-54321",
        ),
        ("reviewer_login", "renamed-reviewer"),
    ),
)
def test_review_gate_rejects_source_review_changed_during_inventory_refresh(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(actions.transport, "_api_json", _github_api())
    changed = _status_review(review_id=12345)
    if field == "reviewer_login":
        changed["user"] = {"id": 2, "login": value}
    else:
        changed[field] = value

    def review_inventory(url: str, **_kwargs: object) -> list[object]:
        if "/pulls/900/reviews?" not in url:
            raise AssertionError(url)
        if "&page=1" in url:
            return [changed]
        return []

    monkeypatch.setattr(actions.transport, "_api_json_array", review_inventory)

    with pytest.raises(ValueError, match="changed during authorization"):
        actions.validate_review_gate(
            _review(),
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_keeps_authorization_when_other_reviewer_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(actions.transport, "_api_json", _github_api())
    valid = _status_review(review_id=12345)
    later_other_reviewer = _status_review(
        review_id=12346,
        reviewer_id=3,
        state="CHANGES_REQUESTED",
    )
    later_other_reviewer["html_url"] = (
        "https://github.com/owner/repo/pull/900#pullrequestreview-12346"
    )

    def review_inventory(url: str, **_kwargs: object) -> list[object]:
        if "/pulls/900/reviews?" not in url:
            raise AssertionError(url)
        if "&page=1" in url:
            return [valid, later_other_reviewer]
        return []

    monkeypatch.setattr(actions.transport, "_api_json_array", review_inventory)

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
        ({"reviewer_kind": "human"}, "external AI"),
        ({"reviewer_model": "   "}, "model provenance"),
        ({"reviewer_context": "worker_context"}, "fresh read-only"),
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


@pytest.mark.parametrize(
    ("status", "behind_by"),
    (
        ("diverged", 1),
        ("behind", 1),
    ),
)
def test_review_authorization_rejects_code_not_containing_current_main(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    behind_by: int,
) -> None:
    def fake(url: str, **_kwargs: object) -> dict[str, object]:
        if url.endswith("/branches/main"):
            return {"commit": {"sha": CURRENT_MAIN_SHA}}
        if f"/compare/{CURRENT_MAIN_SHA}...{REVIEWED_SHA}" in url:
            return {"status": status, "behind_by": behind_by}
        raise AssertionError(url)

    monkeypatch.setattr(actions.transport, "_api_json", fake)

    with pytest.raises(ValueError, match="current main"):
        actions._require_current_main_contained(
            repository="owner/repo",
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_review_authorization_accepts_code_containing_current_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(url: str, **_kwargs: object) -> dict[str, object]:
        if url.endswith("/branches/main"):
            return {"commit": {"sha": CURRENT_MAIN_SHA}}
        if f"/compare/{CURRENT_MAIN_SHA}...{REVIEWED_SHA}" in url:
            return {"status": "ahead", "behind_by": 0}
        raise AssertionError(url)

    monkeypatch.setattr(actions.transport, "_api_json", fake)

    assert (
        actions._require_current_main_contained(
            repository="owner/repo",
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )
        == CURRENT_MAIN_SHA
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
    reviewer_permission: str = "write",
    tag_ref_type: str = "tag",
    tag_ref_sha: str = REVIEW_TAG_OBJECT_SHA,
    tag_object_name: str = REVIEW_TAG,
    tag_target_type: str = "commit",
    tag_target_sha: str = REVIEWED_SHA,
    tag_exists: bool = True,
):
    def fake_object(url: str, **_kwargs: object) -> dict[str, object]:
        if "/collaborators/" in url and url.endswith("/permission"):
            return {"permission": reviewer_permission}
        if url.endswith(f"/git/ref/tags/{REVIEW_TAG}"):
            if not tag_exists:
                raise actions.transport.TransportError(
                    "GitHub API request failed with HTTP 404"
                )
            return {
                "ref": f"refs/tags/{REVIEW_TAG}",
                "object": {"type": tag_ref_type, "sha": tag_ref_sha},
            }
        if url.endswith(f"/git/tags/{REVIEW_TAG_OBJECT_SHA}"):
            return {
                "tag": tag_object_name,
                "object": {"type": tag_target_type, "sha": tag_target_sha},
            }
        if url.endswith("/branches/main"):
            return {"commit": {"sha": CURRENT_MAIN_SHA}}
        if f"/compare/{CURRENT_MAIN_SHA}...{REVIEWED_SHA}" in url:
            return {"status": "ahead", "behind_by": 0}
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


def test_independent_review_status_rejects_review_posted_by_pr_author(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_api, array_api = _review_inventory_api(
        reviews=[_status_review(reviewer_id=1)],
        author_id=1,
    )
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


@pytest.mark.parametrize(
    ("tag_ref_type", "tag_ref_sha", "tag_target_sha", "tag_exists"),
    (
        ("commit", REVIEW_TAG_OBJECT_SHA, REVIEWED_SHA, True),
        ("tag", "c" * 40, REVIEWED_SHA, True),
        ("tag", REVIEW_TAG_OBJECT_SHA, "c" * 40, True),
        ("tag", REVIEW_TAG_OBJECT_SHA, REVIEWED_SHA, False),
    ),
)
def test_independent_review_status_rejects_invalid_review_tag(
    monkeypatch: pytest.MonkeyPatch,
    tag_ref_type: str,
    tag_ref_sha: str,
    tag_target_sha: str,
    tag_exists: bool,
) -> None:
    object_api, array_api = _review_inventory_api(
        reviews=[_status_review()],
        tag_ref_type=tag_ref_type,
        tag_ref_sha=tag_ref_sha,
        tag_target_sha=tag_target_sha,
        tag_exists=tag_exists,
    )
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


@pytest.mark.parametrize(
    ("tag_object_name", "tag_target_type"),
    (
        ("review/ppo-4h-indicator-smoke-v2", "commit"),
        (REVIEW_TAG, "tag"),
    ),
)
def test_independent_review_status_rejects_malformed_annotated_tag_object(
    monkeypatch: pytest.MonkeyPatch,
    tag_object_name: str,
    tag_target_type: str,
) -> None:
    object_api, array_api = _review_inventory_api(
        reviews=[_status_review()],
        tag_object_name=tag_object_name,
        tag_target_type=tag_target_type,
    )
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


def test_independent_review_status_rejects_reviewer_without_write_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_api, array_api = _review_inventory_api(
        reviews=[_status_review()], reviewer_permission="read"
    )
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


@pytest.mark.parametrize(
    "review",
    (
        _status_review(commit_id="c" * 40),
        _status_review(state="DISMISSED"),
        _status_review(
            body=_source_review_body(reviewer_independence="NOT_ESTABLISHED")
        ),
        _status_review(body=_source_review_body(reviewer_kind="human")),
        _status_review(body=_source_review_body(reviewer_model="   ")),
        _status_review(body=_source_review_body(reviewer_context="worker_context")),
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
        _status_review(reviewer_id=1, state="DISMISSED", review_id=index + 1)
        for index in range(100)
    ]
    for index, review in enumerate(invalid, start=1):
        review["html_url"] = (
            "https://github.com/owner/repo/pull/900#pullrequestreview-" + str(index)
        )
    valid = _status_review(review_id=101)
    valid["html_url"] = "https://github.com/owner/repo/pull/900#pullrequestreview-101"

    def fake_object(url: str, **_kwargs: object) -> dict[str, object]:
        if url.endswith("/collaborators/reviewer/permission"):
            return {"permission": "write"}
        if url.endswith(f"/git/ref/tags/{REVIEW_TAG}"):
            return {
                "ref": f"refs/tags/{REVIEW_TAG}",
                "object": {"type": "tag", "sha": REVIEW_TAG_OBJECT_SHA},
            }
        if url.endswith(f"/git/tags/{REVIEW_TAG_OBJECT_SHA}"):
            return {
                "tag": REVIEW_TAG,
                "object": {"type": "commit", "sha": REVIEWED_SHA},
            }
        if url.endswith("/branches/main"):
            return {"commit": {"sha": CURRENT_MAIN_SHA}}
        if f"/compare/{CURRENT_MAIN_SHA}...{REVIEWED_SHA}" in url:
            return {"status": "ahead", "behind_by": 0}
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
    later_author_review = _status_review(
        review_id=12346,
        reviewer_id=1,
        state="DISMISSED",
    )
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


def test_independent_review_status_rejects_superseded_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = _status_review(review_id=12345)
    later_blocking = _status_review(review_id=12346, state="CHANGES_REQUESTED")
    later_blocking["html_url"] = (
        "https://github.com/owner/repo/pull/900#pullrequestreview-12346"
    )
    object_api, array_api = _review_inventory_api(reviews=[valid, later_blocking])
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


def test_independent_review_status_rejects_later_non_authorizing_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = _status_review(review_id=12345)
    later_blocking = _status_review(
        review_id=12346,
        body=_source_review_body(g0="FAIL"),
    )
    later_blocking["html_url"] = (
        "https://github.com/owner/repo/pull/900#pullrequestreview-12346"
    )
    object_api, array_api = _review_inventory_api(reviews=[valid, later_blocking])
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


def test_independent_review_status_keeps_other_reviewers_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = _status_review(review_id=12345)
    later_blocking = _status_review(
        review_id=12346,
        reviewer_id=3,
        body=_source_review_body(g0="FAIL"),
    )
    later_blocking["html_url"] = (
        "https://github.com/owner/repo/pull/900#pullrequestreview-12346"
    )
    object_api, array_api = _review_inventory_api(reviews=[valid, later_blocking])
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
