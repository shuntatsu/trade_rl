from __future__ import annotations

import hashlib
import importlib
from copy import deepcopy

import pytest

from trade_rl.artifacts import content_digest

actions = importlib.import_module("tools.ppo_4h_indicator_smoke_actions")
smoke = importlib.import_module("trade_rl.evaluation.ppo_4h_indicator_smoke")


REVIEWED_SHA = "a" * 40
TRIGGER_SHA = "b" * 40
REVIEW_URL = "https://github.com/owner/repo/pull/758#issuecomment-12345"
REVIEW_BODY = (
    "<!-- hourly-agent-review -->\n"
    f"Exact HEAD: {REVIEWED_SHA}. Fresh result-blind review: "
    "G0 PASS, G1 PASS, G2 PASS.\n"
    "Disposition: G0_G1_CLEAR_G2_EVIDENCE_BOUND."
)


def _review() -> dict[str, object]:
    return {
        "schema": actions.REVIEW_SCHEMA,
        "reviewed_code_sha": REVIEWED_SHA,
        "static_contract_digest": content_digest(smoke.static_protocol_contract()),
        "source_review_url": REVIEW_URL,
        "source_review_body_sha256": hashlib.sha256(
            REVIEW_BODY.encode("utf-8")
        ).hexdigest(),
        "reviewer_surface": "hourly_agent_review_v1",
        "result_blind": True,
        "g0": "PASS",
        "g1": "PASS",
        "g2": "PASS",
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


def test_review_gate_binds_exact_parent_contract_and_review_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(
        actions.transport,
        "_api_json",
        lambda *_args, **_kwargs: {
            "html_url": REVIEW_URL,
            "body": REVIEW_BODY,
        },
    )

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
    monkeypatch.setattr(
        actions.transport,
        "_api_json",
        lambda *_args, **_kwargs: {
            "html_url": REVIEW_URL,
            "body": REVIEW_BODY,
        },
    )
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
        lambda *_args, **_kwargs: {
            "html_url": REVIEW_URL,
            "body": REVIEW_BODY + " edited",
        },
    )

    with pytest.raises(ValueError, match="comment"):
        actions.validate_review_gate(
            _review(),
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_rejects_non_hourly_reviewer_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(
        actions.transport,
        "_api_json",
        lambda *_args, **_kwargs: {
            "html_url": REVIEW_URL,
            "body": REVIEW_BODY.removeprefix("<!-- hourly-agent-review -->\n"),
        },
    )
    review = _review()
    review["source_review_body_sha256"] = hashlib.sha256(
        REVIEW_BODY.removeprefix("<!-- hourly-agent-review -->\n").encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="hourly-agent-review"):
        actions.validate_review_gate(
            review,
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_rejects_blocking_hourly_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = REVIEW_BODY + "\nBLOCKING before economic execution."
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(
        actions.transport,
        "_api_json",
        lambda *_args, **_kwargs: {
            "html_url": REVIEW_URL,
            "body": body,
        },
    )
    review = _review()
    review["source_review_body_sha256"] = hashlib.sha256(
        body.encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="authorize"):
        actions.validate_review_gate(
            review,
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )


def test_review_gate_rejects_hourly_review_that_disclaims_independence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = REVIEW_BODY + "\nReviewer independence: NOT ESTABLISHED."
    monkeypatch.setattr(actions, "_git", _fake_git)
    monkeypatch.setattr(
        actions.transport,
        "_api_json",
        lambda *_args, **_kwargs: {
            "html_url": REVIEW_URL,
            "body": body,
        },
    )
    review = _review()
    review["source_review_body_sha256"] = hashlib.sha256(
        body.encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="authorize"):
        actions.validate_review_gate(
            review,
            repository="owner/repo",
            token="token",
            deadline=999999999.0,
        )
