from __future__ import annotations

import base64
import hashlib
import importlib
import zipfile
from pathlib import Path

import pytest

from trade_rl.artifacts import canonical_json_bytes, content_digest

actions = importlib.import_module("tools.ppo_4h_indicator_smoke_actions")
smoke = importlib.import_module("trade_rl.evaluation.ppo_4h_indicator_smoke")

REVIEWED_SHA = "a" * 40
TAG_OBJECT_SHA = "b" * 40
RUN_ID = 4242
RUN_ATTEMPT = 2
ARTIFACT_ID = 5151
ARTIFACT_SHA256 = "c" * 64
ATTESTATION_SHA256 = "d" * 64
RESPONSE_ID = "response-123"
MODEL = "gemini-model-version"
REVIEW_TAG = "review/ppo-4h-indicator-smoke-v1"
REVIEW_URL = "https://github.com/shuntatsu/trade_rl/pull/767#pullrequestreview-12345"
DESCENDANT_AUTHORITY_SHA = "9" * 40


def _identity() -> str:
    value = {
        "schema": "ppo_4h_gemini_review_identity_v1",
        "repository": "shuntatsu/trade_rl",
        "repository_id": actions.TRUSTED_REPOSITORY_ID,
        "pull_number": 767,
        "reviewed_code_sha": REVIEWED_SHA,
        "review_protocol": "ppo_4h_gemini_semantic_review_v2",
    }
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _attestation(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "ppo_4h_gemini_reviewer_run_v2",
        "review_protocol": "ppo_4h_gemini_semantic_review_v2",
        "review_identity_digest": _identity(),
        "repository": "shuntatsu/trade_rl",
        "repository_id": actions.TRUSTED_REPOSITORY_ID,
        "reviewed_code_sha": REVIEWED_SHA,
        "review_tag": REVIEW_TAG,
        "review_tag_object_sha": TAG_OBJECT_SHA,
        "trusted_workflow_sha": actions.TRUSTED_REVIEWER_AUTHORITY_COMMIT,
        "trusted_workflow_ref": (
            "shuntatsu/trade_rl/.github/workflows/"
            "ppo-4h-gemini-review.yml@refs/heads/main"
        ),
        "trusted_workflow_file_sha256": (actions.TRUSTED_REVIEWER_WORKFLOW_SHA256),
        "trusted_runner_sha256": actions.TRUSTED_REVIEWER_RUNNER_SHA256,
        "reviewer_run_id": RUN_ID,
        "reviewer_run_attempt": RUN_ATTEMPT,
        "request_pull_number": 767,
        "request_comment_id": 31337,
        "requester_login": "shuntatsu",
        "requester_permission": "write",
        "request_body_sha256": "e" * 64,
        "trusted_verification_jobs": {
            "core": 101,
            "ppo-runtime": 102,
            "guide": 103,
        },
        "packet_sha256": "f" * 64,
        "system_instruction_sha256": "1" * 64,
        "gemini_request_sha256": "2" * 64,
        "raw_gemini_response_sha256": "3" * 64,
        "result_blind": True,
        "reviewer_context": "trusted_default_branch_read_only",
        "python_version": "3.12.12",
        "reviewer_provider": "google_gemini",
        "reviewer_model": MODEL,
        "reviewer_response_id": RESPONSE_ID,
        "g0": "PASS",
        "g1": "PASS",
        "g2": "EVIDENCE_BOUND",
        "disposition": "G0_G1_CLEAR_G2_EVIDENCE_BOUND",
        "blocking_findings": [],
        "strongest_counterexample": "counterexample",
        "missing_evidence": [],
        "claim_downgrade": "development smoke only",
        "machine_oracles": ["oracle"],
        "what_this_cannot_prove": ["profitability"],
    }
    value.update(updates)
    return value


def _source(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "ppo_4h_indicator_source_review_v4",
        "reviewed_code_sha": REVIEWED_SHA,
        "static_contract_digest": content_digest(smoke.static_protocol_contract()),
        "review_tag": REVIEW_TAG,
        "review_tag_object_sha": TAG_OBJECT_SHA,
        "review_identity_digest": _identity(),
        "trusted_reviewer_run_id": RUN_ID,
        "trusted_reviewer_run_attempt": RUN_ATTEMPT,
        "trusted_reviewer_artifact_id": ARTIFACT_ID,
        "trusted_reviewer_artifact_sha256": ARTIFACT_SHA256,
        "trusted_reviewer_attestation_sha256": ATTESTATION_SHA256,
        "reviewer_independence": "ESTABLISHED",
        "reviewer_kind": "external_ai",
        "reviewer_provider": "google_gemini",
        "reviewer_model": MODEL,
        "reviewer_response_id": RESPONSE_ID,
        "reviewer_context": "trusted_default_branch_read_only",
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
    value.update(updates)
    return value


def _review_body(**updates: object) -> str:
    value = _source(**updates)
    return (
        "<!-- ppo-4h-indicator-source-review-v4 -->\n"
        + canonical_json_bytes(value).decode("utf-8")
        + "\n"
    )


def _run(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": RUN_ID,
        "name": "PPO 4h Gemini Review",
        "event": "issue_comment",
        "status": "completed",
        "conclusion": "success",
        "run_attempt": RUN_ATTEMPT,
        "head_branch": "main",
        "head_sha": actions.TRUSTED_REVIEWER_AUTHORITY_COMMIT,
        "path": ".github/workflows/ppo-4h-gemini-review.yml",
    }
    value.update(updates)
    return value


def _jobs(**updates: object) -> list[dict[str, object]]:
    jobs = [
        {
            "id": 101,
            "name": "Trusted Lean Core verification",
            "status": "completed",
            "conclusion": "success",
        },
        {
            "id": 102,
            "name": "Trusted PPO Runtime verification",
            "status": "completed",
            "conclusion": "success",
        },
        {
            "id": 103,
            "name": "Trusted Human Guide verification",
            "status": "completed",
            "conclusion": "success",
        },
        {
            "id": 104,
            "name": "Trusted Gemini semantic review",
            "status": "completed",
            "conclusion": "success",
        },
    ]
    for item in jobs:
        if item["name"] in updates:
            item.update(updates[item["name"]])
    return jobs


def test_trusted_reviewer_authority_is_frozen_to_merged_main_identity() -> None:
    assert actions.TRUSTED_REVIEWER_AUTHORITY_COMMIT == (
        "f47d2bf99fdfd027a4f20ab43acdfacb598eff23"
    )
    assert actions.TRUSTED_REVIEWER_WORKFLOW_SHA256 == (
        "3a752a14ff9aebd754754529e3cb0fa781d837e5375cb19f610ce1e2057019b8"
    )
    assert actions.TRUSTED_REVIEWER_RUNNER_SHA256 == (
        "e98589b35062e3c4840f3b589f28c7d6982351d4063557a22e32a915b5eaa9c7"
    )
    assert actions.SOURCE_REVIEW_SCHEMA == "ppo_4h_indicator_source_review_v4"
    assert actions.REVIEW_SCHEMA == "ppo_4h_indicator_smoke_review_v3"
    assert (
        actions.TRUSTED_REVIEWER_ATTESTATION_SCHEMA == "ppo_4h_gemini_reviewer_run_v2"
    )
    assert actions.TRUSTED_REVIEW_PROTOCOL == "ppo_4h_gemini_semantic_review_v2"


def test_checked_out_trusted_reviewer_bytes_match_frozen_authority() -> None:
    # A main synchronization can silently update a trusted workflow action pin.
    # Such a change must fail locally until a separately reviewed new seal is pinned.
    root = Path(__file__).resolve().parents[2]
    for relative, expected in (
        (
            ".github/workflows/ppo-4h-gemini-review.yml",
            actions.TRUSTED_REVIEWER_WORKFLOW_SHA256,
        ),
        (
            "tools/ppo_4h_gemini_review.py",
            actions.TRUSTED_REVIEWER_RUNNER_SHA256,
        ),
    ):
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert actual == expected, f"trusted reviewer authority drift: {relative}"


def test_trusted_attestation_accepts_verified_authority_descendant() -> None:
    result = actions.validate_trusted_reviewer_attestation(
        _attestation(trusted_workflow_sha=DESCENDANT_AUTHORITY_SHA),
        _source(),
        repository="shuntatsu/trade_rl",
        pull_number=767,
        reviewed_code_sha=REVIEWED_SHA,
        run=_run(head_sha=DESCENDANT_AUTHORITY_SHA),
        jobs=_jobs(),
    )

    assert result["trusted_workflow_sha"] == DESCENDANT_AUTHORITY_SHA


def _content_record(raw: bytes) -> dict[str, object]:
    return {
        "type": "file",
        "encoding": "base64",
        "content": base64.b64encode(raw).decode("ascii"),
    }


SEAL_OBJECT_SHA = "7" * 40
BOOTSTRAP_REVIEW_HEAD = "4" * 40
BOOTSTRAP_FULL_CI_RUN = 4747
BOOTSTRAP_REVIEW_PR = 1185
SEAL_REVIEW_COMMENT_ID = 5757


def _bootstrap_note_body() -> str:
    return (
        "Disposition: **APPROVED for authority sealing only**\n"
        f"{BOOTSTRAP_REVIEW_HEAD} "
        f"{actions.TRUSTED_REVIEWER_AUTHORITY_COMMIT} "
        f"{actions.TRUSTED_REVIEWER_WORKFLOW_SHA256} "
        f"{actions.TRUSTED_REVIEWER_RUNNER_SHA256}"
    )


def _seal_ref(
    *, ref_sha: str = SEAL_OBJECT_SHA, kind: str = "tag"
) -> dict[str, object]:
    return {
        "ref": f"refs/tags/{actions.TRUSTED_REVIEWER_AUTHORITY_SEAL}",
        "object": {"type": kind, "sha": ref_sha},
    }


def _seal_object(
    *,
    target: str | None = None,
    workflow_digest: str | None = None,
    bootstrap_note: str | None = None,
) -> dict[str, object]:
    anchor = actions.TRUSTED_REVIEWER_AUTHORITY_COMMIT
    return {
        "tag": actions.TRUSTED_REVIEWER_AUTHORITY_SEAL,
        "sha": SEAL_OBJECT_SHA,
        "object": {"type": "commit", "sha": anchor if target is None else target},
        "message": (
            f"authority_commit={anchor}\n"
            f"workflow_sha256={actions.TRUSTED_REVIEWER_WORKFLOW_SHA256 if workflow_digest is None else workflow_digest}\n"
            f"runner_sha256={actions.TRUSTED_REVIEWER_RUNNER_SHA256}\n"
            + (
                f"bootstrap_review_pr={BOOTSTRAP_REVIEW_PR}\n"
                f"bootstrap_review_head={BOOTSTRAP_REVIEW_HEAD}\n"
                f"bootstrap_full_ci_run={BOOTSTRAP_FULL_CI_RUN}\n"
                f"seal_review_comment={SEAL_REVIEW_COMMENT_ID}\n"
                f"seal_review_comment_sha256={hashlib.sha256(_bootstrap_note_body().encode('utf-8')).hexdigest()}\n"
                f"review_protocol={actions.TRUSTED_REVIEW_PROTOCOL}\n"
                f"attestation_schema={actions.TRUSTED_REVIEWER_ATTESTATION_SCHEMA}\n"
                if bootstrap_note is None
                else bootstrap_note
            )
        ),
    }


def _seal_bootstrap_response(url: str) -> dict[str, object]:
    if url.endswith(f"/pulls/{BOOTSTRAP_REVIEW_PR}"):
        return {
            "number": BOOTSTRAP_REVIEW_PR,
            "merged_at": "2026-11-09T00:00:00Z",
            "head": {
                "sha": BOOTSTRAP_REVIEW_HEAD,
                "repo": {"id": actions.TRUSTED_REPOSITORY_ID},
            },
            "base": {"ref": "main", "repo": {"id": actions.TRUSTED_REPOSITORY_ID}},
        }
    if url.endswith(f"/actions/runs/{BOOTSTRAP_FULL_CI_RUN}"):
        return {
            "id": BOOTSTRAP_FULL_CI_RUN,
            "head_sha": BOOTSTRAP_REVIEW_HEAD,
            "status": "completed",
            "conclusion": "success",
            "event": "pull_request",
            "name": "CI",
            "path": ".github/workflows/ci.yml",
            "run_attempt": 1,
            "repository": {"id": actions.TRUSTED_REPOSITORY_ID},
        }
    if url.endswith(
        f"/actions/runs/{BOOTSTRAP_FULL_CI_RUN}/attempts/1/jobs?per_page=100"
    ):
        names = ("Lean Core", "PPO Runtime", "Human Guide", "Market Store PostgreSQL")
        jobs = [
            {
                "name": name,
                "status": "completed",
                "conclusion": "success",
                "head_sha": BOOTSTRAP_REVIEW_HEAD,
                "run_id": BOOTSTRAP_FULL_CI_RUN,
                "run_attempt": 1,
            }
            for name in names
        ]
        return {"total_count": len(jobs), "jobs": jobs}
    if url.endswith(f"/issues/comments/{SEAL_REVIEW_COMMENT_ID}"):
        return {
            "id": SEAL_REVIEW_COMMENT_ID,
            "issue_url": f"https://api.github.com/repos/shuntatsu/trade_rl/issues/{BOOTSTRAP_REVIEW_PR}",
            "body": _bootstrap_note_body(),
        }
    if url.endswith(
        f"/compare/{BOOTSTRAP_REVIEW_HEAD}..."
        f"{actions.TRUSTED_REVIEWER_AUTHORITY_COMMIT}"
    ):
        return {"status": "ahead", "behind_by": 0}
    raise AssertionError(url)


@pytest.mark.parametrize(
    ("seal_ref", "seal_object", "expected_error"),
    [
        (None, None, "seal"),
        (_seal_ref(kind="commit"), _seal_object(), "annotated"),
        (_seal_ref(), _seal_object(target="0" * 40), "target"),
        (_seal_ref(), _seal_object(workflow_digest="0" * 64), "workflow"),
    ],
)
def test_frozen_reviewer_seal_fail_closed_on_missing_lightweight_or_drift(
    monkeypatch: pytest.MonkeyPatch,
    seal_ref: dict[str, object] | None,
    seal_object: dict[str, object] | None,
    expected_error: str,
) -> None:
    def api(url: str, **_kwargs: object) -> dict[str, object]:
        if "/compare/" in url:
            return {"status": "ahead", "behind_by": 0}
        if "/git/ref/tags/" in url:
            if seal_ref is None:
                raise ValueError("trusted authority seal missing")
            return seal_ref
        if "/git/tags/" in url:
            assert seal_object is not None
            return seal_object
        raise AssertionError(url)

    monkeypatch.setattr(actions.transport, "_api_json", api)
    with pytest.raises(ValueError, match=expected_error):
        actions._require_frozen_reviewer_authority(
            repository="shuntatsu/trade_rl",
            run_head_sha=DESCENDANT_AUTHORITY_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_frozen_authority_rejects_unlinked_bootstrap_review_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def api(url: str, **_kwargs: object) -> dict[str, object]:
        if "/compare/" in url:
            return {"status": "ahead", "behind_by": 0}
        if "/git/ref/tags/" in url:
            return _seal_ref()
        if "/git/tags/" in url:
            return _seal_object(bootstrap_note="")
        return _seal_bootstrap_response(url)

    monkeypatch.setattr(actions.transport, "_api_json", api)
    with pytest.raises(ValueError, match="bootstrap"):
        actions._require_frozen_reviewer_authority_seal(
            repo_path="shuntatsu/trade_rl",
            token="token",
            deadline=999999999.0,
        )


@pytest.mark.parametrize(
    ("case", "failure"),
    [
        ("wrong_pr_head", "PR identity"),
        ("wrong_pr_repo", "PR identity"),
        ("unmerged_pr", "PR identity"),
        ("unrelated_review_head", "anchor history"),
        ("wrong_ci_head", "Full CI"),
        ("failed_ci", "Full CI"),
        ("wrong_ci_repo", "Full CI"),
        ("wrong_note", "review note"),
        ("edited_note", "review note"),
        ("wrong_note_issue", "review note"),
    ],
)
def test_frozen_authority_rejects_unbound_bootstrap_reference(
    monkeypatch: pytest.MonkeyPatch, case: str, failure: str
) -> None:
    def api(url: str, **_kwargs: object) -> dict[str, object]:
        if "/git/ref/tags/" in url:
            return _seal_ref()
        if "/git/tags/" in url:
            return _seal_object()
        value = _seal_bootstrap_response(url)
        if case == "wrong_pr_head" and "/pulls/" in url:
            value["head"]["sha"] = "0" * 40
        if case == "wrong_pr_repo" and "/pulls/" in url:
            value["head"]["repo"]["id"] = 123
        if case == "unmerged_pr" and "/pulls/" in url:
            value["merged_at"] = None
        if case == "unrelated_review_head" and "/compare/" in url:
            value["behind_by"] = 1
        if case == "wrong_ci_head" and "/actions/runs/" in url:
            value["head_sha"] = "0" * 40
        if case == "failed_ci" and "/actions/runs/" in url:
            value["conclusion"] = "failure"
        if case == "wrong_ci_repo" and "/actions/runs/" in url:
            value["repository"]["id"] = 123
        if case == "wrong_note" and "/issues/comments/" in url:
            value["body"] = "APPROVED for authority sealing only"
        if case == "edited_note" and "/issues/comments/" in url:
            value["body"] = _bootstrap_note_body() + " (edited)"
        if case == "wrong_note_issue" and "/issues/comments/" in url:
            value["issue_url"] = "https://api.github.com/repos/owner/repo/issues/1"
        return value

    monkeypatch.setattr(actions.transport, "_api_json", api)
    with pytest.raises(ValueError, match=failure):
        actions._require_frozen_reviewer_authority_seal(
            repo_path="shuntatsu/trade_rl",
            token="token",
            deadline=999999999.0,
        )


def test_frozen_authority_rejects_bootstrap_ci_with_skipped_required_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def api(url: str, **_kwargs: object) -> dict[str, object]:
        if "/git/ref/tags/" in url:
            return _seal_ref()
        if "/git/tags/" in url:
            return _seal_object()
        response = _seal_bootstrap_response(url)
        if "/attempts/1/jobs?" in url:
            response["jobs"][0]["conclusion"] = "skipped"
        return response

    monkeypatch.setattr(actions.transport, "_api_json", api)
    with pytest.raises(ValueError, match="bootstrap.*job"):
        actions._require_frozen_reviewer_authority_seal(
            repo_path="shuntatsu/trade_rl",
            token="token",
            deadline=999999999.0,
        )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("missing_core", "verification job missing"),
        ("duplicate_core", "verification job invalid"),
        ("wrong_run", "verification job invalid"),
        ("wrong_attempt", "verification job invalid"),
        ("wrong_head", "verification job invalid"),
        ("pending_core", "verification job invalid"),
        ("truncated_inventory", "inventory incomplete"),
        ("bad_ci_attempt", "Full CI attempt"),
        ("bool_ci_attempt", "Full CI attempt"),
        ("malformed_job_name", "CI job name malformed"),
    ],
)
def test_bootstrap_ci_rejects_missing_duplicate_stale_or_partial_jobs(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected: str,
) -> None:
    def api(url: str, **_kwargs: object) -> dict[str, object]:
        if "/git/ref/tags/" in url:
            return _seal_ref()
        if "/git/tags/" in url:
            return _seal_object()
        value = _seal_bootstrap_response(url)
        if case == "bad_ci_attempt" and url.endswith(
            f"/actions/runs/{BOOTSTRAP_FULL_CI_RUN}"
        ):
            value["run_attempt"] = None
        if case == "bool_ci_attempt" and url.endswith(
            f"/actions/runs/{BOOTSTRAP_FULL_CI_RUN}"
        ):
            value["run_attempt"] = True
        if "/attempts/1/jobs?" in url:
            jobs = value["jobs"]
            if case == "missing_core":
                jobs.pop(0)
            elif case == "duplicate_core":
                jobs.append(dict(jobs[0]))
            elif case == "wrong_run":
                jobs[0]["run_id"] = 123
            elif case == "wrong_attempt":
                jobs[0]["run_attempt"] = 2
            elif case == "wrong_head":
                jobs[0]["head_sha"] = "0" * 40
            elif case == "pending_core":
                jobs[0]["status"] = "in_progress"
                jobs[0]["conclusion"] = None
            elif case == "malformed_job_name":
                jobs[0]["name"] = None
            elif case == "truncated_inventory":
                value["total_count"] += 1
            if case in {"missing_core", "duplicate_core"}:
                value["total_count"] = len(jobs)
        return value

    monkeypatch.setattr(actions.transport, "_api_json", api)
    with pytest.raises(ValueError, match=expected):
        actions._require_frozen_reviewer_authority_seal(
            repo_path="shuntatsu/trade_rl",
            token="token",
            deadline=999999999.0,
        )


def test_frozen_authority_rejects_seal_ref_mutation_during_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = 0

    def api(url: str, **_kwargs: object) -> dict[str, object]:
        nonlocal reads
        if "/compare/" in url:
            return {"status": "ahead", "behind_by": 0}
        if "/git/ref/tags/" in url:
            reads += 1
            return _seal_ref(ref_sha=SEAL_OBJECT_SHA if reads == 1 else "8" * 40)
        if "/git/tags/" in url:
            return _seal_object()
        return _seal_bootstrap_response(url)

    monkeypatch.setattr(actions.transport, "_api_json", api)
    with pytest.raises(ValueError, match="changed during verification"):
        actions._require_frozen_reviewer_authority(
            repository="shuntatsu/trade_rl",
            run_head_sha=DESCENDANT_AUTHORITY_SHA,
            token="token",
            deadline=999999999.0,
        )
    assert reads == 2


def test_frozen_authority_detects_seal_ref_change_during_bootstrap_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No tag-ref stability claim if PR/CI/comment fetches can complete after
    # a ref is retargeted and the validator never refetches its final SHA.
    state = {"bootstrap_done": False, "ref_reads": 0}

    def api(url: str, **_kwargs: object) -> dict[str, object]:
        if "/git/ref/tags/" in url:
            state["ref_reads"] += 1
            sha = "8" * 40 if state["bootstrap_done"] else SEAL_OBJECT_SHA
            return _seal_ref(ref_sha=sha)
        if "/git/tags/" in url:
            return _seal_object()
        response = _seal_bootstrap_response(url)
        if "/issues/comments/" in url:
            state["bootstrap_done"] = True
        return response

    monkeypatch.setattr(actions.transport, "_api_json", api)
    with pytest.raises(ValueError, match="changed during verification"):
        actions._require_frozen_reviewer_authority_seal(
            repo_path="shuntatsu/trade_rl",
            token="token",
            deadline=999999999.0,
        )
    assert state["ref_reads"] == 2
    assert state["bootstrap_done"]


def test_frozen_authority_accepts_descendant_with_independently_verified_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = b"name: trusted-reviewer\n"
    runner = b"print('trusted reviewer')\n"
    monkeypatch.setattr(
        actions,
        "TRUSTED_REVIEWER_WORKFLOW_SHA256",
        hashlib.sha256(workflow).hexdigest(),
    )
    monkeypatch.setattr(
        actions,
        "TRUSTED_REVIEWER_RUNNER_SHA256",
        hashlib.sha256(runner).hexdigest(),
    )

    def api(url: str, **_kwargs: object) -> dict[str, object]:
        if "/compare/" in url:
            return {"status": "ahead", "behind_by": 0}
        if "/git/ref/tags/" in url:
            return _seal_ref()
        if "/git/tags/" in url:
            return _seal_object()
        if "/contents/.github/workflows/ppo-4h-gemini-review.yml?" in url:
            return _content_record(workflow)
        if "/contents/tools/ppo_4h_gemini_review.py?" in url:
            return _content_record(runner)
        return _seal_bootstrap_response(url)

    monkeypatch.setattr(actions.transport, "_api_json", api)

    actions._require_frozen_reviewer_authority(
        repository="shuntatsu/trade_rl",
        run_head_sha=DESCENDANT_AUTHORITY_SHA,
        token="token",
        deadline=999999999.0,
    )


def test_frozen_authority_rejects_non_descendant_or_changed_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = b"name: trusted-reviewer\n"
    runner = b"print('trusted reviewer')\n"
    monkeypatch.setattr(
        actions,
        "TRUSTED_REVIEWER_WORKFLOW_SHA256",
        hashlib.sha256(workflow).hexdigest(),
    )
    monkeypatch.setattr(
        actions,
        "TRUSTED_REVIEWER_RUNNER_SHA256",
        hashlib.sha256(runner).hexdigest(),
    )

    def non_descendant(url: str, **_kwargs: object) -> dict[str, object]:
        if "/compare/" in url:
            return {"status": "diverged", "behind_by": 1}
        raise AssertionError(url)

    monkeypatch.setattr(actions.transport, "_api_json", non_descendant)
    with pytest.raises(ValueError, match="descendant"):
        actions._require_frozen_reviewer_authority(
            repository="shuntatsu/trade_rl",
            run_head_sha=DESCENDANT_AUTHORITY_SHA,
            token="token",
            deadline=999999999.0,
        )

    def changed_runner(url: str, **_kwargs: object) -> dict[str, object]:
        if "/compare/" in url:
            return {"status": "ahead", "behind_by": 0}
        if "/git/ref/tags/" in url:
            return _seal_ref()
        if "/git/tags/" in url:
            return _seal_object()
        if "/contents/.github/workflows/ppo-4h-gemini-review.yml?" in url:
            return _content_record(workflow)
        if "/contents/tools/ppo_4h_gemini_review.py?" in url:
            return _content_record(b"changed runner\n")
        return _seal_bootstrap_response(url)

    monkeypatch.setattr(actions.transport, "_api_json", changed_runner)
    with pytest.raises(ValueError, match="runner"):
        actions._require_frozen_reviewer_authority(
            repository="shuntatsu/trade_rl",
            run_head_sha=DESCENDANT_AUTHORITY_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_trusted_attestation_accepts_same_account_transport_and_free_model_provenance() -> (
    None
):
    result = actions.validate_trusted_reviewer_attestation(
        _attestation(reviewer_model="gemini-future-version"),
        _source(reviewer_model="gemini-future-version"),
        repository="shuntatsu/trade_rl",
        pull_number=767,
        reviewed_code_sha=REVIEWED_SHA,
        run=_run(),
        jobs=_jobs(),
    )

    assert result["reviewer_provider"] == "google_gemini"
    assert result["reviewer_model"] == "gemini-future-version"


@pytest.mark.parametrize(
    ("attestation_updates", "run_updates", "job_name"),
    (
        ({"trusted_workflow_file_sha256": "9" * 64}, {}, None),
        ({"trusted_runner_sha256": "9" * 64}, {}, None),
        ({}, {"path": ".github/workflows/evil.yml"}, None),
        ({}, {"event": "workflow_dispatch"}, None),
        ({}, {}, "Trusted PPO Runtime verification"),
    ),
)
def test_trusted_attestation_rejects_authority_or_verification_drift(
    attestation_updates: dict[str, object],
    run_updates: dict[str, object],
    job_name: str | None,
) -> None:
    jobs = _jobs()
    if job_name is not None:
        for job in jobs:
            if job["name"] == job_name:
                job["conclusion"] = "failure"
    with pytest.raises(ValueError):
        actions.validate_trusted_reviewer_attestation(
            _attestation(**attestation_updates),
            _source(),
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
            run=_run(**run_updates),
            jobs=jobs,
        )


def test_source_review_same_principal_is_semantically_valid_for_trusted_transport() -> (
    None
):
    body = _review_body()
    record = {
        "id": 12345,
        "html_url": REVIEW_URL,
        "body": body,
        "commit_id": REVIEWED_SHA,
        "user": {"id": 1, "login": "shuntatsu"},
        "state": "COMMENTED",
    }
    pull = {
        "number": 767,
        "state": "open",
        "draft": True,
        "user": {"id": 1, "login": "shuntatsu"},
        "head": {
            "ref": actions.EXECUTION_BRANCH,
            "sha": REVIEWED_SHA,
            "repo": {"full_name": "shuntatsu/trade_rl"},
        },
        "base": {"ref": "main"},
    }

    source = actions._validate_source_review_record(
        record,
        pull,
        repository="shuntatsu/trade_rl",
        pull_number=767,
        reviewed_code_sha=REVIEWED_SHA,
        expected_static_digest=content_digest(smoke.static_protocol_contract()),
    )

    assert source["reviewer_independence"] == "ESTABLISHED"
    assert source["reviewer_provider"] == "google_gemini"


def test_trusted_attestation_is_separate_mandatory_authority() -> None:
    with pytest.raises(ValueError, match="token"):
        actions._require_trusted_reviewer_attestation(
            _source(),
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
        )


def _packet_artifact(
    *,
    run_id: int = RUN_ID,
    attempt: int = RUN_ATTEMPT,
    expired: bool = False,
) -> dict[str, object]:
    return {
        "name": f"ppo-4h-review-packet-{_identity()}-{run_id}-{attempt}",
        "expired": expired,
        "workflow_run": {"id": run_id},
    }


def _result_artifact(
    *,
    run_id: int = RUN_ID,
    attempt: int = RUN_ATTEMPT,
    artifact_id: int = ARTIFACT_ID,
    expired: bool = False,
) -> dict[str, object]:
    return {
        "id": artifact_id,
        "name": f"ppo-4h-gemini-review-{run_id}-{attempt}",
        "expired": expired,
        "workflow_run": {"id": run_id},
    }


def _fake_trusted_evidence_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    attestation: dict[str, object],
    artifact_name: str | None = None,
    run: dict[str, object] | None = None,
    jobs: list[dict[str, object]] | None = None,
    lineage_artifacts: list[dict[str, object]] | None = None,
    result_artifacts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    workflow = b"name: trusted-reviewer\n"
    runner = b"print('trusted reviewer')\n"
    monkeypatch.setattr(
        actions,
        "TRUSTED_REVIEWER_WORKFLOW_SHA256",
        hashlib.sha256(workflow).hexdigest(),
    )
    monkeypatch.setattr(
        actions,
        "TRUSTED_REVIEWER_RUNNER_SHA256",
        hashlib.sha256(runner).hexdigest(),
    )
    attestation = dict(
        attestation,
        trusted_workflow_file_sha256=actions.TRUSTED_REVIEWER_WORKFLOW_SHA256,
        trusted_runner_sha256=actions.TRUSTED_REVIEWER_RUNNER_SHA256,
    )
    attestation_raw = canonical_json_bytes(attestation)
    source = _source(
        trusted_reviewer_attestation_sha256=hashlib.sha256(attestation_raw).hexdigest()
    )
    run_value = _run() if run is None else run
    jobs_value = _jobs() if jobs is None else jobs

    def api(url: str, **_kwargs: object) -> dict[str, object]:
        if url.endswith(f"/actions/runs/{RUN_ID}"):
            return run_value
        if "/compare/" in url:
            if run_value.get("head_sha") == actions.TRUSTED_REVIEWER_AUTHORITY_COMMIT:
                return {"status": "identical", "behind_by": 0}
            return {"status": "diverged", "behind_by": 1}
        if "/git/ref/tags/" in url:
            return _seal_ref()
        if "/git/tags/" in url:
            return _seal_object()
        if "/contents/.github/workflows/ppo-4h-gemini-review.yml?" in url:
            return _content_record(workflow)
        if "/contents/tools/ppo_4h_gemini_review.py?" in url:
            return _content_record(runner)
        if url.endswith(
            f"/actions/runs/{RUN_ID}/attempts/{RUN_ATTEMPT}/jobs?per_page=100"
        ):
            return {"jobs": jobs_value}
        if url.endswith(f"/actions/runs/{RUN_ID}/artifacts?per_page=100&page=1"):
            return {
                "artifacts": (
                    [_result_artifact()]
                    if result_artifacts is None
                    else result_artifacts
                )
            }
        if "/actions/artifacts?per_page=100&page=1" in url:
            return {
                "artifacts": (
                    [_packet_artifact()]
                    if lineage_artifacts is None
                    else lineage_artifacts
                )
            }
        return _seal_bootstrap_response(url)

    def download(
        repository: str,
        reference: object,
        *,
        repository_id: int,
        token: str,
        destination: object,
        deadline: float,
    ) -> dict[str, object]:
        assert repository == "shuntatsu/trade_rl"
        assert repository_id == actions.TRUSTED_REPOSITORY_ID
        assert token == "token"
        assert deadline == 999999999.0
        assert reference.artifact_id == ARTIFACT_ID
        assert reference.run_id == RUN_ID
        assert reference.sha256 == ARTIFACT_SHA256
        path = Path(destination)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("reviewer-attestation.json", attestation_raw)
            archive.writestr("review-packet.json", b"{}")
            archive.writestr("gemini-request.json", b"{}")
            archive.writestr("gemini-response.json", b"{}")
            archive.writestr(
                "reviewer-disposition.txt",
                b"G0_G1_CLEAR_G2_EVIDENCE_BOUND\n",
            )
        return {
            "id": ARTIFACT_ID,
            "name": (
                f"ppo-4h-gemini-review-{RUN_ID}-{RUN_ATTEMPT}"
                if artifact_name is None
                else artifact_name
            ),
            "expired": False,
            "workflow_run": {
                "id": RUN_ID,
                "repository_id": actions.TRUSTED_REPOSITORY_ID,
                "head_repository_id": actions.TRUSTED_REPOSITORY_ID,
            },
            "digest": f"sha256:{ARTIFACT_SHA256}",
        }

    monkeypatch.setattr(actions.transport, "_api_json", api)
    monkeypatch.setattr(actions.transport, "download_artifact_archive", download)
    return source


def test_require_trusted_attestation_refetches_run_jobs_and_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attestation = _attestation()
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=attestation,
    )

    result = actions._require_trusted_reviewer_attestation(
        source,
        repository="shuntatsu/trade_rl",
        pull_number=767,
        reviewed_code_sha=REVIEWED_SHA,
        token="token",
        deadline=999999999.0,
    )

    assert result == {
        **attestation,
        "trusted_workflow_file_sha256": hashlib.sha256(
            b"name: trusted-reviewer\n"
        ).hexdigest(),
        "trusted_runner_sha256": hashlib.sha256(
            b"print('trusted reviewer')\n"
        ).hexdigest(),
    }


def test_require_trusted_attestation_rejects_wrong_artifact_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=_attestation(),
        artifact_name="forged",
    )
    with pytest.raises(ValueError, match="artifact name"):
        actions._require_trusted_reviewer_attestation(
            source,
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_require_trusted_attestation_rejects_attestation_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_trusted_evidence_transport(monkeypatch, attestation=_attestation())
    source = _source(trusted_reviewer_attestation_sha256="9" * 64)
    with pytest.raises(ValueError, match="attestation SHA-256"):
        actions._require_trusted_reviewer_attestation(
            source,
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_require_trusted_attestation_rejects_non_descendant_workflow_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=_attestation(),
        run=_run(head_sha="9" * 40),
    )
    with pytest.raises(ValueError, match="descendant"):
        actions._require_trusted_reviewer_attestation(
            source,
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_require_trusted_attestation_rejects_second_run_for_same_review_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=_attestation(),
        lineage_artifacts=[
            _packet_artifact(),
            _packet_artifact(run_id=7777, attempt=1),
        ],
    )

    with pytest.raises(ValueError, match="review lineage"):
        actions._require_trusted_reviewer_attestation(
            source,
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_require_trusted_attestation_rejects_expired_matching_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=_attestation(),
        lineage_artifacts=[_packet_artifact(expired=True)],
    )

    with pytest.raises(ValueError, match="review lineage"):
        actions._require_trusted_reviewer_attestation(
            source,
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_require_trusted_attestation_allows_prior_attempt_in_same_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=_attestation(),
        lineage_artifacts=[
            _packet_artifact(attempt=1),
            _packet_artifact(),
        ],
    )

    result = actions._require_trusted_reviewer_attestation(
        source,
        repository="shuntatsu/trade_rl",
        pull_number=767,
        reviewed_code_sha=REVIEWED_SHA,
        token="token",
        deadline=999999999.0,
    )

    assert result["reviewer_run_attempt"] == RUN_ATTEMPT


@pytest.mark.parametrize(
    "lineage_artifacts",
    (
        [_packet_artifact(attempt=1)],
        [_packet_artifact(), _packet_artifact(attempt=3)],
        [
            {
                "name": f"ppo-4h-review-packet-{_identity()}-{RUN_ID}-{RUN_ATTEMPT}",
                "workflow_run": {"id": RUN_ID},
            }
        ],
    ),
)
def test_require_trusted_attestation_rejects_incomplete_or_future_lineage(
    monkeypatch: pytest.MonkeyPatch,
    lineage_artifacts: list[dict[str, object]],
) -> None:
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=_attestation(),
        lineage_artifacts=lineage_artifacts,
    )

    with pytest.raises(ValueError, match="review lineage"):
        actions._require_trusted_reviewer_attestation(
            source,
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_require_trusted_attestation_rejects_prior_terminal_result_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=_attestation(),
        result_artifacts=[
            _result_artifact(attempt=1, artifact_id=6161),
            _result_artifact(),
        ],
    )

    with pytest.raises(ValueError, match="terminal"):
        actions._require_trusted_reviewer_attestation(
            source,
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_require_trusted_attestation_rejects_missing_selected_result_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=_attestation(),
        result_artifacts=[],
    )

    with pytest.raises(ValueError, match="selected terminal"):
        actions._require_trusted_reviewer_attestation(
            source,
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_require_trusted_attestation_rejects_selected_result_artifact_id_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=_attestation(),
        result_artifacts=[_result_artifact(artifact_id=6161)],
    )

    with pytest.raises(ValueError, match="artifact id"):
        actions._require_trusted_reviewer_attestation(
            source,
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_require_trusted_attestation_rejects_expired_terminal_result_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=_attestation(),
        result_artifacts=[_result_artifact(expired=True)],
    )

    with pytest.raises(ValueError, match="expired"):
        actions._require_trusted_reviewer_attestation(
            source,
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )


def test_require_trusted_attestation_rejects_duplicate_terminal_result_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=_attestation(),
        result_artifacts=[
            _result_artifact(),
            _result_artifact(artifact_id=6161),
        ],
    )

    with pytest.raises(ValueError, match="duplicate terminal"):
        actions._require_trusted_reviewer_attestation(
            source,
            repository="shuntatsu/trade_rl",
            pull_number=767,
            reviewed_code_sha=REVIEWED_SHA,
            token="token",
            deadline=999999999.0,
        )
