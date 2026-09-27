from __future__ import annotations

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


def _identity() -> str:
    value = {
        "schema": "ppo_4h_gemini_review_identity_v1",
        "repository": "shuntatsu/trade_rl",
        "repository_id": actions.TRUSTED_REPOSITORY_ID,
        "pull_number": 767,
        "reviewed_code_sha": REVIEWED_SHA,
        "review_protocol": "ppo_4h_gemini_semantic_review_v1",
    }
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _attestation(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "ppo_4h_gemini_reviewer_run_v1",
        "review_protocol": "ppo_4h_gemini_semantic_review_v1",
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
        "7d1f8f687b31f83f64a91bffa512804f9b6c4402"
    )
    assert actions.TRUSTED_REVIEWER_WORKFLOW_SHA256 == (
        "98318fa4f4dafecc1d0c1401d55e7abecdfa11d33cf2367ad1701623eb94ab17"
    )
    assert actions.TRUSTED_REVIEWER_RUNNER_SHA256 == (
        "d5951f8a500c686fe8647bdd07ea13ef41f67454e19957fc120ca93ee40ebeab"
    )
    assert actions.SOURCE_REVIEW_SCHEMA == "ppo_4h_indicator_source_review_v4"
    assert actions.REVIEW_SCHEMA == "ppo_4h_indicator_smoke_review_v3"


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


def _fake_trusted_evidence_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    attestation: dict[str, object],
    artifact_name: str | None = None,
    run: dict[str, object] | None = None,
    jobs: list[dict[str, object]] | None = None,
    lineage_artifacts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    attestation_raw = canonical_json_bytes(attestation)
    source = _source(
        trusted_reviewer_attestation_sha256=hashlib.sha256(attestation_raw).hexdigest()
    )
    run_value = _run() if run is None else run
    jobs_value = _jobs() if jobs is None else jobs

    def api(url: str, **_kwargs: object) -> dict[str, object]:
        if url.endswith(f"/actions/runs/{RUN_ID}"):
            return run_value
        if url.endswith(
            f"/actions/runs/{RUN_ID}/attempts/{RUN_ATTEMPT}/jobs?per_page=100"
        ):
            return {"jobs": jobs_value}
        if "/actions/artifacts?per_page=100&page=1" in url:
            return {
                "artifacts": (
                    [_packet_artifact()]
                    if lineage_artifacts is None
                    else lineage_artifacts
                )
            }
        raise AssertionError(url)

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

    assert result == attestation


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


def test_require_trusted_attestation_rejects_unfrozen_workflow_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_trusted_evidence_transport(
        monkeypatch,
        attestation=_attestation(),
        run=_run(head_sha="9" * 40),
    )
    with pytest.raises(ValueError, match="workflow run identity"):
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
