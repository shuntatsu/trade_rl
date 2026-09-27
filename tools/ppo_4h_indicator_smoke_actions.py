"""Authenticated one-shot transport for the 4h PPO indicator smoke."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any

from tools import ppo_checkpoint_actions as transport
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation import ppo_4h_indicator_smoke as smoke

SOURCE_ARTIFACT_ID = 10_331_899_302
SOURCE_ARTIFACT_RUN_ID = 34_803_217_815
SOURCE_ARTIFACT_SHA256 = (
    "89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce"
)
REVIEW_PATH = Path("report/ppo-4h-indicator-smoke-review.json")
REVIEW_SCHEMA = "ppo_4h_indicator_smoke_review_v3"
SOURCE_REVIEW_SCHEMA = "ppo_4h_indicator_source_review_v4"
SOURCE_REVIEW_MARKER = "<!-- ppo-4h-indicator-source-review-v4 -->\n"
REVIEWER_SURFACE = "github_pr_review_v3"
TRUSTED_REPOSITORY_ID = 1_103_009_698
TRUSTED_REVIEWER_AUTHORITY_COMMIT = "1147a97536f505162cb89217db35578d0437741e"
TRUSTED_REVIEWER_WORKFLOW_PATH = ".github/workflows/ppo-4h-gemini-review.yml"
TRUSTED_REVIEWER_WORKFLOW_NAME = "PPO 4h Gemini Review"
TRUSTED_REVIEWER_WORKFLOW_SHA256 = (
    "98318fa4f4dafecc1d0c1401d55e7abecdfa11d33cf2367ad1701623eb94ab17"
)
TRUSTED_REVIEWER_RUNNER_SHA256 = (
    "2cb17eb9c9fe292431255f85a3dbdc591344a498b788422ee4d2d478245dd335"
)
TRUSTED_REVIEWER_ATTESTATION_SCHEMA = "ppo_4h_gemini_reviewer_run_v1"
TRUSTED_REVIEW_PROTOCOL = "ppo_4h_gemini_semantic_review_v1"
TRUSTED_REVIEWER_PROVIDER = "google_gemini"
TRUSTED_REVIEWER_CONTEXT = "trusted_default_branch_read_only"
TRUSTED_REVIEWER_ARTIFACT_PREFIX = "ppo-4h-gemini-review"
TRUSTED_REVIEWER_ATTESTATION_PATH = "reviewer-attestation.json"
_TRUSTED_REVIEWER_ARTIFACT_FILES = {
    "gemini-request.json",
    "gemini-response.json",
    "review-packet.json",
    "reviewer-attestation.json",
    "reviewer-disposition.txt",
}
EXECUTION_BRANCH = "research/ppo-4h-indicator-smoke-execution"
EXECUTION_BASE_BRANCH = "main"
TRIGGER_MESSAGE = "run: execute 4h PPO indicator smoke"
MINIMUM_AVAILABLE_BYTES = 4 * 1024**3
DEADLINE_SECONDS = 300 * 60
_REVIEW_URL_RE = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+)/pull/(?P<pull>[1-9][0-9]*)#"
    r"pullrequestreview-(?P<review>[1-9][0-9]*)$"
)
_REVIEW_TAG_RE = re.compile(
    r"^review/ppo-4h-indicator-smoke-v(?P<version>[1-9][0-9]*)$"
)
_TRUSTED_PACKET_ARTIFACT_RE = re.compile(
    r"^ppo-4h-review-packet-(?P<identity>[0-9a-f]{64})-"
    r"(?P<run>[1-9][0-9]*)-(?P<attempt>[1-9][0-9]*)$"
)
_MAX_TRUSTED_ARTIFACT_PAGES = 20


_TRUSTED_ATTESTATION_FIELDS = {
    "schema",
    "review_protocol",
    "review_identity_digest",
    "repository",
    "repository_id",
    "reviewed_code_sha",
    "review_tag",
    "review_tag_object_sha",
    "trusted_workflow_sha",
    "trusted_workflow_ref",
    "trusted_workflow_file_sha256",
    "trusted_runner_sha256",
    "reviewer_run_id",
    "reviewer_run_attempt",
    "request_pull_number",
    "request_comment_id",
    "requester_login",
    "requester_permission",
    "request_body_sha256",
    "trusted_verification_jobs",
    "packet_sha256",
    "system_instruction_sha256",
    "gemini_request_sha256",
    "raw_gemini_response_sha256",
    "result_blind",
    "reviewer_context",
    "python_version",
    "reviewer_provider",
    "reviewer_model",
    "reviewer_response_id",
    "g0",
    "g1",
    "g2",
    "disposition",
    "blocking_findings",
    "strongest_counterexample",
    "missing_evidence",
    "claim_downgrade",
    "machine_oracles",
    "what_this_cannot_prove",
}
_TRUSTED_JOB_NAMES = {
    "core": "Trusted Lean Core verification",
    "ppo-runtime": "Trusted PPO Runtime verification",
    "guide": "Trusted Human Guide verification",
}


def _review_identity_digest(
    *,
    repository: str,
    pull_number: int,
    reviewed_code_sha: str,
) -> str:
    reviewed = transport._require_commit_sha(
        reviewed_code_sha, field="reviewed code SHA"
    )
    identity = {
        "schema": "ppo_4h_gemini_review_identity_v1",
        "repository": repository,
        "repository_id": TRUSTED_REPOSITORY_ID,
        "pull_number": transport._strict_positive_int(
            pull_number, field="execution pull number"
        ),
        "reviewed_code_sha": reviewed,
        "review_protocol": TRUSTED_REVIEW_PROTOCOL,
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def validate_trusted_reviewer_attestation(
    attestation: object,
    source: dict[str, Any],
    *,
    repository: str,
    pull_number: int,
    reviewed_code_sha: str,
    run: object,
    jobs: object,
) -> dict[str, Any]:
    if (
        not isinstance(attestation, dict)
        or set(attestation) != _TRUSTED_ATTESTATION_FIELDS
    ):
        raise ValueError("trusted reviewer attestation shape is unsupported")
    reviewed = transport._require_commit_sha(
        reviewed_code_sha, field="reviewed code SHA"
    )
    expected_identity = _review_identity_digest(
        repository=repository,
        pull_number=pull_number,
        reviewed_code_sha=reviewed,
    )
    expected_run_id = transport._strict_positive_int(
        source.get("trusted_reviewer_run_id"), field="trusted reviewer run id"
    )
    expected_attempt = transport._strict_positive_int(
        source.get("trusted_reviewer_run_attempt"),
        field="trusted reviewer run attempt",
    )
    transport._strict_positive_int(
        source.get("trusted_reviewer_artifact_id"),
        field="trusted reviewer artifact id",
    )
    transport._require_sha256(
        source.get("trusted_reviewer_artifact_sha256"),
        field="trusted reviewer artifact SHA-256",
    )
    transport._require_sha256(
        source.get("trusted_reviewer_attestation_sha256"),
        field="trusted reviewer attestation SHA-256",
    )
    if not isinstance(run, dict):
        raise ValueError("trusted reviewer workflow run is malformed")
    if (
        run.get("id") != expected_run_id
        or run.get("name") != TRUSTED_REVIEWER_WORKFLOW_NAME
        or run.get("event") != "issue_comment"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or run.get("run_attempt") != expected_attempt
        or run.get("head_branch") != EXECUTION_BASE_BRANCH
        or run.get("path") != TRUSTED_REVIEWER_WORKFLOW_PATH
    ):
        raise ValueError("trusted reviewer workflow run identity is invalid")
    run_head = transport._require_commit_sha(
        run.get("head_sha"), field="trusted reviewer workflow SHA"
    )
    if run_head != TRUSTED_REVIEWER_AUTHORITY_COMMIT:
        raise ValueError("trusted reviewer workflow run identity is invalid")
    if not isinstance(jobs, list):
        raise ValueError("trusted reviewer job inventory is malformed")
    by_name: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get("name"), str):
            continue
        if job["name"] in by_name:
            raise ValueError("trusted reviewer job inventory has duplicate names")
        by_name[job["name"]] = job
    attested_jobs = attestation.get("trusted_verification_jobs")
    if not isinstance(attested_jobs, dict) or set(attested_jobs) != set(
        _TRUSTED_JOB_NAMES
    ):
        raise ValueError("trusted reviewer attested job roster is malformed")
    for key, name in _TRUSTED_JOB_NAMES.items():
        job = by_name.get(name)
        if (
            job is None
            or job.get("status") != "completed"
            or job.get("conclusion") != "success"
            or transport._strict_positive_int(
                job.get("id"), field=f"trusted reviewer {key} job id"
            )
            != transport._strict_positive_int(
                attested_jobs.get(key), field=f"attested {key} job id"
            )
        ):
            raise ValueError(f"trusted reviewer verification job is invalid: {key}")
    review_job = by_name.get("Trusted Gemini semantic review")
    if (
        review_job is None
        or review_job.get("status") != "completed"
        or review_job.get("conclusion") != "success"
    ):
        raise ValueError("trusted Gemini semantic review job is not Green")
    if (
        attestation.get("schema") != TRUSTED_REVIEWER_ATTESTATION_SCHEMA
        or attestation.get("review_protocol") != TRUSTED_REVIEW_PROTOCOL
        or attestation.get("review_identity_digest") != expected_identity
        or attestation.get("repository") != repository
        or attestation.get("repository_id") != TRUSTED_REPOSITORY_ID
        or attestation.get("reviewed_code_sha") != reviewed
        or attestation.get("review_tag") != source.get("review_tag")
        or attestation.get("review_tag_object_sha")
        != source.get("review_tag_object_sha")
        or attestation.get("trusted_workflow_sha") != run_head
        or attestation.get("trusted_workflow_ref")
        != (f"{repository}/{TRUSTED_REVIEWER_WORKFLOW_PATH}@refs/heads/main")
        or attestation.get("trusted_workflow_file_sha256")
        != TRUSTED_REVIEWER_WORKFLOW_SHA256
        or attestation.get("trusted_runner_sha256") != TRUSTED_REVIEWER_RUNNER_SHA256
        or attestation.get("reviewer_run_id") != expected_run_id
        or attestation.get("reviewer_run_attempt") != expected_attempt
        or attestation.get("request_pull_number") != pull_number
        or attestation.get("requester_permission") not in {"write", "admin"}
        or attestation.get("result_blind") is not True
        or attestation.get("reviewer_context") != TRUSTED_REVIEWER_CONTEXT
        or attestation.get("reviewer_provider") != TRUSTED_REVIEWER_PROVIDER
        or attestation.get("g0") != "PASS"
        or attestation.get("g1") != "PASS"
        or attestation.get("g2") not in {"PASS", "EVIDENCE_BOUND"}
        or attestation.get("disposition") != "G0_G1_CLEAR_G2_EVIDENCE_BOUND"
        or attestation.get("blocking_findings") != []
    ):
        raise ValueError("trusted reviewer attestation does not authorize execution")
    for field in ("reviewer_model", "reviewer_response_id", "requester_login"):
        value = attestation.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"trusted reviewer attestation {field} is missing")
    if (
        source.get("review_identity_digest") != expected_identity
        or source.get("reviewer_provider") != attestation.get("reviewer_provider")
        or source.get("reviewer_model") != attestation.get("reviewer_model")
        or source.get("reviewer_response_id") != attestation.get("reviewer_response_id")
        or source.get("reviewer_context") != attestation.get("reviewer_context")
        or source.get("result_blind") != attestation.get("result_blind")
        or source.get("g0") != attestation.get("g0")
        or source.get("g1") != attestation.get("g1")
        or source.get("g2") != attestation.get("g2")
        or source.get("disposition") != attestation.get("disposition")
        or source.get("blocking_findings") != attestation.get("blocking_findings")
    ):
        raise ValueError("source review differs from trusted reviewer attestation")
    return attestation


def _canonical_trusted_attestation(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("trusted reviewer attestation is missing")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    expected = transport._require_sha256(
        expected_sha256,
        field="trusted reviewer attestation SHA-256",
    )
    if actual != expected:
        raise ValueError("trusted reviewer attestation SHA-256 mismatch")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("trusted reviewer attestation is invalid JSON") from None
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("trusted reviewer attestation is not canonical JSON")
    return value


def _require_trusted_reviewer_lineage(
    *,
    repository: str,
    identity_digest: str,
    expected_run_id: int,
    expected_run_attempt: int,
    token: str,
    deadline: float,
) -> None:
    identity = transport._require_sha256(
        identity_digest,
        field="trusted review identity digest",
    )
    run_id = transport._strict_positive_int(
        expected_run_id,
        field="trusted reviewer run id",
    )
    run_attempt = transport._strict_positive_int(
        expected_run_attempt,
        field="trusted reviewer run attempt",
    )
    repo_path = transport._repo_path(repository)
    matches: set[tuple[int, int]] = set()
    exhausted = False
    for page in range(1, _MAX_TRUSTED_ARTIFACT_PAGES + 1):
        payload = transport._api_json(
            "https://api.github.com/repos/"
            f"{repo_path}/actions/artifacts?per_page=100&page={page}",
            token=token,
            deadline=deadline,
        )
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("trusted review lineage artifact inventory is malformed")
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            name = artifact.get("name")
            if not isinstance(name, str):
                continue
            match = _TRUSTED_PACKET_ARTIFACT_RE.fullmatch(name)
            if match is None or match.group("identity") != identity:
                continue
            if artifact.get("expired") is not False:
                raise ValueError(
                    "trusted review lineage expiry status is missing or expired"
                )
            parsed_run = int(match.group("run"))
            parsed_attempt = int(match.group("attempt"))
            workflow_run = artifact.get("workflow_run")
            if (
                not isinstance(workflow_run, dict)
                or transport._strict_positive_int(
                    workflow_run.get("id"),
                    field="trusted review lineage workflow run id",
                )
                != parsed_run
            ):
                raise ValueError(
                    "trusted review lineage artifact identity is malformed"
                )
            key = (parsed_run, parsed_attempt)
            if key in matches:
                raise ValueError("trusted review lineage contains duplicate attempts")
            matches.add(key)
        if len(artifacts) < 100:
            exhausted = True
            break
    if not exhausted:
        raise ValueError(
            "trusted review lineage artifact inventory exceeds page budget"
        )
    if not matches:
        raise ValueError("trusted review lineage evidence is missing")
    if any(candidate_run != run_id for candidate_run, _ in matches):
        raise ValueError("trusted review lineage contains another workflow run")
    if any(candidate_attempt > run_attempt for _, candidate_attempt in matches):
        raise ValueError("trusted review lineage contains a later attempt")
    if (run_id, run_attempt) not in matches:
        raise ValueError("trusted review lineage does not contain the selected attempt")


def _require_trusted_reviewer_attestation(
    source: dict[str, Any],
    *,
    repository: str,
    pull_number: int,
    reviewed_code_sha: str,
    token: str | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    if not token:
        raise ValueError("trusted reviewer GitHub token is required")
    if deadline is None:
        raise ValueError("trusted reviewer deadline is required")
    run_id = transport._strict_positive_int(
        source.get("trusted_reviewer_run_id"),
        field="trusted reviewer run id",
    )
    run_attempt = transport._strict_positive_int(
        source.get("trusted_reviewer_run_attempt"),
        field="trusted reviewer run attempt",
    )
    artifact_id = transport._strict_positive_int(
        source.get("trusted_reviewer_artifact_id"),
        field="trusted reviewer artifact id",
    )
    artifact_sha = transport._require_sha256(
        source.get("trusted_reviewer_artifact_sha256"),
        field="trusted reviewer artifact SHA-256",
    )
    attestation_sha = transport._require_sha256(
        source.get("trusted_reviewer_attestation_sha256"),
        field="trusted reviewer attestation SHA-256",
    )
    expected_identity = _review_identity_digest(
        repository=repository,
        pull_number=pull_number,
        reviewed_code_sha=reviewed_code_sha,
    )
    if source.get("review_identity_digest") != expected_identity:
        raise ValueError("trusted review identity digest differs")
    _require_trusted_reviewer_lineage(
        repository=repository,
        identity_digest=expected_identity,
        expected_run_id=run_id,
        expected_run_attempt=run_attempt,
        token=token,
        deadline=deadline,
    )
    repo_path = transport._repo_path(repository)
    run = transport._api_json(
        f"https://api.github.com/repos/{repo_path}/actions/runs/{run_id}",
        token=token,
        deadline=deadline,
    )
    jobs_payload = transport._api_json(
        "https://api.github.com/repos/"
        f"{repo_path}/actions/runs/{run_id}/attempts/{run_attempt}/jobs?per_page=100",
        token=token,
        deadline=deadline,
    )
    jobs = jobs_payload.get("jobs")
    reference = transport.ArtifactReference(
        artifact_id=artifact_id,
        run_id=run_id,
        sha256=artifact_sha,
    )
    expected_name = f"{TRUSTED_REVIEWER_ARTIFACT_PREFIX}-{run_id}-{run_attempt}"
    with tempfile.TemporaryDirectory(prefix="ppo-4h-trusted-review-") as temp:
        root = Path(temp)
        archive = root / "review.zip"
        metadata = transport.download_artifact_archive(
            repository,
            reference,
            repository_id=TRUSTED_REPOSITORY_ID,
            token=token,
            destination=archive,
            deadline=deadline,
        )
        if metadata.get("name") != expected_name:
            raise ValueError("trusted reviewer artifact name differs")
        extracted = root / "review"
        transport.safe_extract_zip(archive, extracted)
        files = {
            path.relative_to(extracted).as_posix()
            for path in extracted.rglob("*")
            if path.is_file()
        }
        if files != _TRUSTED_REVIEWER_ARTIFACT_FILES:
            raise ValueError("trusted reviewer artifact file roster differs")
        attestation = _canonical_trusted_attestation(
            extracted / TRUSTED_REVIEWER_ATTESTATION_PATH,
            expected_sha256=attestation_sha,
        )
    return validate_trusted_reviewer_attestation(
        attestation,
        source,
        repository=repository,
        pull_number=pull_number,
        reviewed_code_sha=reviewed_code_sha,
        run=run,
        jobs=jobs,
    )


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        raise ValueError("git state could not be verified") from None
    return result.stdout.strip()


def _canonical_review(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("smoke review evidence is missing")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("smoke review evidence is invalid JSON") from None
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("smoke review evidence is not canonical JSON")
    expected = {
        "schema",
        "reviewed_code_sha",
        "static_contract_digest",
        "review_tag",
        "review_tag_object_sha",
        "review_identity_digest",
        "trusted_reviewer_run_id",
        "trusted_reviewer_run_attempt",
        "trusted_reviewer_artifact_id",
        "trusted_reviewer_artifact_sha256",
        "trusted_reviewer_attestation_sha256",
        "source_review_url",
        "source_review_body_sha256",
        "reviewer_surface",
        "result_blind",
        "g0",
        "g1",
        "g2",
        "authorized_development_smoke",
        "unused_data_accessed",
        "final_data_accessed",
    }
    if set(value) != expected or value.get("schema") != REVIEW_SCHEMA:
        raise ValueError("smoke review evidence shape is unsupported")
    return value


def _canonical_source_review(body: str) -> dict[str, Any]:
    marker_index = body.find(SOURCE_REVIEW_MARKER)
    if marker_index < 0:
        raise ValueError("source review has no structured authorization payload")
    if body.find(SOURCE_REVIEW_MARKER, marker_index + len(SOURCE_REVIEW_MARKER)) >= 0:
        raise ValueError("source review has multiple structured authorization payloads")
    raw_text = body[marker_index + len(SOURCE_REVIEW_MARKER) :]
    if not raw_text.endswith("\n"):
        raise ValueError("source review structured payload must end with a newline")
    raw = raw_text[:-1].encode("utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("source review structured payload is invalid JSON") from None
    expected = {
        "schema",
        "reviewed_code_sha",
        "static_contract_digest",
        "review_tag",
        "review_tag_object_sha",
        "review_identity_digest",
        "trusted_reviewer_run_id",
        "trusted_reviewer_run_attempt",
        "trusted_reviewer_artifact_id",
        "trusted_reviewer_artifact_sha256",
        "trusted_reviewer_attestation_sha256",
        "reviewer_independence",
        "reviewer_kind",
        "reviewer_provider",
        "reviewer_model",
        "reviewer_response_id",
        "reviewer_context",
        "result_blind",
        "g0",
        "g1",
        "g2",
        "disposition",
        "blocking_findings",
        "authorized_development_smoke",
        "unused_data_accessed",
        "final_data_accessed",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("schema") != SOURCE_REVIEW_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise ValueError("source review structured payload is not canonical")
    findings = value.get("blocking_findings")
    if not isinstance(findings, list) or any(
        not isinstance(item, str) or not item for item in findings
    ):
        raise ValueError("source review blocking findings are malformed")
    return value


def _github_principal_id(record: object, *, field: str) -> int:
    if not isinstance(record, dict):
        raise ValueError(f"{field} GitHub record is malformed")
    user = record.get("user")
    if not isinstance(user, dict):
        raise ValueError(f"{field} GitHub principal is missing")
    return transport._strict_positive_int(
        user.get("id"), field=f"{field} GitHub user id"
    )


def _github_principal_login(record: object, *, field: str) -> str:
    if not isinstance(record, dict):
        raise ValueError(f"{field} GitHub record is malformed")
    user = record.get("user")
    if not isinstance(user, dict):
        raise ValueError(f"{field} GitHub principal is missing")
    login = user.get("login")
    if not isinstance(login, str) or not login:
        raise ValueError(f"{field} GitHub login is malformed")
    return login


def _require_reviewer_write_permission(
    record: object,
    *,
    repository: str,
    token: str,
    deadline: float,
) -> None:
    login = urllib.parse.quote(
        _github_principal_login(record, field="source review"), safe=""
    )
    permission = transport._api_json(
        "https://api.github.com/repos/"
        f"{transport._repo_path(repository)}/collaborators/{login}/permission",
        token=token,
        deadline=deadline,
    ).get("permission")
    if permission not in {"write", "admin"}:
        raise ValueError("source reviewer lacks repository write permission")


def _require_review_tag_identity(
    source: dict[str, Any],
    *,
    repository: str,
    reviewed_code_sha: str,
    token: str,
    deadline: float,
) -> None:
    review_tag = source.get("review_tag")
    if not isinstance(review_tag, str) or _REVIEW_TAG_RE.fullmatch(review_tag) is None:
        raise ValueError("source review tag name is malformed")
    expected_tag_object_sha = transport._require_commit_sha(
        source.get("review_tag_object_sha"), field="review tag object SHA"
    )
    reviewed = transport._require_commit_sha(
        reviewed_code_sha, field="reviewed code SHA"
    )
    path = transport._repo_path(repository)
    ref_name = urllib.parse.quote(f"tags/{review_tag}", safe="/")
    try:
        tag_ref = transport._api_json(
            f"https://api.github.com/repos/{path}/git/ref/{ref_name}",
            token=token,
            deadline=deadline,
        )
    except transport.TransportError:
        raise ValueError("review tag identity could not be verified") from None
    if tag_ref.get("ref") != f"refs/tags/{review_tag}":
        raise ValueError("review tag ref identity differs")
    ref_object = tag_ref.get("object")
    if not isinstance(ref_object, dict) or ref_object.get("type") != "tag":
        raise ValueError("review tag must be annotated")
    actual_tag_object_sha = transport._require_commit_sha(
        ref_object.get("sha"), field="review tag object SHA"
    )
    if actual_tag_object_sha != expected_tag_object_sha:
        raise ValueError("review tag object SHA differs from reviewed identity")

    try:
        tag_object = transport._api_json(
            f"https://api.github.com/repos/{path}/git/tags/{actual_tag_object_sha}",
            token=token,
            deadline=deadline,
        )
    except transport.TransportError:
        raise ValueError("review tag identity could not be verified") from None
    if tag_object.get("tag") != review_tag:
        raise ValueError("review tag object name differs from reviewed identity")
    target = tag_object.get("object")
    if not isinstance(target, dict) or target.get("type") != "commit":
        raise ValueError("review tag object does not point to a commit")
    tagged_commit_sha = transport._require_commit_sha(
        target.get("sha"), field="review tag commit SHA"
    )
    if tagged_commit_sha != reviewed:
        raise ValueError("review tag does not point to reviewed code commit")

    try:
        current_tag_ref = transport._api_json(
            f"https://api.github.com/repos/{path}/git/ref/{ref_name}",
            token=token,
            deadline=deadline,
        )
    except transport.TransportError:
        raise ValueError("review tag identity could not be verified") from None
    current_ref_object = current_tag_ref.get("object")
    if (
        current_tag_ref.get("ref") != f"refs/tags/{review_tag}"
        or not isinstance(current_ref_object, dict)
        or current_ref_object.get("type") != "tag"
        or current_ref_object.get("sha") != actual_tag_object_sha
    ):
        raise ValueError("review tag changed during identity check")


def _review_inventory(
    *,
    repository: str,
    pull_number: int,
    token: str,
    deadline: float,
) -> list[object]:
    path = transport._repo_path(repository)
    inventory: list[object] = []
    page = 1
    while True:
        records = transport._api_json_array(
            f"https://api.github.com/repos/{path}/pulls/{pull_number}/reviews"
            f"?per_page=100&page={page}",
            token=token,
            deadline=deadline,
        )
        inventory.extend(records)
        if len(records) < 100:
            return inventory
        page += 1


def _require_current_main_contained(
    *,
    repository: str,
    reviewed_code_sha: str,
    token: str,
    deadline: float,
) -> str:
    """Require reviewed code to contain the repository's current main commit."""
    reviewed = transport._require_commit_sha(
        reviewed_code_sha, field="reviewed code SHA"
    )
    path = transport._repo_path(repository)
    branch = transport._api_json(
        f"https://api.github.com/repos/{path}/branches/{EXECUTION_BASE_BRANCH}",
        token=token,
        deadline=deadline,
    )
    commit = branch.get("commit")
    if not isinstance(commit, dict):
        raise ValueError("current main branch record is malformed")
    current_main = transport._require_commit_sha(
        commit.get("sha"), field="current main SHA"
    )
    comparison = transport._api_json(
        f"https://api.github.com/repos/{path}/compare/{current_main}...{reviewed}",
        token=token,
        deadline=deadline,
    )
    behind_by = comparison.get("behind_by")
    if isinstance(behind_by, bool) or not isinstance(behind_by, int) or behind_by < 0:
        raise ValueError("current main comparison is malformed")
    if comparison.get("status") not in {"ahead", "identical"} or behind_by != 0:
        raise ValueError("reviewed code does not contain current main")
    return current_main


def _validate_execution_pull(
    pull: object,
    *,
    repository: str,
    pull_number: int,
    expected_head_sha: str,
) -> int:
    if not isinstance(pull, dict):
        raise ValueError("source review pull request record is malformed")
    if pull.get("number") != pull_number:
        raise ValueError("source review pull request number differs")
    if pull.get("state") != "open":
        raise ValueError("source review pull request must remain open")
    if pull.get("draft") is not True:
        raise ValueError("source review execution pull request must remain draft")

    head = pull.get("head")
    base = pull.get("base")
    if not isinstance(head, dict) or not isinstance(base, dict):
        raise ValueError("source review pull request refs are malformed")
    if head.get("ref") != EXECUTION_BRANCH:
        raise ValueError("source review pull request is not the execution branch")
    if head.get("sha") != expected_head_sha:
        raise ValueError(
            "source review pull request head differs from expected exact head"
        )
    head_repo = head.get("repo")
    if not isinstance(head_repo, dict) or head_repo.get("full_name") != repository:
        raise ValueError("source review pull request belongs to another repository")
    if base.get("ref") != EXECUTION_BASE_BRANCH:
        raise ValueError("source review pull request base differs from main")
    return _github_principal_id(pull, field="pull request author")


def _validate_source_review_record(
    record: object,
    pull: object,
    *,
    repository: str,
    pull_number: int,
    reviewed_code_sha: str,
    expected_static_digest: str,
    expected_pull_head_sha: str | None = None,
    expected_url: str | None = None,
    expected_body_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict) or not isinstance(pull, dict):
        raise ValueError("source review GitHub records are malformed")
    review_url = record.get("html_url")
    body = record.get("body")
    if not isinstance(review_url, str) or not isinstance(body, str):
        raise ValueError("source review identity or body is malformed")
    match = _REVIEW_URL_RE.fullmatch(review_url)
    if match is None:
        raise ValueError("source review URL is not an exact GitHub PR review reference")
    owner, name = transport._repo_path(repository).split("/", 1)
    if (
        match.group("owner") != owner
        or match.group("repo") != name
        or int(match.group("pull")) != pull_number
    ):
        raise ValueError("source review belongs to another pull request")
    if expected_url is not None and review_url != expected_url:
        raise ValueError("source review identity or bytes differ")
    if (
        expected_body_sha256 is not None
        and hashlib.sha256(body.encode("utf-8")).hexdigest() != expected_body_sha256
    ):
        raise ValueError("source review identity or bytes differ")
    if record.get("commit_id") != reviewed_code_sha:
        raise ValueError("source review commit differs from reviewed code SHA")
    if record.get("state") not in {"COMMENTED", "APPROVED"}:
        raise ValueError("source review state does not authorize execution")
    _validate_execution_pull(
        pull,
        repository=repository,
        pull_number=pull_number,
        expected_head_sha=(
            reviewed_code_sha
            if expected_pull_head_sha is None
            else expected_pull_head_sha
        ),
    )

    source = _canonical_source_review(body)
    if source.get("reviewed_code_sha") != reviewed_code_sha:
        raise ValueError("source review payload binds a different code SHA")
    if source.get("static_contract_digest") != expected_static_digest:
        raise ValueError("source review payload binds a different static contract")
    expected_identity = _review_identity_digest(
        repository=repository,
        pull_number=pull_number,
        reviewed_code_sha=reviewed_code_sha,
    )
    if source.get("review_identity_digest") != expected_identity:
        raise ValueError("source review identity digest differs")
    transport._strict_positive_int(
        source.get("trusted_reviewer_run_id"),
        field="trusted reviewer run id",
    )
    transport._strict_positive_int(
        source.get("trusted_reviewer_run_attempt"),
        field="trusted reviewer run attempt",
    )
    transport._strict_positive_int(
        source.get("trusted_reviewer_artifact_id"),
        field="trusted reviewer artifact id",
    )
    transport._require_sha256(
        source.get("trusted_reviewer_artifact_sha256"),
        field="trusted reviewer artifact SHA-256",
    )
    transport._require_sha256(
        source.get("trusted_reviewer_attestation_sha256"),
        field="trusted reviewer attestation SHA-256",
    )
    if source.get("reviewer_independence") != "ESTABLISHED":
        raise ValueError("source review independence is not established")
    if source.get("reviewer_kind") != "external_ai":
        raise ValueError("source review is not from an external AI reviewer")
    if source.get("reviewer_provider") != TRUSTED_REVIEWER_PROVIDER:
        raise ValueError("source review is not from Google Gemini")
    reviewer_response_id = source.get("reviewer_response_id")
    if not isinstance(reviewer_response_id, str) or not reviewer_response_id.strip():
        raise ValueError("source review response identity is missing")
    reviewer_model = source.get("reviewer_model")
    if not isinstance(reviewer_model, str) or not reviewer_model.strip():
        raise ValueError("source review model provenance is missing")
    if source.get("reviewer_context") != TRUSTED_REVIEWER_CONTEXT:
        raise ValueError("source review did not use the trusted read-only context")
    if source.get("result_blind") is not True:
        raise ValueError("source review is not result-blind")
    if source.get("g0") != "PASS":
        raise ValueError("source review G0 does not pass")
    if source.get("g1") != "PASS":
        raise ValueError("source review G1 does not pass")
    if source.get("g2") not in {"PASS", "EVIDENCE_BOUND"}:
        raise ValueError("source review G2 does not authorize execution")
    if source.get("disposition") != "G0_G1_CLEAR_G2_EVIDENCE_BOUND":
        raise ValueError("source review disposition does not authorize execution")
    if source.get("blocking_findings") != []:
        raise ValueError("source review has blocking findings")
    if (
        source.get("authorized_development_smoke") is not True
        or source.get("unused_data_accessed") is not False
        or source.get("final_data_accessed") is not False
    ):
        raise ValueError("source review does not authorize the development smoke")
    return source


def validate_independent_review_status(
    record: object,
    pull: object,
    *,
    repository: str,
    pull_number: int,
    reviewed_code_sha: str,
) -> dict[str, Any]:
    """Validate one GitHub review event as exact-head independent authorization."""
    reviewed = transport._require_commit_sha(
        reviewed_code_sha, field="reviewed code SHA"
    )
    return _validate_source_review_record(
        record,
        pull,
        repository=repository,
        pull_number=pull_number,
        reviewed_code_sha=reviewed,
        expected_static_digest=content_digest(smoke.static_protocol_contract()),
    )


def find_authorizing_source_review(
    *,
    repository: str,
    pull_number: int,
    reviewed_code_sha: str,
    token: str,
    deadline: float,
) -> dict[str, Any]:
    """Return an authorizing latest-per-reviewer review for the exact code HEAD."""
    reviewed = transport._require_commit_sha(
        reviewed_code_sha, field="reviewed code SHA"
    )
    _require_current_main_contained(
        repository=repository,
        reviewed_code_sha=reviewed,
        token=token,
        deadline=deadline,
    )
    path = transport._repo_path(repository)
    pull = transport._api_json(
        f"https://api.github.com/repos/{path}/pulls/{pull_number}",
        token=token,
        deadline=deadline,
    )
    _validate_execution_pull(
        pull,
        repository=repository,
        pull_number=pull_number,
        expected_head_sha=reviewed,
    )
    inventory = _review_inventory(
        repository=repository,
        pull_number=pull_number,
        token=token,
        deadline=deadline,
    )

    for record in reversed(inventory):
        try:
            source = validate_independent_review_status(
                record,
                pull,
                repository=repository,
                pull_number=pull_number,
                reviewed_code_sha=reviewed,
            )
            _require_review_tag_identity(
                source,
                repository=repository,
                reviewed_code_sha=reviewed,
                token=token,
                deadline=deadline,
            )
            _require_reviewer_write_permission(
                record,
                repository=repository,
                token=token,
                deadline=deadline,
            )
            _require_trusted_reviewer_attestation(
                source,
                repository=repository,
                pull_number=pull_number,
                reviewed_code_sha=reviewed,
                token=token,
                deadline=deadline,
            )
        except ValueError:
            continue
        if isinstance(record, dict):
            return record
    raise ValueError("independent research review is pending")


def execute_review_status_from_environment(
    environment: dict[str, str] | None = None,
) -> int:
    """Project the independent-review gate into a visible GitHub check."""
    env = dict(os.environ if environment is None else environment)
    repository = env.get("GITHUB_REPOSITORY", "")
    token = env.get("GITHUB_TOKEN", "")
    event_raw = env.get("GITHUB_EVENT_PATH", "")
    summary_raw = env.get("GITHUB_STEP_SUMMARY", "")
    if not token:
        raise ValueError("GITHUB_TOKEN is required")
    if not event_raw:
        raise ValueError("GITHUB_EVENT_PATH is required")
    if not summary_raw:
        raise ValueError("GITHUB_STEP_SUMMARY is required")
    event_path = Path(event_raw)
    summary_path = Path(summary_raw)
    if not event_path.is_file():
        raise ValueError("GITHUB_EVENT_PATH is required")
    try:
        event = json.loads(event_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError("GitHub event payload is unreadable") from None
    if not isinstance(event, dict):
        raise ValueError("GitHub event payload is malformed")
    pull = event.get("pull_request")
    if not isinstance(pull, dict):
        raise ValueError("GitHub event has no pull request")
    pull_number = transport._strict_positive_int(
        pull.get("number"), field="pull request number"
    )
    head = pull.get("head")
    if not isinstance(head, dict):
        raise ValueError("GitHub event pull request head is malformed")
    reviewed = transport._require_commit_sha(
        head.get("sha"), field="pull request head SHA"
    )
    deadline = time.monotonic() + 60.0
    try:
        record = find_authorizing_source_review(
            repository=repository,
            pull_number=pull_number,
            reviewed_code_sha=reviewed,
            token=token,
            deadline=deadline,
        )
    except ValueError as error:
        message = (
            "# Independent Research Review\n\n"
            f"**PENDING** for exact HEAD `{reviewed}`.\n\n"
            f"{error}. Economic execution remains blocked.\n"
        )
        summary_path.write_text(message, encoding="utf-8")
        print(message)
        return 1

    reviewer = record.get("user")
    login = reviewer.get("login") if isinstance(reviewer, dict) else None
    review_url = record.get("html_url")
    message = (
        "# Independent Research Review\n\n"
        f"**READY** for exact HEAD `{reviewed}`.\n\n"
        f"Qualifying formal review: {review_url} by {login}. "
        "This authorizes only the review-evidence transition; "
        "economic execution still requires the authenticated one-shot trigger.\n"
    )
    summary_path.write_text(message, encoding="utf-8")
    print(message)
    return 0


def validate_review_gate(
    review: dict[str, Any],
    *,
    repository: str,
    token: str,
    deadline: float,
) -> None:
    """Bind the trigger commit to one exact reviewed code parent and review comment."""
    head = _git("rev-parse", "HEAD")
    parent = _git("rev-parse", "HEAD^")
    if _git("log", "-1", "--pretty=%s") != TRIGGER_MESSAGE:
        raise ValueError("smoke trigger commit message differs")
    reviewed = transport._require_commit_sha(
        review.get("reviewed_code_sha"), field="reviewed code SHA"
    )
    if reviewed != parent:
        raise ValueError("smoke trigger parent differs from reviewed code SHA")
    changed = tuple(
        line
        for line in _git("diff", "--name-only", reviewed, head).splitlines()
        if line
    )
    if changed != (REVIEW_PATH.as_posix(),):
        raise ValueError(
            "smoke trigger commit may change only the review evidence file"
        )
    expected_static = content_digest(smoke.static_protocol_contract())
    if review.get("static_contract_digest") != expected_static:
        raise ValueError("smoke review binds a different static protocol")
    if (
        review.get("reviewer_surface") != REVIEWER_SURFACE
        or review.get("result_blind") is not True
        or review.get("g0") != "PASS"
        or review.get("g1") != "PASS"
        or review.get("g2") not in {"PASS", "EVIDENCE_BOUND"}
        or review.get("authorized_development_smoke") is not True
        or review.get("unused_data_accessed") is not False
        or review.get("final_data_accessed") is not False
    ):
        raise ValueError("smoke review does not authorize the development run")
    review_url = review.get("source_review_url")
    review_sha = transport._require_sha256(
        review.get("source_review_body_sha256"),
        field="source review body SHA-256",
    )
    if not isinstance(review_url, str):
        raise ValueError("source review URL is malformed")
    match = _REVIEW_URL_RE.fullmatch(review_url)
    if match is None:
        raise ValueError("source review URL is not an exact GitHub PR review reference")
    owner, name = transport._repo_path(repository).split("/", 1)
    if match.group("owner") != owner or match.group("repo") != name:
        raise ValueError("source review belongs to another repository")
    pull_number = int(match.group("pull"))
    review_id = match.group("review")
    record = transport._api_json(
        "https://api.github.com/repos/"
        f"{transport._repo_path(repository)}/pulls/{pull_number}/reviews/{review_id}",
        token=token,
        deadline=deadline,
    )
    pull = transport._api_json(
        "https://api.github.com/repos/"
        f"{transport._repo_path(repository)}/pulls/{pull_number}",
        token=token,
        deadline=deadline,
    )
    source = _validate_source_review_record(
        record,
        pull,
        repository=repository,
        pull_number=pull_number,
        reviewed_code_sha=reviewed,
        expected_static_digest=expected_static,
        expected_pull_head_sha=head,
        expected_url=review_url,
        expected_body_sha256=review_sha,
    )
    if review.get("review_tag") != source.get("review_tag") or review.get(
        "review_tag_object_sha"
    ) != source.get("review_tag_object_sha"):
        raise ValueError(
            "trigger review tag identity differs from the authenticated source review"
        )
    _require_review_tag_identity(
        source,
        repository=repository,
        reviewed_code_sha=reviewed,
        token=token,
        deadline=deadline,
    )
    _require_reviewer_write_permission(
        record,
        repository=repository,
        token=token,
        deadline=deadline,
    )
    _require_trusted_reviewer_attestation(
        source,
        repository=repository,
        pull_number=pull_number,
        reviewed_code_sha=reviewed,
        token=token,
        deadline=deadline,
    )
    _require_current_main_contained(
        repository=repository,
        reviewed_code_sha=reviewed,
        token=token,
        deadline=deadline,
    )

    for field in (
        "review_identity_digest",
        "trusted_reviewer_run_id",
        "trusted_reviewer_run_attempt",
        "trusted_reviewer_artifact_id",
        "trusted_reviewer_artifact_sha256",
        "trusted_reviewer_attestation_sha256",
        "result_blind",
        "g0",
        "g1",
        "g2",
        "authorized_development_smoke",
        "unused_data_accessed",
        "final_data_accessed",
    ):
        if review.get(field) != source.get(field):
            raise ValueError(
                "trigger review evidence differs from the authenticated source review"
            )


def _download_source(
    *,
    repository: str,
    token: str,
    deadline: float,
    root: Path,
) -> Path:
    metadata = transport._api_json(
        f"https://api.github.com/repos/{transport._repo_path(repository)}",
        token=token,
        deadline=deadline,
    )
    repository_id = transport._strict_positive_int(
        metadata.get("id"), field="repository id"
    )
    reference = transport.ArtifactReference(
        SOURCE_ARTIFACT_ID,
        SOURCE_ARTIFACT_RUN_ID,
        SOURCE_ARTIFACT_SHA256,
    )
    archive = root / "source.zip"
    transport.download_artifact_archive(
        repository,
        reference,
        repository_id=repository_id,
        token=token,
        destination=archive,
        deadline=deadline,
    )
    extracted = root / "source"
    transport.extract_verified_archive(
        archive,
        extracted,
        SOURCE_ARTIFACT_SHA256,
    )
    return transport.find_source_root(extracted)


def execute_from_environment(environment: dict[str, str] | None = None) -> int:
    env = dict(os.environ if environment is None else environment)
    repository = env.get("GITHUB_REPOSITORY", "")
    token = env.get("SMOKE_GITHUB_TOKEN", "")
    output_raw = env.get("SMOKE_OUTPUT", "output/smoke")
    if not token:
        raise ValueError("SMOKE_GITHUB_TOKEN is required")
    if transport.available_memory_bytes() < MINIMUM_AVAILABLE_BYTES:
        raise ValueError("less than 4 GiB memory is available for PPO smoke")
    deadline = time.monotonic() + DEADLINE_SECONDS
    review = _canonical_review(REVIEW_PATH)
    validate_review_gate(
        review,
        repository=repository,
        token=token,
        deadline=deadline,
    )
    output = Path(output_raw)
    if output.exists() or output.is_symlink():
        raise FileExistsError("smoke output path already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ppo-4h-smoke-source-") as temp:
        source = _download_source(
            repository=repository,
            token=token,
            deadline=deadline,
            root=Path(temp),
        )
        smoke.prepare_smoke(source, output)
        validate_review_gate(
            review,
            repository=repository,
            token=token,
            deadline=deadline,
        )
        smoke.run_smoke(source, output)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["review-status"]:
        return execute_review_status_from_environment()
    if args:
        raise ValueError("unsupported 4h PPO smoke action")
    return execute_from_environment()


if __name__ == "__main__":
    raise SystemExit(main())
