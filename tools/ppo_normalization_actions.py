"""Authenticated one-shot transport for corrected PPO normalization execution.

The transport is intentionally separate from the production PPO/evaluation owners. It
binds result-blind implementation/review/source authority, establishes one repository-
global activation, moves complete evidence between trusted jobs, and reveals the
comparison only after a fresh no-refit verifier artifact is API-bound.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ACTIVATION_AUTHORITY_SCHEMA: Final = (
    "ppo_normalization_execution_activation_authority_v1"
)
EXECUTION_ACTIVATION_SCHEMA: Final = "ppo_normalization_execution_activation_v1"
VERIFIER_ARTIFACT_AUTHORITY_SCHEMA: Final = (
    "ppo_normalization_replication_verifier_artifact_authority_v1"
)
EXECUTION_FAILURE_SCHEMA: Final = "ppo_normalization_execution_failure_v1"


def canonical_json_bytes(value: object) -> bytes:
    """Encode the transport's JSON-only evidence deterministically."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    artifact_id: int
    run_id: int
    sha256: str


def _checkpoint_transport() -> Any:
    try:
        from tools import ppo_checkpoint_actions
    except ImportError:  # pragma: no cover - direct trusted-script execution
        import ppo_checkpoint_actions  # type: ignore[import-not-found,no-redef]

    return ppo_checkpoint_actions


def build_candidate_run_provenance() -> dict[str, object]:
    from trade_rl.evaluation.runs import (
        build_candidate_run_provenance as build_provenance,
    )

    return build_provenance()


def replication_arm_specs() -> tuple[Any, ...]:
    from trade_rl.evaluation.ppo_normalization_execution import (
        replication_arm_specs as specs,
    )

    return specs()


def prepare_replication_execution(root: Path, activation: dict[str, object]) -> None:
    from trade_rl.evaluation.ppo_normalization_execution import (
        prepare_replication_execution as prepare,
    )

    prepare(root, activation)


def execute_replication_slot(source: Path, root: Path, slot: str) -> dict[str, object]:
    from trade_rl.evaluation.ppo_normalization_execution import (
        execute_replication_slot as execute,
    )

    return execute(source, root, slot)


def replication_slot_state(root: Path, spec: Any) -> dict[str, object]:
    from trade_rl.evaluation.ppo_normalization_execution import (
        replication_slot_state as state,
    )

    return state(root, spec)


def verify_replication_slot(source: Path, root: Path, slot: str) -> dict[str, object]:
    from trade_rl.evaluation.ppo_normalization_execution import (
        verify_replication_slot as verify,
    )

    return verify(source, root, slot)


def publish_replication_decision(root: Path) -> dict[str, object]:
    from trade_rl.evaluation.ppo_normalization_execution import (
        publish_replication_decision as publish,
    )

    return publish(root)


REQUEST_SCHEMA: Final = "ppo_normalization_execution_request_v1"
REVIEW_REQUEST_MARKER: Final = "<!-- ppo-normalization-execution-request-v1 -->"
REQUEST_PATH: Final = Path("report/ppo-normalization-execution-request.json")
ACTIVATION_TAG: Final = "activation/ppo-normalization-corrected-v1"

PROTOCOL_SHA256: Final = (
    "0013470ed5858eaa3b9391f97f4b18f772495d21c128832f74e1c50b090df304"
)
IMPLEMENTATION_DIGEST: Final = (
    "1de7baa54625c106a75a97047f40e085ad7b1bddd00012aa342ba9f1233de97c"
)
IMPLEMENTATION_SEAL_SHA256: Final = (
    "68fd92cfe823498625e7f44b0a2c5781748f0cfa3f454010d72ab674ac18272a"
)
FRESH_RECONSTRUCTION_SHA256: Final = (
    "fd3d9319550d9010102aac35cee9f46c05265bcc770f1ace41a67a3a65a69611"
)
ASSURANCE_REVIEW_SHA256: Final = (
    "bfdd2f74fd4c43372617091cb3be1b010c42347cf3cea680bee2b41555fc9d95"
)
IMPLEMENTATION_SEAL_TAG: Final = "seal/ppo-normalization-implementation-20260927-v2"
IMPLEMENTATION_SEAL_TAG_OBJECT_SHA: Final = "2bfab7f00b77c263bf9579b88be87b9cc575f9a1"
IMPLEMENTATION_CODE_SHA: Final = "72ec5a082a1e298ccfe904f06c16d423e12a76b2"
IMPLEMENTATION_MERGE_SHA: Final = "6d1e5fdc5a87cc068d1da7ddcde15e53b083a002"
REVIEW_TAG: Final = "review/ppo-normalization-replication-v2"
REVIEW_TAG_OBJECT_SHA: Final = "d6ea370eb4dd43b77988ed2c7f517ae0645d4e54"
SOURCE_ARTIFACT_RUN_ID: Final = 34_803_217_815
SOURCE_ARTIFACT_ID: Final = 10_331_899_302
SOURCE_ARTIFACT_SHA256: Final = (
    "89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce"
)
NULL_ACTIVATION_RESOURCE_SHA256: Final = (
    "6dfffbc047a9578c305f7591727e8e7ecb3174eaf63b47bfb00c72a7069cf1bd"
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a full lowercase commit SHA")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def build_execution_request() -> dict[str, object]:
    """Build the fixed result-blind request record independent of its PR HEAD."""

    return {
        "schema": REQUEST_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_digest": IMPLEMENTATION_DIGEST,
        "implementation_seal_sha256": IMPLEMENTATION_SEAL_SHA256,
        "fresh_reconstruction_sha256": FRESH_RECONSTRUCTION_SHA256,
        "assurance_review_sha256": ASSURANCE_REVIEW_SHA256,
        "source_artifact": {
            "run_id": SOURCE_ARTIFACT_RUN_ID,
            "artifact_id": SOURCE_ARTIFACT_ID,
            "sha256": SOURCE_ARTIFACT_SHA256,
        },
        "economic_execution_authorized": True,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


def validate_execution_request(payload: object) -> dict[str, object]:
    """Require a request to equal the frozen one-shot contract exactly."""

    if not isinstance(payload, dict):
        raise ValueError("execution request must be an object")
    expected = build_execution_request()
    if set(payload) != set(expected):
        raise ValueError("execution request has an unexpected shape")
    if payload.get("schema") != REQUEST_SCHEMA:
        raise ValueError("execution request schema differs")
    if payload.get("implementation_seal_sha256") != IMPLEMENTATION_SEAL_SHA256:
        raise ValueError("execution request implementation seal differs")
    for field in (
        "economic_result_inspected",
        "unused_data_accessed",
        "final_test_accessed",
        "production_eligible",
        "live_trading_authorized",
    ):
        if payload.get(field) is not False:
            raise ValueError(f"execution request illegally enables {field}")
    if payload.get("economic_execution_authorized") is not True:
        raise ValueError("execution request does not authorize economics")
    if payload != expected:
        raise ValueError("execution request differs from frozen authority")
    return payload


def parse_execution_request_comment(body: object) -> dict[str, str]:
    """Parse the canonical exact-head request comment."""

    if not isinstance(body, str):
        raise ValueError("execution request comment is malformed")
    prefix = REVIEW_REQUEST_MARKER + "\n"
    if not body.startswith(prefix) or not body.endswith("\n"):
        raise ValueError("execution request comment is not canonical")
    raw = body[len(prefix) : -1].encode("utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("execution request comment is invalid JSON") from None
    expected_keys = {"execution_code_sha", "request_sha256"}
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or canonical_json_bytes(payload) != raw
    ):
        raise ValueError("execution request comment is not canonical")
    code_sha = _require_sha(
        payload.get("execution_code_sha"), field="execution code SHA"
    )
    request_sha = _require_sha256(
        payload.get("request_sha256"), field="execution request SHA-256"
    )
    return {"execution_code_sha": code_sha, "request_sha256": request_sha}


def validate_execution_pr_files(files: list[str]) -> None:
    """Allow exactly one result-blind request record beyond current main."""

    if files != [REQUEST_PATH.as_posix()]:
        raise ValueError("execution PR may change only the canonical request record")


def build_execution_activation(
    request: dict[str, object],
    provenance: dict[str, object],
) -> dict[str, object]:
    """Bind the current execution runtime to the already sealed implementation."""

    validate_execution_request(request)
    if provenance.get("implementation_digest") != IMPLEMENTATION_DIGEST:
        raise ValueError("execution provenance implementation digest differs from seal")
    return {
        "schema": EXECUTION_ACTIVATION_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_digest": IMPLEMENTATION_DIGEST,
        "implementation_seal_sha256": IMPLEMENTATION_SEAL_SHA256,
        "fresh_reconstruction_sha256": FRESH_RECONSTRUCTION_SHA256,
        "assurance_review_sha256": ASSURANCE_REVIEW_SHA256,
        "provenance": provenance,
        "economic_execution_authorized": True,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


def activation_authority_payload(activation: dict[str, object]) -> dict[str, object]:
    """Return the non-Python authority bytes written only inside the one-shot run."""

    return {
        "schema": ACTIVATION_AUTHORITY_SCHEMA,
        "activation_sha256": content_digest(activation),
    }


def build_verifier_artifact_authority(
    *,
    repository_id: int,
    run_id: int,
    artifact_id: int,
    artifact_sha256: str,
    code_sha: str,
    workflow_sha: str,
    activation_digest: str,
    implementation_digest: str,
    verification_set_sha256: str,
) -> dict[str, object]:
    """Bind the complete fresh verification artifact to GitHub authority."""

    artifact_digest = _require_sha256(
        artifact_sha256, field="verifier artifact SHA-256"
    )
    implementation = _require_sha256(
        implementation_digest, field="implementation digest"
    )
    return {
        "schema": VERIFIER_ARTIFACT_AUTHORITY_SCHEMA,
        "repository_id": _positive_int(repository_id, field="repository id"),
        "run_id": _positive_int(run_id, field="workflow run id"),
        "artifact_id": _positive_int(artifact_id, field="verifier artifact id"),
        "artifact_sha256": artifact_digest,
        "artifact_api_digest": f"sha256:{artifact_digest}",
        "code_sha": _require_sha(code_sha, field="execution code SHA"),
        "workflow_sha": _require_sha(workflow_sha, field="workflow SHA"),
        "activation_digest": _require_sha256(
            activation_digest, field="activation digest"
        ),
        "implementation_digest": implementation,
        "verification_set_sha256": _require_sha256(
            verification_set_sha256, field="verification set SHA-256"
        ),
    }


def build_execution_failure_receipt(
    *,
    repository: str,
    repository_id: int,
    run_id: int,
    run_attempt: int,
    execution_code_sha: str,
    request_sha256: str,
    activation_sha256: str | None,
    activation_tag_created: bool,
    phase: str,
    slot: str | None,
    slot_consumed: bool | None,
) -> dict[str, object]:
    """Build non-economic durable provenance for a failed one-shot run."""

    if not isinstance(phase, str) or not phase:
        raise ValueError("execution failure phase must be non-empty")
    if slot is not None and (not isinstance(slot, str) or not slot):
        raise ValueError("execution failure slot must be non-empty text")
    if slot_consumed is not None and not isinstance(slot_consumed, bool):
        raise ValueError("execution failure slot_consumed must be boolean or null")
    if not isinstance(activation_tag_created, bool):
        raise ValueError("activation_tag_created must be boolean")
    activation = (
        None
        if activation_sha256 is None
        else _require_sha256(activation_sha256, field="activation SHA-256")
    )
    return {
        "schema": EXECUTION_FAILURE_SCHEMA,
        "repository": repository,
        "repository_id": _positive_int(repository_id, field="repository id"),
        "run_id": _positive_int(run_id, field="workflow run id"),
        "run_attempt": _positive_int(run_attempt, field="workflow run attempt"),
        "execution_code_sha": _require_sha(
            execution_code_sha, field="execution code SHA"
        ),
        "request_sha256": _require_sha256(
            request_sha256, field="execution request SHA-256"
        ),
        "activation_sha256": activation,
        "activation_tag_created": activation_tag_created,
        "phase": phase,
        "slot": slot,
        "slot_consumed": slot_consumed,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


class TransportError(RuntimeError):
    """Secret-free transport failure suitable for workflow logs."""


def _repo_path(repository: str) -> str:
    parts = repository.split("/")
    if len(parts) != 2 or any(
        not part or re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None for part in parts
    ):
        raise ValueError("GITHUB_REPOSITORY is malformed")
    return "/".join(urllib.parse.quote(part, safe="") for part in parts)


def _api_json(
    url: str,
    *,
    token: str,
    method: str = "GET",
    payload: object | None = None,
    allow_not_found: bool = False,
) -> dict[str, Any] | None:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "trade-rl-ppo-normalization-one-shot",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = None if payload is None else canonical_json_bytes(payload)
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status not in {200, 201}:
                raise TransportError("GitHub API returned an unexpected status")
            raw = response.read(2 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as error:
        if allow_not_found and error.code == 404:
            return None
        raise TransportError(
            f"GitHub API request failed with HTTP {error.code}"
        ) from None
    except (OSError, urllib.error.URLError, TimeoutError):
        raise TransportError("GitHub API request could not be completed") from None
    if len(raw) > 2 * 1024 * 1024:
        raise TransportError("GitHub API response exceeds the metadata limit")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise TransportError("GitHub API returned malformed JSON") from None
    if not isinstance(value, dict):
        raise TransportError("GitHub API response is not an object")
    return value


def _api_array(url: str, *, token: str) -> list[Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "trade-rl-ppo-normalization-one-shot",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as error:
        raise TransportError(
            f"GitHub API request failed with HTTP {error.code}"
        ) from None
    except (OSError, urllib.error.URLError, TimeoutError):
        raise TransportError("GitHub API request could not be completed") from None
    if len(raw) > 4 * 1024 * 1024:
        raise TransportError("GitHub API response exceeds the metadata limit")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise TransportError("GitHub API returned malformed JSON") from None
    if not isinstance(value, list):
        raise TransportError("GitHub API response is not an array")
    return value


def _canonical_file(path: Path, *, field: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} is missing or unsafe")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"{field} is invalid JSON") from None
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise ValueError(f"{field} must be canonical JSON")
    return payload, raw


def _write_canonical_once(path: Path, payload: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"evidence path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ValueError("evidence parent must not be a symlink")
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())


def _git_head(root: Path) -> str:
    try:
        value = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        raise ValueError("execution checkout HEAD could not be verified") from None
    return _require_sha(value, field="execution checkout HEAD")


def _tree_manifest(root: Path) -> dict[str, object]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("evidence tree must be a regular directory")
    files: list[dict[str, object]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        if path.is_symlink():
            raise ValueError("evidence tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("evidence tree contains a special file")
        raw = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    return {"schema": "ppo_normalization_evidence_tree_v1", "files": files}


def _tree_digest(root: Path) -> str:
    return content_digest(_tree_manifest(root))


def _copy_tree(source: Path, destination: Path) -> None:
    _tree_manifest(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("evidence destination already exists")
    shutil.copytree(source, destination)
    if _tree_digest(source) != _tree_digest(destination):
        raise ValueError("copied evidence tree differs")


def _validate_tag(
    repository: str,
    *,
    token: str,
    tag: str,
    expected_object_sha: str,
    expected_target_sha: str,
    expected_message: str | None = None,
) -> None:
    repo = _repo_path(repository)
    encoded_tag = urllib.parse.quote(tag, safe="/")
    ref = _api_json(
        f"https://api.github.com/repos/{repo}/git/ref/tags/{encoded_tag}",
        token=token,
    )
    if (
        ref is None
        or not isinstance(ref.get("object"), dict)
        or ref["object"].get("sha") != expected_object_sha
        or ref["object"].get("type") != "tag"
    ):
        raise ValueError(f"annotated tag authority differs: {tag}")
    tag_object = _api_json(
        f"https://api.github.com/repos/{repo}/git/tags/{expected_object_sha}",
        token=token,
    )
    if (
        tag_object is None
        or tag_object.get("tag") != tag
        or not isinstance(tag_object.get("object"), dict)
        or tag_object["object"].get("sha") != expected_target_sha
        or tag_object["object"].get("type") != "commit"
    ):
        raise ValueError(f"annotated tag target differs: {tag}")
    if expected_message is not None and tag_object.get("message") != expected_message:
        raise ValueError(f"annotated tag message differs: {tag}")


def _require_no_activation_tag(repository: str, *, token: str) -> None:
    repo = _repo_path(repository)
    tag = urllib.parse.quote(ACTIVATION_TAG, safe="/")
    existing = _api_json(
        f"https://api.github.com/repos/{repo}/git/ref/tags/{tag}",
        token=token,
        allow_not_found=True,
    )
    if existing is not None:
        raise ValueError(
            "repository-global PPO normalization activation already exists"
        )


def _source_reference() -> ArtifactReference:
    return ArtifactReference(
        SOURCE_ARTIFACT_ID,
        SOURCE_ARTIFACT_RUN_ID,
        SOURCE_ARTIFACT_SHA256,
    )


def _validate_source_artifact(
    repository: str,
    *,
    repository_id: int,
    token: str,
) -> dict[str, Any]:
    repo = _repo_path(repository)
    metadata = _api_json(
        f"https://api.github.com/repos/{repo}/actions/artifacts/{SOURCE_ARTIFACT_ID}",
        token=token,
    )
    if metadata is None:
        raise ValueError("source artifact metadata is missing")
    reference = _source_reference()
    run = metadata.get("workflow_run")
    if (
        _positive_int(metadata.get("id"), field="source artifact id")
        != reference.artifact_id
        or metadata.get("expired") is not False
        or not isinstance(run, dict)
        or _positive_int(run.get("id"), field="source artifact run id")
        != reference.run_id
        or _positive_int(
            run.get("repository_id"), field="source artifact repository id"
        )
        != repository_id
        or _positive_int(
            run.get("head_repository_id"), field="source artifact head repository id"
        )
        != repository_id
        or metadata.get("digest") != f"sha256:{reference.sha256}"
    ):
        raise ValueError(
            "source artifact GitHub metadata differs from sealed authority"
        )
    return metadata


def _require_green_exact_head_ci(
    repository: str,
    *,
    token: str,
    pull_number: int,
    head_sha: str,
) -> int:
    repo = _repo_path(repository)
    runs = _api_json(
        f"https://api.github.com/repos/{repo}/actions/runs"
        f"?head_sha={head_sha}&per_page=100",
        token=token,
    )
    # The endpoint wraps runs in an object.
    if runs is None or not isinstance(runs.get("workflow_runs"), list):
        raise ValueError("exact-head CI inventory is malformed")
    candidates = [
        item
        for item in runs["workflow_runs"]
        if isinstance(item, dict)
        and item.get("name") == "CI"
        and item.get("head_sha") == head_sha
        and item.get("status") == "completed"
        and any(
            isinstance(pr, dict) and pr.get("number") == pull_number
            for pr in item.get("pull_requests", [])
        )
    ]
    if not candidates:
        raise ValueError("exact-head Full CI is missing")
    run = max(candidates, key=lambda item: int(item.get("id", 0)))
    run_id = _positive_int(run.get("id"), field="CI run id")
    jobs = _api_json(
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
        token=token,
    )
    if jobs is None or not isinstance(jobs.get("jobs"), list):
        raise ValueError("exact-head CI job inventory is malformed")
    by_name = {
        job.get("name"): job
        for job in jobs["jobs"]
        if isinstance(job, dict) and isinstance(job.get("name"), str)
    }
    required = (
        "Lean Core",
        "PPO Runtime",
        "Human Guide",
        "Generic Independent Research Review",
    )
    for name in required:
        job = by_name.get(name)
        if (
            not isinstance(job, dict)
            or job.get("status") != "completed"
            or job.get("conclusion") != "success"
        ):
            raise ValueError(f"exact-head CI job is not Green: {name}")
    return run_id


def _decode_contents_file(record: object) -> bytes:
    if not isinstance(record, dict) or record.get("encoding") != "base64":
        raise ValueError("request file GitHub record is malformed")
    content = record.get("content")
    if not isinstance(content, str):
        raise ValueError("request file GitHub record has no content")
    try:
        return base64.b64decode(content, validate=False)
    except ValueError:
        raise ValueError("request file GitHub content is invalid base64") from None


def _request_outputs(event: dict[str, Any], *, token: str) -> dict[str, str]:
    repository_record = event.get("repository")
    issue = event.get("issue")
    comment = event.get("comment")
    if (
        not isinstance(repository_record, dict)
        or not isinstance(issue, dict)
        or not isinstance(issue.get("pull_request"), dict)
        or not isinstance(comment, dict)
        or not isinstance(comment.get("user"), dict)
    ):
        raise ValueError("GitHub execution request event is malformed")
    repository = repository_record.get("full_name")
    repository_id = repository_record.get("id")
    pull_number = issue.get("number")
    requester = comment["user"].get("login")
    if (
        not isinstance(repository, str)
        or isinstance(repository_id, bool)
        or not isinstance(repository_id, int)
        or repository_id < 1
        or isinstance(pull_number, bool)
        or not isinstance(pull_number, int)
        or pull_number < 1
        or not isinstance(requester, str)
        or not requester
    ):
        raise ValueError("GitHub execution request identity is malformed")
    parsed = parse_execution_request_comment(comment.get("body"))
    head_sha = parsed["execution_code_sha"]
    repo = _repo_path(repository)

    permission = _api_json(
        f"https://api.github.com/repos/{repo}/collaborators/"
        f"{urllib.parse.quote(requester, safe='')}/permission",
        token=token,
    )
    if permission is None or permission.get("permission") not in {"write", "admin"}:
        raise ValueError("execution requester does not have write authority")

    pull = _api_json(
        f"https://api.github.com/repos/{repo}/pulls/{pull_number}",
        token=token,
    )
    if pull is None:
        raise ValueError("execution pull request is missing")
    head = pull.get("head")
    base = pull.get("base")
    if (
        pull.get("state") != "open"
        or pull.get("draft") is not True
        or not isinstance(head, dict)
        or head.get("sha") != head_sha
        or not isinstance(head.get("repo"), dict)
        or head["repo"].get("id") != repository_id
        or not isinstance(base, dict)
        or base.get("ref") != "main"
    ):
        raise ValueError(
            "execution pull request is not the required open Draft exact head"
        )

    main = _api_json(
        f"https://api.github.com/repos/{repo}/branches/main",
        token=token,
    )
    if main is None or not isinstance(main.get("commit"), dict):
        raise ValueError("current main could not be resolved")
    main_sha = _require_sha(main["commit"].get("sha"), field="current main SHA")
    compare = _api_json(
        f"https://api.github.com/repos/{repo}/compare/{main_sha}...{head_sha}",
        token=token,
    )
    if (
        compare is None
        or compare.get("behind_by") != 0
        or not isinstance(compare.get("merge_base_commit"), dict)
        or compare["merge_base_commit"].get("sha") != main_sha
    ):
        raise ValueError("execution head does not contain current main")
    files = compare.get("files")
    if not isinstance(files, list):
        raise ValueError("execution PR compare file inventory is malformed")
    validate_execution_pr_files(
        [str(item.get("filename")) for item in files if isinstance(item, dict)]
    )

    contents = _api_json(
        f"https://api.github.com/repos/{repo}/contents/"
        f"{urllib.parse.quote(REQUEST_PATH.as_posix(), safe='/')}?ref={head_sha}",
        token=token,
    )
    raw = _decode_contents_file(contents)
    try:
        request_payload = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("execution request record is invalid JSON") from None
    if (
        not isinstance(request_payload, dict)
        or canonical_json_bytes(request_payload) != raw
    ):
        raise ValueError("execution request record must be canonical JSON")
    validate_execution_request(request_payload)
    request_sha = hashlib.sha256(raw).hexdigest()
    if request_sha != parsed["request_sha256"]:
        raise ValueError(
            "execution request comment digest differs from committed request"
        )

    _require_green_exact_head_ci(
        repository,
        token=token,
        pull_number=pull_number,
        head_sha=head_sha,
    )
    implementation_pull = _api_json(
        f"https://api.github.com/repos/{repo}/pulls/770",
        token=token,
    )
    if (
        implementation_pull is None
        or implementation_pull.get("merged") is not True
        or not isinstance(implementation_pull.get("head"), dict)
        or implementation_pull["head"].get("sha") != IMPLEMENTATION_CODE_SHA
        or implementation_pull.get("merge_commit_sha") != IMPLEMENTATION_MERGE_SHA
    ):
        raise ValueError("merged implementation authority differs")
    _validate_tag(
        repository,
        token=token,
        tag=IMPLEMENTATION_SEAL_TAG,
        expected_object_sha=IMPLEMENTATION_SEAL_TAG_OBJECT_SHA,
        expected_target_sha=IMPLEMENTATION_CODE_SHA,
        expected_message=f"implementation_seal_sha256={IMPLEMENTATION_SEAL_SHA256}\n",
    )
    _validate_tag(
        repository,
        token=token,
        tag=REVIEW_TAG,
        expected_object_sha=REVIEW_TAG_OBJECT_SHA,
        expected_target_sha=IMPLEMENTATION_CODE_SHA,
    )
    _validate_source_artifact(
        repository,
        repository_id=repository_id,
        token=token,
    )
    _require_no_activation_tag(repository, token=token)

    activation_identity = content_digest(
        {
            "schema": "ppo_normalization_activation_identity_v1",
            "repository_id": repository_id,
            "request_sha256": request_sha,
            "protocol_sha256": PROTOCOL_SHA256,
            "implementation_digest": IMPLEMENTATION_DIGEST,
            "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        }
    )
    return {
        "execution_code_sha": head_sha,
        "request_sha256": request_sha,
        "activation_identity": activation_identity,
    }


def request_main(environment: dict[str, str] | None = None) -> int:
    env = dict(os.environ if environment is None else environment)
    token = env.get("GITHUB_TOKEN", "")
    event_path = Path(env.get("GITHUB_EVENT_PATH", ""))
    if not token:
        raise ValueError("GITHUB_TOKEN is required")
    if not event_path.is_file():
        raise ValueError("GITHUB_EVENT_PATH is required")
    try:
        event = json.loads(event_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError("GitHub execution request event is unreadable") from None
    if not isinstance(event, dict):
        raise ValueError("GitHub execution request event is malformed")
    for key, value in _request_outputs(event, token=token).items():
        print(f"{key}={value}")
    return 0


def _write_activation_authority(
    target_root: Path, activation: dict[str, object]
) -> str:
    path = target_root / "trade_rl/evaluation/ppo_normalization_activation.json"
    payload, raw = _canonical_file(path, field="committed activation authority")
    if hashlib.sha256(
        raw
    ).hexdigest() != NULL_ACTIVATION_RESOURCE_SHA256 or payload != {
        "schema": ACTIVATION_AUTHORITY_SCHEMA,
        "activation_sha256": None,
    }:
        raise ValueError("committed activation authority is not the sealed null state")
    authority = activation_authority_payload(activation)
    path.write_bytes(canonical_json_bytes(authority))
    return str(authority["activation_sha256"])


def _restore_activation_authority(
    target_root: Path,
    replication_root: Path,
    *,
    expected_digest: str,
) -> dict[str, object]:
    expected = _require_sha256(expected_digest, field="activation digest")
    activation, raw = _canonical_file(
        replication_root / "activation.json",
        field="replication activation",
    )
    if content_digest(activation) != expected:
        raise ValueError(
            "replication activation digest differs from workflow authority"
        )
    path = target_root / "trade_rl/evaluation/ppo_normalization_activation.json"
    path.write_bytes(
        canonical_json_bytes(
            {
                "schema": ACTIVATION_AUTHORITY_SCHEMA,
                "activation_sha256": expected,
            }
        )
    )
    return activation


def _create_activation_tag(
    repository: str,
    *,
    token: str,
    code_sha: str,
    request_sha256: str,
    activation_digest: str,
    run_id: int,
    run_attempt: int,
) -> str:
    _require_no_activation_tag(repository, token=token)
    record = {
        "schema": "ppo_normalization_repository_activation_v1",
        "execution_code_sha": _require_sha(code_sha, field="execution code SHA"),
        "request_sha256": _require_sha256(
            request_sha256, field="execution request SHA-256"
        ),
        "activation_sha256": _require_sha256(
            activation_digest, field="activation SHA-256"
        ),
        "implementation_digest": IMPLEMENTATION_DIGEST,
        "implementation_seal_sha256": IMPLEMENTATION_SEAL_SHA256,
        "fresh_reconstruction_sha256": FRESH_RECONSTRUCTION_SHA256,
        "assurance_review_sha256": ASSURANCE_REVIEW_SHA256,
        "run_id": _positive_int(run_id, field="workflow run id"),
        "run_attempt": _positive_int(run_attempt, field="workflow run attempt"),
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    repo = _repo_path(repository)
    tag_object = _api_json(
        f"https://api.github.com/repos/{repo}/git/tags",
        token=token,
        method="POST",
        payload={
            "tag": ACTIVATION_TAG,
            "message": canonical_json_bytes(record).decode("utf-8"),
            "object": code_sha,
            "type": "commit",
        },
    )
    if tag_object is None:
        raise TransportError("activation tag object was not created")
    tag_object_sha = _require_sha(
        tag_object.get("sha"), field="activation tag object SHA"
    )
    created_ref = _api_json(
        f"https://api.github.com/repos/{repo}/git/refs",
        token=token,
        method="POST",
        payload={"ref": f"refs/tags/{ACTIVATION_TAG}", "sha": tag_object_sha},
    )
    if (
        created_ref is None
        or not isinstance(created_ref.get("object"), dict)
        or created_ref["object"].get("sha") != tag_object_sha
    ):
        raise TransportError("repository-global activation tag ref was not created")
    return tag_object_sha


def _download_source(
    repository: str,
    *,
    repository_id: int,
    token: str,
    temp_root: Path,
) -> Path:
    transport = _checkpoint_transport()
    reference = _source_reference()
    archive = temp_root / "source.zip"
    deadline = time.monotonic() + 20 * 60
    transport.download_artifact_archive(
        repository,
        reference,
        repository_id=repository_id,
        token=token,
        destination=archive,
        deadline=deadline,
    )
    extracted = temp_root / "source"
    transport.extract_verified_archive(
        archive,
        extracted,
        SOURCE_ARTIFACT_SHA256,
    )
    return transport.find_source_root(extracted)


def _download_workflow_artifact_tree(
    repository: str,
    *,
    repository_id: int,
    token: str,
    artifact_id: int,
    run_id: int,
    sha256: str,
    temp_root: Path,
    label: str,
) -> Path:
    """Re-download and hash one same-run artifact before safe extraction."""

    transport = _checkpoint_transport()
    digest = _require_sha256(sha256, field=f"{label} artifact SHA-256")
    reference = ArtifactReference(
        artifact_id,
        run_id,
        digest,
    )
    archive = temp_root / f"{label}.zip"
    deadline = time.monotonic() + 20 * 60
    transport.download_artifact_archive(
        repository,
        reference,
        repository_id=repository_id,
        token=token,
        destination=archive,
        deadline=deadline,
    )
    extracted = temp_root / label
    transport.extract_verified_archive(
        archive,
        extracted,
        digest,
    )
    return extracted


def _read_request_from_checkout(
    target_root: Path,
    *,
    expected_code_sha: str,
    expected_request_sha: str,
) -> dict[str, object]:
    if _git_head(target_root) != expected_code_sha:
        raise ValueError("execution checkout HEAD differs from request")
    payload, raw = _canonical_file(
        target_root / REQUEST_PATH,
        field="committed execution request",
    )
    validate_execution_request(payload)
    if hashlib.sha256(raw).hexdigest() != expected_request_sha:
        raise ValueError("committed execution request SHA-256 differs")
    return payload


def execute_main(environment: dict[str, str] | None = None) -> int:
    env = dict(os.environ if environment is None else environment)
    repository = env.get("GITHUB_REPOSITORY", "")
    repository_id = _positive_int(
        int(env.get("GITHUB_REPOSITORY_ID", "0")), field="repository id"
    )
    run_id = _positive_int(int(env.get("GITHUB_RUN_ID", "0")), field="workflow run id")
    run_attempt = _positive_int(
        int(env.get("GITHUB_RUN_ATTEMPT", "0")), field="workflow run attempt"
    )
    token = env.get("GITHUB_TOKEN", "")
    code_sha = _require_sha(env.get("EXECUTION_CODE_SHA"), field="execution code SHA")
    request_sha = _require_sha256(
        env.get("REQUEST_SHA256"), field="execution request SHA-256"
    )
    target_root = Path(env.get("TARGET_ROOT", "target"))
    output_root = Path(env.get("OUTPUT_ROOT", "output/execution"))
    failure_path = output_root.parent / "execution-failure.json"
    replication_root = output_root / "replication"
    if not token:
        raise ValueError("GITHUB_TOKEN is required")

    phase = "preflight"
    current_slot: str | None = None
    activation_digest: str | None = None
    activation_tag_created = False
    try:
        if output_root.exists() or output_root.is_symlink():
            raise FileExistsError("execution output already exists")
        request = _read_request_from_checkout(
            target_root,
            expected_code_sha=code_sha,
            expected_request_sha=request_sha,
        )
        provenance = build_candidate_run_provenance()
        activation = build_execution_activation(request, provenance)
        activation_digest = content_digest(activation)

        phase = "activation"
        tag_object_sha = _create_activation_tag(
            repository,
            token=token,
            code_sha=code_sha,
            request_sha256=request_sha,
            activation_digest=activation_digest,
            run_id=run_id,
            run_attempt=run_attempt,
        )
        activation_tag_created = True
        written_digest = _write_activation_authority(target_root, activation)
        if written_digest != activation_digest:
            raise RuntimeError("written activation authority digest differs")

        output_root.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryDirectory(prefix="ppo-normalization-source-") as temp:
            phase = "source"
            source = _download_source(
                repository,
                repository_id=repository_id,
                token=token,
                temp_root=Path(temp),
            )
            phase = "prepare"
            prepare_replication_execution(replication_root, activation)
            phase = "slot"
            for spec in replication_arm_specs():
                current_slot = spec.slot
                execute_replication_slot(source, replication_root, spec.slot)

        phase = "complete-check"
        current_slot = None
        for spec in replication_arm_specs():
            state = replication_slot_state(replication_root, spec)
            if (
                state["consumed"] is not True
                or state["failed"] is True
                or state["result_published"] is not True
            ):
                raise ValueError(
                    "complete ten-slot execution evidence was not produced"
                )
        replication_tree_sha = _tree_digest(replication_root)
        _write_canonical_once(
            output_root / "transport.json",
            {
                "schema": "ppo_normalization_execution_transport_v1",
                "repository": repository,
                "repository_id": repository_id,
                "run_id": run_id,
                "run_attempt": run_attempt,
                "execution_code_sha": code_sha,
                "request_sha256": request_sha,
                "activation_sha256": activation_digest,
                "activation_tag": ACTIVATION_TAG,
                "activation_tag_object_sha": tag_object_sha,
                "source_artifact": request["source_artifact"],
                "replication_tree_sha256": replication_tree_sha,
                "all_slots_complete": True,
                "economic_result_inspected": False,
                "unused_data_accessed": False,
                "final_test_accessed": False,
                "production_eligible": False,
                "live_trading_authorized": False,
            },
        )
        print(f"activation_digest={activation_digest}")
        print(
            "execution_artifact_name="
            f"ppo-normalization-execution-{activation_digest}-{run_id}"
        )
        return 0
    except Exception:
        slot_consumed: bool | None = None
        if current_slot is not None and replication_root.exists():
            try:
                spec = next(
                    spec
                    for spec in replication_arm_specs()
                    if spec.slot == current_slot
                )
                state = replication_slot_state(replication_root, spec)
                consumed = state.get("consumed")
                if isinstance(consumed, bool):
                    slot_consumed = consumed
            except Exception:
                slot_consumed = None
        try:
            _write_canonical_once(
                failure_path,
                build_execution_failure_receipt(
                    repository=repository,
                    repository_id=repository_id,
                    run_id=run_id,
                    run_attempt=run_attempt,
                    execution_code_sha=code_sha,
                    request_sha256=request_sha,
                    activation_sha256=activation_digest,
                    activation_tag_created=activation_tag_created,
                    phase=phase,
                    slot=current_slot,
                    slot_consumed=slot_consumed,
                ),
            )
        except Exception:
            pass
        raise


def _validate_transport_tree(
    artifact_root: Path,
    *,
    expected_activation_digest: str,
) -> Path:
    receipt, _raw = _canonical_file(
        artifact_root / "transport.json",
        field="execution transport receipt",
    )
    replication_root = artifact_root / "replication"
    if (
        receipt.get("activation_sha256") != expected_activation_digest
        or receipt.get("all_slots_complete") is not True
        or receipt.get("replication_tree_sha256") != _tree_digest(replication_root)
    ):
        raise ValueError("execution transport receipt differs from evidence tree")
    return replication_root


def _verification_set_digest(replication_root: Path) -> str:
    records: list[dict[str, str]] = []
    for spec in replication_arm_specs():
        path = replication_root / "slots" / spec.slot / "verified.json"
        payload, raw = _canonical_file(path, field="slot verification evidence")
        if payload.get("slot") != spec.slot:
            raise ValueError("verification set slot identity differs")
        records.append({"slot": spec.slot, "sha256": hashlib.sha256(raw).hexdigest()})
    return content_digest(
        {
            "schema": "ppo_normalization_replication_verification_set_v1",
            "records": records,
        }
    )


def verify_main(environment: dict[str, str] | None = None) -> int:
    env = dict(os.environ if environment is None else environment)
    repository = env.get("GITHUB_REPOSITORY", "")
    repository_id = _positive_int(
        int(env.get("GITHUB_REPOSITORY_ID", "0")), field="repository id"
    )
    run_id = _positive_int(int(env.get("GITHUB_RUN_ID", "0")), field="workflow run id")
    token = env.get("GITHUB_TOKEN", "")
    code_sha = _require_sha(env.get("EXECUTION_CODE_SHA"), field="execution code SHA")
    activation_digest = _require_sha256(
        env.get("ACTIVATION_DIGEST"), field="activation digest"
    )
    execution_artifact_id = _positive_int(
        int(env.get("EXECUTION_ARTIFACT_ID", "0")), field="execution artifact id"
    )
    execution_artifact_sha = _require_sha256(
        env.get("EXECUTION_ARTIFACT_SHA256"), field="execution artifact SHA-256"
    )
    target_root = Path(env.get("TARGET_ROOT", "target"))
    output_root = Path(env.get("OUTPUT_ROOT", "output/verification"))
    if not token:
        raise ValueError("GITHUB_TOKEN is required")
    if _git_head(target_root) != code_sha:
        raise ValueError("verifier checkout HEAD differs from execution authority")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("verification output already exists")
    with tempfile.TemporaryDirectory(
        prefix="ppo-normalization-execution-artifact-"
    ) as temp:
        artifact_root = _download_workflow_artifact_tree(
            repository,
            repository_id=repository_id,
            token=token,
            artifact_id=execution_artifact_id,
            run_id=run_id,
            sha256=execution_artifact_sha,
            temp_root=Path(temp),
            label="execution",
        )
        _copy_tree(artifact_root, output_root)
    replication_root = _validate_transport_tree(
        output_root,
        expected_activation_digest=activation_digest,
    )
    _restore_activation_authority(
        target_root,
        replication_root,
        expected_digest=activation_digest,
    )
    with tempfile.TemporaryDirectory(prefix="ppo-normalization-verify-source-") as temp:
        source = _download_source(
            repository,
            repository_id=repository_id,
            token=token,
            temp_root=Path(temp),
        )
        for spec in replication_arm_specs():
            verify_replication_slot(source, replication_root, spec.slot)
    verification_set_sha = _verification_set_digest(replication_root)
    _write_canonical_once(
        output_root / "verification-transport.json",
        {
            "schema": "ppo_normalization_verification_transport_v1",
            "repository": repository,
            "repository_id": repository_id,
            "run_id": run_id,
            "execution_code_sha": code_sha,
            "activation_sha256": activation_digest,
            "execution_artifact_id": execution_artifact_id,
            "execution_artifact_sha256": execution_artifact_sha,
            "verification_set_sha256": verification_set_sha,
            "replication_tree_sha256": _tree_digest(replication_root),
            "all_slots_verified": True,
            "no_refit": True,
        },
    )
    print(f"verification_set_sha256={verification_set_sha}")
    print(
        "verification_artifact_name="
        f"ppo-normalization-verification-{activation_digest}-{run_id}"
    )
    return 0


def _validate_verification_transport_tree(
    artifact_root: Path,
    *,
    expected_activation_digest: str,
    expected_verification_set_sha256: str,
) -> Path:
    receipt, _raw = _canonical_file(
        artifact_root / "verification-transport.json",
        field="verification transport receipt",
    )
    replication_root = artifact_root / "replication"
    expected_set = _require_sha256(
        expected_verification_set_sha256,
        field="verification set SHA-256",
    )
    if (
        receipt.get("schema") != "ppo_normalization_verification_transport_v1"
        or receipt.get("activation_sha256") != expected_activation_digest
        or receipt.get("verification_set_sha256") != expected_set
        or receipt.get("all_slots_verified") is not True
        or receipt.get("no_refit") is not True
        or receipt.get("replication_tree_sha256") != _tree_digest(replication_root)
        or _verification_set_digest(replication_root) != expected_set
    ):
        raise ValueError("verification transport receipt differs from evidence tree")
    return replication_root


def finalize_main(environment: dict[str, str] | None = None) -> int:
    env = dict(os.environ if environment is None else environment)
    repository = env.get("GITHUB_REPOSITORY", "")
    repository_id = _positive_int(
        int(env.get("GITHUB_REPOSITORY_ID", "0")), field="repository id"
    )
    run_id = _positive_int(int(env.get("GITHUB_RUN_ID", "0")), field="workflow run id")
    token = env.get("GITHUB_TOKEN", "")
    code_sha = _require_sha(env.get("EXECUTION_CODE_SHA"), field="execution code SHA")
    workflow_sha = _require_sha(env.get("GITHUB_WORKFLOW_SHA"), field="workflow SHA")
    activation_digest = _require_sha256(
        env.get("ACTIVATION_DIGEST"), field="activation digest"
    )
    verification_artifact_id = _positive_int(
        int(env.get("VERIFICATION_ARTIFACT_ID", "0")),
        field="verification artifact id",
    )
    verification_artifact_sha = _require_sha256(
        env.get("VERIFICATION_ARTIFACT_SHA256"),
        field="verification artifact SHA-256",
    )
    verification_set_sha = _require_sha256(
        env.get("VERIFICATION_SET_SHA256"),
        field="verification set SHA-256",
    )
    target_root = Path(env.get("TARGET_ROOT", "target"))
    output_root = Path(env.get("OUTPUT_ROOT", "output/final"))
    if not token:
        raise ValueError("GITHUB_TOKEN is required")
    if _git_head(target_root) != code_sha:
        raise ValueError("finalizer checkout HEAD differs from execution authority")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("final output already exists")
    with tempfile.TemporaryDirectory(
        prefix="ppo-normalization-verification-artifact-"
    ) as temp:
        artifact_root = _download_workflow_artifact_tree(
            repository,
            repository_id=repository_id,
            token=token,
            artifact_id=verification_artifact_id,
            run_id=run_id,
            sha256=verification_artifact_sha,
            temp_root=Path(temp),
            label="verification",
        )
        _copy_tree(artifact_root, output_root)
    replication_root = _validate_verification_transport_tree(
        output_root,
        expected_activation_digest=activation_digest,
        expected_verification_set_sha256=verification_set_sha,
    )
    _restore_activation_authority(
        target_root,
        replication_root,
        expected_digest=activation_digest,
    )
    authority = build_verifier_artifact_authority(
        repository_id=repository_id,
        run_id=run_id,
        artifact_id=verification_artifact_id,
        artifact_sha256=verification_artifact_sha,
        code_sha=code_sha,
        workflow_sha=workflow_sha,
        activation_digest=activation_digest,
        implementation_digest=IMPLEMENTATION_DIGEST,
        verification_set_sha256=verification_set_sha,
    )
    _write_canonical_once(replication_root / "verifier-authority.json", authority)
    _write_canonical_once(
        replication_root / "verifier-authority.sha256.json",
        {"sha256": hashlib.sha256(canonical_json_bytes(authority)).hexdigest()},
    )
    publish_replication_decision(replication_root)
    _write_canonical_once(
        output_root / "final-transport.json",
        {
            "schema": "ppo_normalization_final_transport_v1",
            "repository": repository,
            "repository_id": repository_id,
            "run_id": run_id,
            "execution_code_sha": code_sha,
            "activation_sha256": activation_digest,
            "verification_artifact_id": verification_artifact_id,
            "verification_artifact_sha256": verification_artifact_sha,
            "verification_set_sha256": verification_set_sha,
            "comparison_published": True,
            "production_eligible": False,
            "live_trading_authorized": False,
        },
    )
    print(f"final_artifact_name=ppo-normalization-final-{activation_digest}-{run_id}")
    return 0


def render_request_main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise ValueError("render-request requires OUTPUT_PATH")
    output = Path(argv[0])
    _write_canonical_once(output, build_execution_request())
    print(hashlib.sha256(output.read_bytes()).hexdigest())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args == ["request"]:
            return request_main()
        if args == ["execute"]:
            return execute_main()
        if args == ["verify"]:
            return verify_main()
        if args == ["finalize"]:
            return finalize_main()
        if args and args[0] == "render-request":
            return render_request_main(args[1:])
        raise ValueError("unsupported PPO normalization one-shot action")
    except Exception as error:
        print(
            f"PPO normalization one-shot action failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


__all__ = [
    "ACTIVATION_TAG",
    "ASSURANCE_REVIEW_SHA256",
    "FRESH_RECONSTRUCTION_SHA256",
    "IMPLEMENTATION_DIGEST",
    "IMPLEMENTATION_SEAL_SHA256",
    "PROTOCOL_SHA256",
    "REVIEW_REQUEST_MARKER",
    "SOURCE_ARTIFACT_ID",
    "SOURCE_ARTIFACT_RUN_ID",
    "SOURCE_ARTIFACT_SHA256",
    "activation_authority_payload",
    "build_execution_activation",
    "build_execution_failure_receipt",
    "build_execution_request",
    "build_verifier_artifact_authority",
    "content_digest",
    "parse_execution_request_comment",
    "validate_execution_pr_files",
    "validate_execution_request",
]


if __name__ == "__main__":
    raise SystemExit(main())
