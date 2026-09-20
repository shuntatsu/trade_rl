"""Manual GitHub Actions transport for the PPO feature checkpoint CLI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.artifacts.verified_file import file_digest_and_size, open_regular_binary

SOURCE_ARTIFACT_ID = 10_331_899_302
SOURCE_ARTIFACT_RUN_ID = 34_803_217_815
SOURCE_ARTIFACT_NAME = "issue539-calibrated-causal-successor-v3-34803217815"
SOURCE_ARTIFACT_SHA256 = (
    "89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce"
)
WORKFLOW_PATH = ".github/workflows/ppo-feature-checkpoint.yml"
RECEIPT_SCHEMA = "ppo_feature_checkpoint_actions_receipt_v2"
REVIEW_EVIDENCE_SCHEMA = "ppo_feature_review_evidence_v1"
REVIEW_EVIDENCE_MARKER = "<!-- ppo-feature-review-evidence-v1 -->\n"
CHECKPOINT_ARTIFACT_PREFIX = "ppo-feature-checkpoint"
TOTAL_DEADLINE_SECONDS = 300 * 60
START_MINIMUM_AVAILABLE_BYTES = 4 * 1024**3
STOP_MINIMUM_AVAILABLE_BYTES = int(1.5 * 1024**3)
MAX_ARCHIVE_BYTES = 6 * 1024**3
MAX_ARCHIVE_EXPANDED_BYTES = 8 * 1024**3
MAX_ARCHIVE_MEMBER_BYTES = 6 * 1024**3
MAX_ARCHIVE_MEMBERS = 100_000
MAX_CHILD_DIAGNOSTIC_BYTES = 384
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_REPO_PART_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_REVIEW_REFERENCE_RE = re.compile(
    r"^https://github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/"
    r"pull/(?P<pull>[1-9][0-9]*)#issuecomment-(?P<comment>[1-9][0-9]*)$"
)
_STAGE_NAMES = {"prepare", "arm", "finalize"}


class TransportError(RuntimeError):
    """A transport failure with a secret-free message suitable for receipts."""


@dataclass(frozen=True)
class ArtifactReference:
    artifact_id: int
    run_id: int
    sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.artifact_id,
            "run_id": self.run_id,
            "sha256": self.sha256,
        }


def _strict_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_commit_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a full 40-character commit SHA")
    return value


def parse_artifact_references(raw: str) -> tuple[ArtifactReference, ...]:
    """Parse exact, same-repository artifact references from workflow input JSON."""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint artifact input must be a JSON array") from error
    if not isinstance(value, list):
        raise ValueError("checkpoint artifact input must be a JSON array")
    references: list[ArtifactReference] = []
    seen: set[tuple[int, int]] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "run_id", "sha256"}:
            raise ValueError("checkpoint artifact reference has an unexpected shape")
        artifact_id = _strict_positive_int(item["id"], field="artifact id")
        run_id = _strict_positive_int(item["run_id"], field="artifact run id")
        digest = _require_sha256(item["sha256"], field="artifact SHA-256")
        key = artifact_id, run_id
        if key in seen:
            raise ValueError("checkpoint artifact input contains a duplicate reference")
        seen.add(key)
        references.append(ArtifactReference(artifact_id, run_id, digest))
    return tuple(references)


def parse_review_reference(
    review_reference: str,
    *,
    repository: str,
) -> tuple[int, int]:
    """Parse one exact same-repository pull-request issue-comment URL."""
    if not isinstance(review_reference, str):
        raise ValueError("review reference must be a GitHub pull-request comment URL")
    match = _REVIEW_REFERENCE_RE.fullmatch(review_reference)
    if match is None:
        raise ValueError(
            "review reference must be an exact GitHub PR issue-comment URL"
        )
    expected_owner, expected_repo = _repo_path(repository).split("/", 1)
    if match.group("owner") != expected_owner or match.group("repo") != expected_repo:
        raise ValueError("review reference belongs to another repository")
    return int(match.group("pull")), int(match.group("comment"))


def validate_review_evidence_comment(
    comment: object,
    pull: object,
    *,
    review_reference: str,
    review_record_sha256: str,
    repository: str,
    repository_id: int,
    code_sha: str,
    workflow_sha: str,
    protocol_digest: str,
    prepare_reference: ArtifactReference,
) -> dict[str, Any]:
    """Validate one GitHub-bound, result-blind review evidence record."""
    pull_number, comment_id = parse_review_reference(
        review_reference,
        repository=repository,
    )
    expected_review_sha = _require_sha256(
        review_record_sha256,
        field="review record SHA-256",
    )
    expected_code_sha = _require_commit_sha(code_sha, field="review code SHA")
    expected_workflow_sha = _require_commit_sha(
        workflow_sha,
        field="review workflow SHA",
    )
    expected_protocol = _require_sha256(
        protocol_digest,
        field="review protocol digest",
    )
    expected_repository_id = _strict_positive_int(
        repository_id,
        field="review repository id",
    )
    if not isinstance(comment, dict) or not isinstance(pull, dict):
        raise ValueError("review evidence GitHub records are malformed")
    expected_issue_url = (
        f"https://api.github.com/repos/{_repo_path(repository)}/issues/{pull_number}"
    )
    if (
        comment.get("html_url") != review_reference
        or comment.get("issue_url") != expected_issue_url
    ):
        raise ValueError("review evidence comment identity differs")
    body = comment.get("body")
    if not isinstance(body, str):
        raise ValueError("review evidence comment body is malformed")
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != expected_review_sha:
        raise ValueError("review evidence comment body SHA-256 differs")
    head = pull.get("head")
    if (
        not isinstance(head, dict)
        or head.get("sha") != expected_code_sha
        or not isinstance(head.get("repo"), dict)
        or _strict_positive_int(
            head["repo"].get("id"),
            field="review pull head repository id",
        )
        != expected_repository_id
    ):
        raise ValueError("review evidence pull request is not bound to the code SHA")

    if not body.startswith(REVIEW_EVIDENCE_MARKER) or not body.endswith("\n"):
        raise ValueError("review evidence body does not use the canonical marker")
    raw_payload = body[len(REVIEW_EVIDENCE_MARKER) : -1].encode("utf-8")
    try:
        payload = json.loads(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("review evidence payload is not valid JSON") from error
    expected_keys = {
        "schema",
        "outcome",
        "code_sha",
        "workflow_sha",
        "protocol_digest",
        "prepare_artifact",
        "result_blind",
        "economic_execution_authorized",
        "unused_data_accessed",
        "final_data_accessed",
        "source_review_reference",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or canonical_json_bytes(payload) != raw_payload
        or payload.get("schema") != REVIEW_EVIDENCE_SCHEMA
        or payload.get("outcome") != "G0_G1_CLEAR_G2_EVIDENCE_BOUND"
        or payload.get("code_sha") != expected_code_sha
        or payload.get("workflow_sha") != expected_workflow_sha
        or payload.get("protocol_digest") != expected_protocol
        or payload.get("prepare_artifact") != prepare_reference.to_mapping()
        or payload.get("result_blind") is not True
        or payload.get("economic_execution_authorized") is not True
        or payload.get("unused_data_accessed") is not False
        or payload.get("final_data_accessed") is not False
    ):
        raise ValueError("review evidence payload differs from the execution contract")
    source_reference = payload.get("source_review_reference")
    source_pull, source_comment = parse_review_reference(
        source_reference,
        repository=repository,
    )
    if source_pull != pull_number or source_comment == comment_id:
        raise ValueError("review evidence source review reference is malformed")
    return payload


def validate_operator_approval(
    stage: str,
    approved_protocol_digest: str,
    review_reference: str,
    actual_protocol_digest: str | None,
    *,
    repository: str | None = None,
    review_record_sha256: str | None = None,
) -> str | None:
    """Validate operator-selected protocol and immutable review-record identity."""
    if stage not in _STAGE_NAMES:
        raise ValueError("checkpoint stage is not supported")
    if stage == "prepare":
        return None
    approved = _require_sha256(
        approved_protocol_digest, field="approved protocol digest"
    )
    if not isinstance(actual_protocol_digest, str):
        raise ValueError("approved protocol digest cannot be checked before prepare")
    actual = _require_sha256(actual_protocol_digest, field="checkpoint protocol digest")
    if approved != actual:
        raise ValueError("approved protocol digest does not match the checkpoint")
    if repository is None:
        raise ValueError("review reference repository is required")
    parse_review_reference(review_reference, repository=repository)
    _require_sha256(review_record_sha256, field="review record SHA-256")
    return review_reference


def validate_artifact_metadata(
    metadata: object,
    reference: ArtifactReference,
    *,
    repository_id: int,
) -> dict[str, Any]:
    """Bind artifact API metadata to the exact repository, run and SHA input."""
    if not isinstance(metadata, dict):
        raise ValueError("GitHub artifact metadata is malformed")
    if _strict_positive_int(metadata.get("id"), field="artifact metadata id") != (
        reference.artifact_id
    ):
        raise ValueError("GitHub artifact id differs from the requested reference")
    if metadata.get("expired") is not False:
        raise ValueError("GitHub artifact is expired or has no expiry status")
    name = metadata.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("GitHub artifact name is malformed")
    run = metadata.get("workflow_run")
    if not isinstance(run, dict):
        raise ValueError("GitHub artifact workflow run is missing")
    if _strict_positive_int(run.get("id"), field="artifact workflow run id") != (
        reference.run_id
    ):
        raise ValueError("GitHub artifact belongs to a different workflow run")
    for field in ("repository_id", "head_repository_id"):
        if _strict_positive_int(run.get(field), field=f"workflow {field}") != (
            repository_id
        ):
            raise ValueError("GitHub artifact does not belong to this repository")
    api_digest = metadata.get("digest")
    if api_digest != f"sha256:{reference.sha256}":
        raise ValueError("GitHub artifact API digest differs from the requested SHA")
    return metadata


def validate_prior_receipt(
    receipt: object,
    reference: ArtifactReference,
    metadata: dict[str, Any],
    identity: dict[str, Any],
    *,
    protocol_digest: str | None,
    approved_protocol_digest: str | None = None,
    review_reference: str | None = None,
    review_record_sha256: str | None = None,
    runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Require every imported checkpoint ZIP to carry a matching run receipt."""
    fields = {
        "schema",
        "stage",
        "status",
        "repository",
        "repository_id",
        "run_id",
        "run_attempt",
        "artifact_name",
        "code_sha",
        "workflow",
        "workflow_ref",
        "workflow_sha",
        "github_sha",
        "workflow_sha256",
        "uv_lock_sha256",
        "pyproject_sha256",
        "checkpoint_protocol_digest",
        "approved_protocol_digest",
        "review_reference",
        "review_record_sha256",
        "input_artifacts",
        "source_artifact",
        "python_version",
        "uv_version",
        "started_at",
        "finished_at",
        "completed_commands",
        "error_type",
        "error_message",
    }
    if not isinstance(receipt, dict) or set(receipt) != fields:
        raise ValueError("checkpoint artifact receipt is missing or malformed")
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise ValueError("checkpoint artifact receipt schema is unsupported")
    if receipt["stage"] not in _STAGE_NAMES:
        raise ValueError("checkpoint artifact receipt stage is invalid")
    if receipt["status"] not in {"started", "failed", "succeeded"}:
        raise ValueError("checkpoint artifact receipt status is invalid")
    expected_identity = (
        "repository",
        "repository_id",
        "code_sha",
        "workflow",
        "workflow_ref",
        "workflow_sha",
        "github_sha",
    )
    for field in expected_identity:
        expected = identity.get(field)
        if field.endswith("sha") or field == "code_sha":
            _require_commit_sha(expected, field=f"current {field}")
        if receipt.get(field) != expected:
            label = "workflow identity" if field.startswith("workflow") else field
            raise ValueError(f"checkpoint artifact receipt {label} differs")
    if (
        _strict_positive_int(
            receipt.get("repository_id"), field="receipt repository id"
        )
        != _strict_positive_int(identity.get("repository_id"), field="repository id")
        or _strict_positive_int(receipt.get("run_id"), field="receipt run id")
        != reference.run_id
    ):
        raise ValueError("checkpoint artifact receipt run/repository differs")
    attempt = _strict_positive_int(
        receipt.get("run_attempt"), field="receipt run attempt"
    )
    artifact_name = f"{CHECKPOINT_ARTIFACT_PREFIX}-{reference.run_id}-{attempt}"
    if (
        receipt.get("artifact_name") != artifact_name
        or metadata.get("name") != artifact_name
    ):
        raise ValueError("checkpoint artifact name differs from its receipt")
    expected_hashes = {
        "workflow_sha256": Path(".github/workflows/ppo-feature-checkpoint.yml"),
        "uv_lock_sha256": Path("uv.lock"),
        "pyproject_sha256": Path("pyproject.toml"),
    }
    for field, path in expected_hashes.items():
        value = _require_sha256(receipt.get(field), field=f"receipt {field}")
        if not path.is_file() or path.is_symlink() or value != _sha256_file(path):
            raise ValueError(
                f"checkpoint receipt {field} differs from current checkout"
            )
    workflow_run = metadata.get("workflow_run")
    if not isinstance(workflow_run, dict) or workflow_run.get(
        "head_sha"
    ) != receipt.get("github_sha"):
        raise ValueError("checkpoint artifact API workflow head differs from receipt")
    if not isinstance(receipt.get("source_artifact"), dict) or receipt[
        "source_artifact"
    ] != {
        "id": SOURCE_ARTIFACT_ID,
        "run_id": SOURCE_ARTIFACT_RUN_ID,
        "name": SOURCE_ARTIFACT_NAME,
        "sha256": SOURCE_ARTIFACT_SHA256,
    }:
        raise ValueError("checkpoint receipt source artifact identity differs")
    current_runtime = runtime if runtime is not None else _runtime_provenance()
    if receipt.get("python_version") != current_runtime["python_version"]:
        raise ValueError("checkpoint receipt Python runtime differs")
    if receipt.get("uv_version") != current_runtime["uv_version"]:
        raise ValueError("checkpoint receipt uv version differs")
    for field in ("started_at",):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            raise ValueError(f"checkpoint receipt {field} is malformed")
    finished_value = receipt.get("finished_at")
    if finished_value is not None and not isinstance(finished_value, str):
        raise ValueError("checkpoint receipt finished_at is malformed")
    if not isinstance(receipt.get("completed_commands"), list):
        raise ValueError("checkpoint receipt command history is malformed")
    finished_at = receipt.get("finished_at")
    if receipt.get("status") == "started" and finished_at is not None:
        raise ValueError("started checkpoint receipt has a finish time")
    if receipt.get("status") != "started" and not isinstance(finished_at, str):
        raise ValueError("finished checkpoint receipt has no finish time")
    if receipt.get("error_type") is not None and not isinstance(
        receipt.get("error_type"), str
    ):
        raise ValueError("checkpoint receipt error type is malformed")
    if receipt.get("error_message") is not None and not isinstance(
        receipt.get("error_message"), str
    ):
        raise ValueError("checkpoint receipt error message is malformed")
    if protocol_digest is not None:
        expected_digest = _require_sha256(
            protocol_digest, field="restored checkpoint protocol digest"
        )
        if receipt.get("checkpoint_protocol_digest") != expected_digest:
            raise ValueError("checkpoint artifact receipt protocol digest differs")
    receipt_digest = receipt.get("checkpoint_protocol_digest")
    if receipt_digest is not None:
        _require_sha256(receipt_digest, field="receipt checkpoint protocol digest")
    receipt_approval = receipt.get("approved_protocol_digest")
    receipt_review_reference = receipt.get("review_reference")
    receipt_review_sha = receipt.get("review_record_sha256")
    if receipt["stage"] == "prepare":
        if (
            receipt_approval is not None
            or receipt_review_reference is not None
            or receipt_review_sha is not None
        ):
            raise ValueError("prepare receipt cannot claim review approval")
    else:
        _require_sha256(receipt_approval, field="receipt approved protocol digest")
        if receipt_approval != receipt_digest:
            raise ValueError("economic receipt approval differs from its protocol")
        parse_review_reference(
            receipt_review_reference,
            repository=identity["repository"],
        )
        _require_sha256(
            receipt_review_sha,
            field="receipt review record SHA-256",
        )
        if approved_protocol_digest is not None and receipt_approval != (
            approved_protocol_digest
        ):
            raise ValueError("checkpoint receipt uses a different approved protocol")
        if review_reference is not None and receipt_review_reference != (
            review_reference
        ):
            raise ValueError("checkpoint receipt uses a different review reference")
        if review_record_sha256 is not None and receipt_review_sha != (
            review_record_sha256
        ):
            raise ValueError("checkpoint receipt uses a different review record")
    input_artifacts = receipt.get("input_artifacts")
    if not isinstance(input_artifacts, list):
        raise ValueError("checkpoint artifact receipt input roster is malformed")
    parse_artifact_references(json.dumps(input_artifacts))
    return receipt


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        response: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        return None


def _api_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirectHandler())


def _repo_path(repository: str) -> str:
    parts = repository.split("/")
    if len(parts) != 2 or any(
        _SAFE_REPO_PART_RE.fullmatch(part) is None for part in parts
    ):
        raise ValueError("GITHUB_REPOSITORY is malformed")
    return "/".join(urllib.parse.quote(part, safe="") for part in parts)


def _api_json(url: str, *, token: str, deadline: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "trade-rl-ppo-feature-checkpoint",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with _api_opener().open(request, timeout=60) as response:
            if response.status != 200:
                raise TransportError("GitHub API returned an unexpected status")
            raw = response.read(2 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as error:
        raise TransportError(
            f"GitHub API request failed with HTTP {error.code}"
        ) from None
    except (OSError, urllib.error.URLError, TimeoutError):
        raise TransportError("GitHub API request could not be completed") from None
    if time.monotonic() >= deadline:
        raise TimeoutError(
            "checkpoint command budget expired during GitHub API request"
        )
    if len(raw) > 2 * 1024 * 1024:
        raise TransportError("GitHub API response exceeds the metadata limit")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise TransportError("GitHub API returned malformed JSON") from None
    if not isinstance(value, dict):
        raise TransportError("GitHub API response is not an object")
    return value


def _download_response(url: str, *, token: str, deadline: float) -> Any:
    current_url = url
    include_auth = True
    for _ in range(4):
        headers = {"User-Agent": "trade-rl-ppo-feature-checkpoint"}
        if include_auth:
            headers.update(
                {
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                }
            )
        request = urllib.request.Request(current_url, headers=headers)
        try:
            response = _api_opener().open(request, timeout=60)
        except urllib.error.HTTPError as error:
            if error.code not in {301, 302, 303, 307, 308}:
                raise TransportError(
                    f"GitHub artifact download failed with HTTP {error.code}"
                ) from None
            location = error.headers.get("Location")
            error.close()
            if not isinstance(location, str):
                raise TransportError("GitHub artifact redirect has no location")
            parsed = urllib.parse.urlsplit(location)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise TransportError("GitHub artifact redirect is not safe HTTPS")
            current_url = location
            include_auth = False
            continue
        except (OSError, urllib.error.URLError, TimeoutError):
            raise TransportError(
                "GitHub artifact download could not be completed"
            ) from None
        if response.status != 200:
            response.close()
            raise TransportError(
                "GitHub artifact download returned an unexpected status"
            )
        if time.monotonic() >= deadline:
            response.close()
            raise TimeoutError(
                "checkpoint command budget expired during artifact download"
            )
        return response
    raise TransportError("GitHub artifact download exceeded the redirect limit")


def download_artifact_archive(
    repository: str,
    reference: ArtifactReference,
    *,
    repository_id: int,
    token: str,
    destination: Path,
    deadline: float,
) -> dict[str, Any]:
    """Download one artifact with metadata and streaming raw-ZIP SHA checks."""
    if not token:
        raise ValueError("CHECKPOINT_GITHUB_TOKEN is required")
    encoded_repo = _repo_path(repository)
    metadata_url = f"https://api.github.com/repos/{encoded_repo}/actions/artifacts/{reference.artifact_id}"
    metadata = validate_artifact_metadata(
        _api_json(metadata_url, token=token, deadline=deadline),
        reference,
        repository_id=repository_id,
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("artifact archive destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("artifact archive parent must not be a symlink")
    download_url = f"{metadata_url}/zip"
    response = _download_response(download_url, token=token, deadline=deadline)
    digest = hashlib.sha256()
    size = 0
    try:
        with destination.open("xb") as output:
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "checkpoint command budget expired during artifact download"
                    )
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    raise TransportError(
                        "GitHub artifact archive exceeds the size limit"
                    )
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        response.close()
        raise
    response.close()
    actual_digest = digest.hexdigest()
    if actual_digest != reference.sha256:
        raise ValueError("downloaded artifact archive SHA-256 mismatch")
    if size == 0:
        raise ValueError("downloaded artifact archive is empty")
    return metadata


def _member_path(name: str) -> tuple[str, ...]:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or name.startswith("/")
        or PureWindowsPath(name).drive
        or PureWindowsPath(name).is_absolute()
    ):
        raise ValueError("ZIP member path is absolute or malformed")
    trimmed = name[:-1] if name.endswith("/") else name
    parts = tuple(trimmed.split("/"))
    if not trimmed or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("ZIP member path contains an unsafe component")
    return parts


def _checked_archive_members(
    archive: zipfile.ZipFile,
) -> list[tuple[zipfile.ZipInfo, tuple[str, ...], bool]]:
    infos = archive.infolist()
    if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
        raise ValueError("ZIP member roster is empty or exceeds the limit")
    members: list[tuple[zipfile.ZipInfo, tuple[str, ...], bool]] = []
    entry_types: dict[tuple[str, ...], str] = {}
    expanded_size = 0
    for info in infos:
        parts = _member_path(info.orig_filename)
        is_directory = info.is_dir()
        mode = (info.external_attr >> 16) & 0xFFFF
        unix_kind = stat.S_IFMT(mode)
        if unix_kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError("ZIP archive contains a symlink or special file")
        if unix_kind == stat.S_IFLNK:
            raise ValueError("ZIP archive contains a symlink")
        if info.flag_bits & 0x1:
            raise ValueError("encrypted ZIP members are not accepted")
        if is_directory and unix_kind == stat.S_IFREG:
            raise ValueError("ZIP directory entry has a regular-file mode")
        if not is_directory and unix_kind == stat.S_IFDIR:
            raise ValueError("ZIP file entry has a directory mode")
        if info.file_size < 0 or info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError("ZIP member exceeds the uncompressed size limit")
        expanded_size += info.file_size
        if expanded_size > MAX_ARCHIVE_EXPANDED_BYTES:
            raise ValueError("ZIP archive exceeds the expanded size limit")
        if parts in entry_types:
            raise ValueError("ZIP archive contains duplicate normalized paths")
        entry_types[parts] = "directory" if is_directory else "file"
        members.append((info, parts, is_directory))
    for parts, entry_kind in entry_types.items():
        if entry_kind == "file" and any(
            parent in entry_types and entry_types[parent] == "file"
            for parent in (parts[:index] for index in range(1, len(parts)))
        ):
            raise ValueError("ZIP archive contains a file/directory collision")
        if entry_kind == "file" and any(
            candidate[: len(parts)] == parts
            for candidate in entry_types
            if len(candidate) > len(parts)
        ):
            raise ValueError("ZIP archive contains a file/directory collision")
        for index in range(1, len(parts)):
            parent = parts[:index]
            if parent in entry_types and entry_types[parent] == "file":
                raise ValueError("ZIP archive contains a file/directory collision")
    return members


def _assert_real_directory_path(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("archive extraction path contains a symlink")
        if current.exists() and not current.is_dir():
            raise ValueError("archive extraction parent is not a directory")


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    """Validate a ZIP completely, then extract regular files into a new tree."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("ZIP extraction destination already exists")
    parent = destination.parent
    _assert_real_directory_path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".{destination.name}.extract-{uuid.uuid4().hex}"
    if stage.exists() or stage.is_symlink():
        raise FileExistsError("ZIP extraction staging path already exists")
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = _checked_archive_members(archive)
            stage.mkdir()
            for info, parts, is_directory in members:
                target = stage.joinpath(*parts)
                if is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                _assert_real_directory_path(target.parent)
                copied = 0
                with archive.open(info, "r") as source, target.open("xb") as output:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > info.file_size:
                            raise ValueError(
                                "ZIP member expanded beyond its declared size"
                            )
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if copied != info.file_size:
                    raise ValueError(
                        "ZIP member size differs from its central directory"
                    )
        stage.rename(destination)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, OSError) as error:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise ValueError("ZIP archive could not be safely extracted") from error
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise


def extract_verified_archive(
    archive_path: Path,
    destination: Path,
    expected_sha256: str,
) -> None:
    """Hash raw archive bytes before opening any member for extraction."""
    expected = _require_sha256(expected_sha256, field="archive SHA-256")
    actual, size = file_digest_and_size(archive_path, field="downloaded archive")
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise ValueError("downloaded archive size is outside the supported range")
    if actual != expected:
        raise ValueError("archive SHA-256 mismatch")
    safe_extract_zip(archive_path, destination)


def _file_bytes_equal(left: Path, right: Path) -> bool:
    with (
        open_regular_binary(left, field="left checkpoint file") as left_stream,
        open_regular_binary(right, field="right checkpoint file") as right_stream,
    ):
        while True:
            left_chunk = left_stream.read(1024 * 1024)
            right_chunk = right_stream.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _copy_regular_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_real_directory_path(destination.parent)
    with (
        open_regular_binary(source, field="checkpoint source file") as source_stream,
        destination.open("xb") as target_stream,
    ):
        shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
        target_stream.flush()
        os.fsync(target_stream.fileno())


def merge_checkpoint_trees(source_root: Path, target_root: Path) -> None:
    """Merge one verified checkpoint tree without replacing conflicting bytes."""
    _assert_real_directory_path(source_root)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("checkpoint source tree must be a regular directory")
    if target_root.exists() or target_root.is_symlink():
        if target_root.is_symlink() or not target_root.is_dir():
            raise ValueError("checkpoint merge destination must be a directory")
    else:
        target_root.mkdir(parents=True)
    entries = sorted(
        source_root.rglob("*"),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    for source in entries:
        if source.is_symlink():
            raise ValueError("checkpoint source tree contains a symlink")
        relative = source.relative_to(source_root)
        target = target_root / relative
        if source.is_dir():
            if target.exists() and (target.is_symlink() or not target.is_dir()):
                raise ValueError(
                    "checkpoint path collision changes a file to a directory"
                )
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not source.is_file():
            raise ValueError("checkpoint source tree contains a special file")
        if target.exists() or target.is_symlink():
            if (
                target.is_symlink()
                or not target.is_file()
                or not _file_bytes_equal(source, target)
            ):
                raise ValueError("checkpoint path collision has different bytes")
            continue
        if any(
            parent.is_symlink() or not parent.is_dir()
            for parent in target.parents
            if parent != target_root.parent
        ):
            raise ValueError("checkpoint merge parent is not a regular directory")
        _copy_regular_file(source, target)


def find_source_root(extracted_root: Path) -> Path:
    """Find exactly one directory containing the frozen Dataset and Study inputs."""
    matches: list[Path] = []
    for directory, child_dirs, _files in os.walk(extracted_root, followlinks=False):
        candidate = Path(directory)
        child_dirs[:] = [
            name
            for name in child_dirs
            if not (candidate / name).is_symlink() and (candidate / name).is_dir()
        ]
        dataset = candidate / "dataset"
        study = candidate / "study"
        required = (
            dataset / "manifest.json",
            dataset / "arrays.npz",
            study / "plan.json",
        )
        if all(path.is_file() and not path.is_symlink() for path in required):
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError("source artifact must contain exactly one Dataset/Study root")
    return matches[0]


def _strict_seed(value: int, protocol: dict[str, Any]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed must be a strict integer")
    seeds = protocol["core_protocol"].get("seeds")
    if not isinstance(seeds, list) or value not in seeds:
        raise ValueError("seed is not preregistered")
    return value


def commands_for_stage(
    stage: str,
    source_root: Path,
    checkpoint_root: Path,
    protocol: dict[str, Any] | None,
    *,
    factor: str | None,
    seed: int | None,
) -> list[list[str]]:
    """Build bounded invocations of the existing PPO checkpoint CLI only."""
    if stage not in _STAGE_NAMES:
        raise ValueError("checkpoint stage is not supported")
    prefix = [
        sys.executable,
        "-m",
        "trade_rl.evaluation.ppo_feature_checkpoint",
    ]
    common = [
        "--source",
        str(source_root),
        "--output",
        str(checkpoint_root),
    ]
    if stage == "prepare":
        if factor is not None or seed is not None:
            raise ValueError("prepare does not accept a factor or seed")
        return [[*prefix, "prepare", *common]]
    if protocol is None:
        raise ValueError("arm and finalize require a validated checkpoint protocol")
    if stage == "finalize":
        if factor is not None or seed is not None:
            raise ValueError("finalize does not accept a factor or seed")
        return [[*prefix, "finalize", *common]]
    if factor not in {"baseline", "btc_relative"}:
        raise ValueError("factor is not preregistered")
    if seed is None:
        raise ValueError("arm requires a seed")
    checked_seed = _strict_seed(seed, protocol)
    factors = protocol["core_protocol"].get("factors")
    symbols = protocol["core_protocol"].get("symbols")
    scenario_roster = protocol["execution"].get("scenario_names_by_factor")
    if not isinstance(factors, dict) or factor not in factors:
        raise ValueError("factor is not in the validated core protocol")
    if (
        not isinstance(symbols, list)
        or not symbols
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
    ):
        raise ValueError("validated protocol symbol roster is malformed")
    if not isinstance(scenario_roster, dict):
        raise ValueError("validated protocol scenario roster is malformed")
    scenarios = scenario_roster.get(factor)
    if (
        not isinstance(scenarios, list)
        or not scenarios
        or any(not isinstance(scenario, str) or not scenario for scenario in scenarios)
    ):
        raise ValueError("validated factor scenario roster is malformed")
    commands = [
        [
            *prefix,
            "fit",
            *common,
            "--factor",
            factor,
            "--seed",
            str(checked_seed),
        ]
    ]
    for scenario in scenarios:
        for symbol_index in range(len(symbols)):
            commands.append(
                [
                    *prefix,
                    "replay-cell",
                    *common,
                    "--factor",
                    factor,
                    "--seed",
                    str(checked_seed),
                    "--scenario",
                    scenario,
                    "--symbol-index",
                    str(symbol_index),
                ]
            )
    commands.append(
        [
            *prefix,
            "assemble-arm",
            *common,
            "--factor",
            factor,
            "--seed",
            str(checked_seed),
        ]
    )
    return commands


def available_memory_bytes() -> int:
    """Return Linux MemAvailable; the production runner is ubuntu-24.04."""
    try:
        with Path("/proc/meminfo").open(encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    fields = line.split()
                    if len(fields) != 3 or fields[2] != "kB":
                        break
                    return int(fields[1]) * 1024
    except (OSError, ValueError):
        pass
    raise TransportError("Linux MemAvailable could not be read")


def _process_group_has_live_members(process_group_id: int) -> bool:
    if os.name != "posix" or not Path("/proc").is_dir():
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        return True
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="ascii")
        except (OSError, UnicodeDecodeError):
            continue
        closing_paren = raw.rfind(")")
        if closing_paren < 0:
            continue
        fields = raw[closing_paren + 1 :].split()
        if len(fields) > 2 and fields[0] not in {"Z", "X"}:
            try:
                if int(fields[2]) == process_group_id:
                    return True
            except ValueError:
                continue
    return False


def _wait_for_process_group_exit(process_group_id: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_has_live_members(process_group_id):
            return True
        time.sleep(0.05)
    return not _process_group_has_live_members(process_group_id)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        process_group_id = process.pid
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        _wait_for_process_group_exit(process_group_id, 5)
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if not _wait_for_process_group_exit(process_group_id, 5):
            raise TransportError("checkpoint child process group did not stop")
        process.wait()
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _read_child_diagnostic(log_path: Path) -> str:
    """Return a bounded, control-character-free tail from a failed child log."""
    try:
        with open_regular_binary(log_path, field="checkpoint child log") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - MAX_CHILD_DIAGNOSTIC_BYTES))
            raw = stream.read(MAX_CHILD_DIAGNOSTIC_BYTES)
    except (OSError, ValueError):
        return ""
    decoded = raw.decode("utf-8", errors="replace")
    return re.sub(r"[\x00-\x1f\x7f]", " ", decoded).strip()


def run_process_group(
    command: list[str],
    *,
    checkpoint_root: Path,
    deadline: float,
    log_directory: Path,
    command_index: int,
    memory_reader: Callable[[], int] = available_memory_bytes,
    poll_seconds: float = 5.0,
) -> None:
    """Run one CLI stage in a killable process group without forwarding tokens."""
    if not command or any(not isinstance(part, str) for part in command):
        raise ValueError("checkpoint command is malformed")
    if memory_reader() < STOP_MINIMUM_AVAILABLE_BYTES:
        raise TransportError("available memory is below the 1.5 GiB stop threshold")
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / f"command-{command_index:04d}.log"
    if log_path.exists() or log_path.is_symlink():
        raise FileExistsError("checkpoint command log already exists")
    child_env = os.environ.copy()
    child_env.pop("CHECKPOINT_GITHUB_TOKEN", None)
    child_env.pop("GITHUB_TOKEN", None)
    child_env.pop("GH_TOKEN", None)
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            while True:
                status = process.poll()
                if status is not None:
                    if status != 0:
                        raise TransportError(
                            f"checkpoint command {command_index} exited with status {status}"
                        )
                    _kill_process_group(process)
                    return
                if memory_reader() < STOP_MINIMUM_AVAILABLE_BYTES:
                    raise TransportError(
                        "available memory fell below the 1.5 GiB stop threshold"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "checkpoint stage exceeded its 300-minute budget"
                    )
                time.sleep(min(poll_seconds, remaining))
        except BaseException as error:
            _kill_process_group(process)
            if not isinstance(error, Exception):
                raise
            diagnostic = _read_child_diagnostic(log_path)
            if diagnostic:
                message = f"{error}; child output tail: {diagnostic}"
                if isinstance(error, TransportError):
                    raise TransportError(message) from error
                if isinstance(error, TimeoutError):
                    raise TimeoutError(message) from error
                raise RuntimeError(message) from error
            raise


def run_stage_commands(
    commands: list[list[str]],
    *,
    checkpoint_root: Path,
    deadline: float,
    command_runner: Callable[..., None] = run_process_group,
    log_directory: Path | None = None,
    on_success: Callable[[int, list[str]], None] | None = None,
) -> None:
    """Run stage commands in order; propagate errors and leave checkpoint files intact."""
    if not commands:
        raise ValueError("checkpoint stage has no commands")
    logs = log_directory or (checkpoint_root.parent / ".checkpoint-command-logs")
    for index, command in enumerate(commands, start=1):
        command_runner(
            command,
            checkpoint_root=checkpoint_root,
            deadline=deadline,
            log_directory=logs,
            command_index=index,
        )
        if on_success is not None:
            on_success(index, command)


def _safe_workflow_identity(environment: dict[str, str]) -> dict[str, Any]:
    repository = environment.get("GITHUB_REPOSITORY", "")
    _repo_path(repository)
    values = {
        "repository": repository,
        "code_sha": environment.get("CHECKPOINT_CODE_SHA", ""),
        "workflow": environment.get("GITHUB_WORKFLOW", ""),
        "workflow_ref": environment.get("GITHUB_WORKFLOW_REF", ""),
        "workflow_sha": environment.get("GITHUB_WORKFLOW_SHA", ""),
        "github_sha": environment.get("GITHUB_SHA", ""),
    }
    _require_commit_sha(values["code_sha"], field="code_sha")
    _require_commit_sha(values["workflow_sha"], field="workflow_sha")
    _require_commit_sha(values["github_sha"], field="github_sha")
    if not values["workflow"] or not values["workflow_ref"]:
        raise ValueError("workflow identity is incomplete")
    return values


def _runtime_provenance() -> dict[str, str]:
    child_environment = os.environ.copy()
    child_environment.pop("CHECKPOINT_GITHUB_TOKEN", None)
    child_environment.pop("GITHUB_TOKEN", None)
    child_environment.pop("GH_TOKEN", None)
    try:
        result = subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env=child_environment,
        )
    except (OSError, subprocess.SubprocessError):
        raise TransportError(
            "the configured uv executable could not be verified"
        ) from None
    version_match = re.match(r"^uv ([0-9]+\.[0-9]+\.[0-9]+)(?:\s|$)", result.stdout)
    if version_match is None or version_match.group(1) != "0.10.0":
        raise ValueError("the actual uv executable is not pinned at 0.10.0")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("the Python runtime is not pinned to Python 3.12")
    return {
        "python_version": sys.version,
        "uv_version": version_match.group(1),
    }


def _check_checkout(code_sha: str) -> None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        raise TransportError(
            "the requested code checkout could not be verified"
        ) from None
    current = result.stdout.strip()
    _require_commit_sha(current, field="checked-out code SHA")
    if current != code_sha:
        raise ValueError("checked-out code SHA differs from workflow input")


def _atomic_receipt(path: Path, receipt: dict[str, Any]) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("checkpoint receipt path is not a regular file")
    temporary = path.parent / f".receipt-{uuid.uuid4().hex}.tmp"
    raw = canonical_json_bytes(receipt)
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    return file_digest_and_size(path, field=path.name)[0]


def _new_receipt(
    environment: dict[str, str],
    stage: str,
    identity: dict[str, Any],
    runtime: dict[str, str],
) -> dict[str, Any]:
    run_id_text = environment.get("GITHUB_RUN_ID", "")
    attempt_text = environment.get("GITHUB_RUN_ATTEMPT", "")
    if not run_id_text.isdecimal() or run_id_text.startswith("0"):
        raise ValueError("workflow run id is missing or malformed")
    if not attempt_text.isdecimal() or attempt_text.startswith("0"):
        raise ValueError("workflow run attempt is missing or malformed")
    run_id = _strict_positive_int(int(run_id_text), field="workflow run id")
    run_attempt = _strict_positive_int(int(attempt_text), field="workflow run attempt")
    artifact_name = f"{CHECKPOINT_ARTIFACT_PREFIX}-{run_id}-{run_attempt}"
    workflow_file = Path(WORKFLOW_PATH)
    for path in (workflow_file, Path("uv.lock"), Path("pyproject.toml")):
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"required workflow provenance file is missing: {path.name}"
            )
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "stage": stage,
        "status": "started",
        **identity,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "artifact_name": artifact_name,
        "workflow_sha256": _sha256_file(workflow_file),
        "uv_lock_sha256": _sha256_file(Path("uv.lock")),
        "pyproject_sha256": _sha256_file(Path("pyproject.toml")),
        "checkpoint_protocol_digest": None,
        "approved_protocol_digest": None,
        "review_reference": None,
        "review_record_sha256": None,
        "input_artifacts": [],
        "source_artifact": {
            "id": SOURCE_ARTIFACT_ID,
            "run_id": SOURCE_ARTIFACT_RUN_ID,
            "name": SOURCE_ARTIFACT_NAME,
            "sha256": SOURCE_ARTIFACT_SHA256,
        },
        **runtime,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "completed_commands": [],
        "error_type": None,
        "error_message": None,
    }
    return receipt


def _initial_receipt(stage: str) -> dict[str, Any]:
    """Write a complete but deliberately non-importable receipt before preflight."""
    return {
        "schema": RECEIPT_SCHEMA,
        "stage": stage,
        "status": "started",
        "repository": None,
        "repository_id": None,
        "run_id": None,
        "run_attempt": None,
        "artifact_name": None,
        "code_sha": None,
        "workflow": None,
        "workflow_ref": None,
        "workflow_sha": None,
        "github_sha": None,
        "workflow_sha256": None,
        "uv_lock_sha256": None,
        "pyproject_sha256": None,
        "checkpoint_protocol_digest": None,
        "approved_protocol_digest": None,
        "review_reference": None,
        "review_record_sha256": None,
        "input_artifacts": [],
        "source_artifact": {
            "id": SOURCE_ARTIFACT_ID,
            "run_id": SOURCE_ARTIFACT_RUN_ID,
            "name": SOURCE_ARTIFACT_NAME,
            "sha256": SOURCE_ARTIFACT_SHA256,
        },
        "python_version": sys.version,
        "uv_version": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "completed_commands": [],
        "error_type": None,
        "error_message": None,
    }


def validate_producer_run(
    run: object,
    reference: ArtifactReference,
    identity: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    """Bind the receipt to GitHub's authoritative workflow-run record."""
    if not isinstance(run, dict):
        raise ValueError("GitHub producer workflow run is malformed")
    if _strict_positive_int(run.get("id"), field="producer run id") != (
        reference.run_id
    ):
        raise ValueError("GitHub producer workflow run id differs")
    if _strict_positive_int(run.get("run_attempt"), field="producer run attempt") != (
        receipt.get("run_attempt")
    ):
        raise ValueError("GitHub producer run attempt differs from receipt")
    if run.get("path") != WORKFLOW_PATH or run.get("name") != identity.get("workflow"):
        raise ValueError("GitHub producer workflow identity differs")
    if run.get("head_sha") != receipt.get("github_sha"):
        raise ValueError("GitHub producer run head differs from receipt")
    repository = run.get("repository")
    if not isinstance(repository, dict) or _strict_positive_int(
        repository.get("id"), field="producer repository id"
    ) != identity.get("repository_id"):
        raise ValueError("GitHub producer repository differs")
    head_repository = run.get("head_repository")
    if not isinstance(head_repository, dict) or _strict_positive_int(
        head_repository.get("id"), field="producer head repository id"
    ) != identity.get("repository_id"):
        raise ValueError("GitHub producer head repository differs")


def _validate_run_api_record(
    repository: str,
    reference: ArtifactReference,
    receipt: dict[str, Any],
    identity: dict[str, Any],
    *,
    token: str,
    deadline: float,
) -> None:
    attempt = _strict_positive_int(
        receipt.get("run_attempt"), field="receipt run attempt"
    )
    run = _api_json(
        "https://api.github.com/repos/"
        f"{_repo_path(repository)}/actions/runs/{reference.run_id}/attempts/{attempt}",
        token=token,
        deadline=deadline,
    )
    validate_producer_run(run, reference, identity, receipt)


def _receipt_for_import(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint artifact receipt is missing")
    raw = path.read_bytes()
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("checkpoint artifact receipt is invalid JSON") from None
    if not isinstance(receipt, dict) or canonical_json_bytes(receipt) != raw:
        raise ValueError("checkpoint artifact receipt is not canonical JSON")
    return receipt


def _download_and_extract_one(
    repository: str,
    reference: ArtifactReference,
    *,
    repository_id: int,
    token: str,
    directory: Path,
    deadline: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    archive_path = directory / f"artifact-{reference.artifact_id}.zip"
    metadata = download_artifact_archive(
        repository,
        reference,
        repository_id=repository_id,
        token=token,
        destination=archive_path,
        deadline=deadline,
    )
    extraction = directory / f"artifact-{reference.artifact_id}"
    extract_verified_archive(archive_path, extraction, reference.sha256)
    allowed = {"checkpoint", "receipt.json"}
    if {path.name for path in extraction.iterdir()} != allowed:
        raise ValueError(
            "checkpoint artifact must contain only checkpoint/ and receipt.json"
        )
    checkpoint_tree = extraction / "checkpoint"
    if checkpoint_tree.is_symlink() or not checkpoint_tree.is_dir():
        raise ValueError("checkpoint artifact has no checkpoint tree")
    receipt = _receipt_for_import(extraction / "receipt.json")
    return metadata, receipt


def _checkpoint_module() -> Any:
    from trade_rl.evaluation import ppo_feature_checkpoint

    return ppo_feature_checkpoint


def _restore_checkpoint_inputs(
    references: tuple[ArtifactReference, ...],
    *,
    output: Path,
    source_root: Path,
    identity: dict[str, Any],
    token: str,
    approved_protocol_digest: str,
    review_reference: str,
    review_record_sha256: str,
    stage: str,
    runtime: dict[str, str],
    deadline: float,
) -> tuple[Path, dict[str, Any], str]:
    if not references:
        raise ValueError("arm and finalize require a prepared checkpoint artifact")
    destination = output / "checkpoint"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("checkpoint restore destination already exists")
    staging = output / f".checkpoint-restore-{uuid.uuid4().hex}"
    staging.mkdir()
    token_repo = identity["repository"]
    repo_path = _repo_path(token_repo)
    repo_meta = _api_json(
        f"https://api.github.com/repos/{repo_path}", token=token, deadline=deadline
    )
    if (
        _strict_positive_int(repo_meta.get("id"), field="repository API id")
        != (identity["repository_id"])
    ):
        raise ValueError("workflow repository id differs from the GitHub API")
    prior: list[tuple[ArtifactReference, dict[str, Any], dict[str, Any]]] = []
    with tempfile.TemporaryDirectory(prefix="ppo-checkpoint-inputs-") as temporary:
        temp_root = Path(temporary)
        for index, reference in enumerate(references):
            artifact_dir = temp_root / f"input-{index}"
            artifact_dir.mkdir()
            metadata, receipt = _download_and_extract_one(
                token_repo,
                reference,
                repository_id=identity["repository_id"],
                token=token,
                directory=artifact_dir,
                deadline=deadline,
            )
            _validate_run_api_record(
                token_repo,
                reference,
                receipt,
                identity,
                token=token,
                deadline=deadline,
            )
            validate_prior_receipt(
                receipt,
                reference,
                metadata,
                identity,
                protocol_digest=None,
                runtime=runtime,
            )
            if receipt["stage"] != "prepare":
                if receipt["approved_protocol_digest"] != approved_protocol_digest:
                    raise ValueError(
                        "checkpoint inputs use a different approved protocol"
                    )
                if receipt["review_reference"] != review_reference:
                    raise ValueError(
                        "checkpoint inputs use a different review reference"
                    )
            merge_checkpoint_trees(
                artifact_dir / f"artifact-{reference.artifact_id}" / "checkpoint",
                staging,
            )
            prior.append((reference, metadata, receipt))

        module = _checkpoint_module()
        protocol = module.validate_checkpoint_protocol(source_root, staging)
        digest = content_digest(protocol)
        validate_operator_approval(
            stage,
            approved_protocol_digest,
            review_reference,
            digest,
            repository=identity["repository"],
            review_record_sha256=review_record_sha256,
        )
        prepare_rows = [
            (reference, receipt)
            for reference, _metadata, receipt in prior
            if receipt["stage"] == "prepare" and receipt["status"] == "succeeded"
        ]
        if len(prepare_rows) != 1:
            raise ValueError(
                "review evidence requires exactly one succeeded prepare artifact"
            )
        prepare_reference, _prepare_receipt = prepare_rows[0]
        pull_number, comment_id = parse_review_reference(
            review_reference,
            repository=identity["repository"],
        )
        review_comment = _api_json(
            "https://api.github.com/repos/"
            f"{_repo_path(identity['repository'])}/issues/comments/{comment_id}",
            token=token,
            deadline=deadline,
        )
        review_pull = _api_json(
            "https://api.github.com/repos/"
            f"{_repo_path(identity['repository'])}/pulls/{pull_number}",
            token=token,
            deadline=deadline,
        )
        validate_review_evidence_comment(
            review_comment,
            review_pull,
            review_reference=review_reference,
            review_record_sha256=review_record_sha256,
            repository=identity["repository"],
            repository_id=identity["repository_id"],
            code_sha=identity["code_sha"],
            workflow_sha=identity["workflow_sha"],
            protocol_digest=digest,
            prepare_reference=prepare_reference,
        )
        for reference, metadata, receipt in prior:
            validate_prior_receipt(
                receipt,
                reference,
                metadata,
                identity,
                protocol_digest=digest,
                approved_protocol_digest=approved_protocol_digest,
                review_reference=review_reference,
                review_record_sha256=review_record_sha256,
                runtime=runtime,
            )
    staging.rename(destination)
    return destination, protocol, digest


def _download_source(
    *,
    output: Path,
    repository: str,
    repository_id: int,
    token: str,
    deadline: float,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="ppo-feature-source-") as temporary:
        source_temp = Path(temporary)
        reference = ArtifactReference(
            SOURCE_ARTIFACT_ID,
            SOURCE_ARTIFACT_RUN_ID,
            SOURCE_ARTIFACT_SHA256,
        )
        metadata = download_artifact_archive(
            repository,
            reference,
            repository_id=repository_id,
            token=token,
            destination=source_temp / "source.zip",
            deadline=deadline,
        )
        if metadata.get("name") != SOURCE_ARTIFACT_NAME:
            raise ValueError("source artifact name differs from the frozen identity")
        extracted = source_temp / "source"
        extract_verified_archive(
            source_temp / "source.zip", extracted, SOURCE_ARTIFACT_SHA256
        )
        located = find_source_root(extracted)
        durable_source = output / "source"
        if durable_source.exists() or durable_source.is_symlink():
            raise FileExistsError("source output path already exists")
        shutil.copytree(located, durable_source, symlinks=False)
    return durable_source


def _parse_seed_environment(raw: str) -> int:
    if not raw or not raw.isdecimal() or (len(raw) > 1 and raw.startswith("0")):
        raise ValueError("seed input must be a canonical nonnegative integer")
    value = int(raw)
    if value < 0:
        raise ValueError("seed input must be nonnegative")
    return value


def execute_from_environment(environment: dict[str, str] | None = None) -> int:
    """Execute one workflow stage and write a non-economic receipt on every exit."""
    env = dict(os.environ if environment is None else environment)
    output_raw = env.get("CHECKPOINT_OUTPUT", "")
    if not output_raw:
        raise ValueError("CHECKPOINT_OUTPUT is required")
    output = Path(output_raw).absolute()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError("checkpoint output directory must not be a symlink")
    _assert_real_directory_path(output)
    stage = env.get("CHECKPOINT_STAGE", "")
    receipt_path = output / "receipt.json"
    identity: dict[str, Any] = {}
    receipt = _initial_receipt(stage)
    _atomic_receipt(receipt_path, receipt)
    try:
        if stage not in _STAGE_NAMES:
            raise ValueError("CHECKPOINT_STAGE must be prepare, arm, or finalize")
        if available_memory_bytes() < START_MINIMUM_AVAILABLE_BYTES:
            raise TransportError("less than 4 GiB memory is available at stage start")
        identity = _safe_workflow_identity(env)
        _check_checkout(identity["code_sha"])
        token = env.get("CHECKPOINT_GITHUB_TOKEN", "")
        if not token:
            raise ValueError("CHECKPOINT_GITHUB_TOKEN is required")
        deadline = time.monotonic() + TOTAL_DEADLINE_SECONDS
        repo_meta = _api_json(
            f"https://api.github.com/repos/{_repo_path(identity['repository'])}",
            token=token,
            deadline=deadline,
        )
        identity["repository_id"] = _strict_positive_int(
            repo_meta.get("id"), field="repository API id"
        )
        runtime = _runtime_provenance()
        receipt = _new_receipt(env, stage, identity, runtime)
        _atomic_receipt(receipt_path, receipt)
        references = parse_artifact_references(
            env.get("CHECKPOINT_ARTIFACTS_JSON", "[]")
        )
        receipt["input_artifacts"] = [
            reference.to_mapping() for reference in references
        ]
        if stage == "prepare" and references:
            raise ValueError(
                "prepare requires an empty checkpoint root and no prior artifacts"
            )
        if stage != "prepare" and not references:
            raise ValueError("arm and finalize require checkpoint artifact inputs")

        source_root = _download_source(
            output=output,
            repository=identity["repository"],
            repository_id=identity["repository_id"],
            token=token,
            deadline=deadline,
        )
        module = _checkpoint_module()
        checkpoint_root = output / "checkpoint"
        protocol: dict[str, Any] | None = None
        protocol_digest: str | None = None
        approval_digest = env.get("CHECKPOINT_APPROVED_PROTOCOL_DIGEST", "")
        review_input = env.get("CHECKPOINT_REVIEW_REFERENCE", "")
        review_record_input = env.get("CHECKPOINT_REVIEW_RECORD_SHA256", "")
        review_reference: str | None = None
        if stage == "prepare":
            if checkpoint_root.exists() or checkpoint_root.is_symlink():
                raise FileExistsError("prepare requires a new, empty checkpoint root")
        else:
            review_reference = review_input
            checkpoint_root, protocol, protocol_digest = _restore_checkpoint_inputs(
                references,
                output=output,
                source_root=source_root,
                identity=identity,
                token=token,
                approved_protocol_digest=approval_digest,
                review_reference=review_reference,
                review_record_sha256=review_record_input,
                stage=stage,
                runtime=runtime,
                deadline=deadline,
            )
            validate_operator_approval(
                stage,
                approval_digest,
                review_input,
                protocol_digest,
                repository=identity["repository"],
                review_record_sha256=review_record_input,
            )
            receipt["approved_protocol_digest"] = approval_digest
            receipt["review_reference"] = review_reference
            receipt["review_record_sha256"] = review_record_input
            receipt["checkpoint_protocol_digest"] = protocol_digest
            _atomic_receipt(receipt_path, receipt)

        commands = commands_for_stage(
            stage,
            source_root,
            checkpoint_root,
            protocol,
            factor=(env.get("CHECKPOINT_FACTOR") or None) if stage == "arm" else None,
            seed=(
                _parse_seed_environment(env.get("CHECKPOINT_SEED", ""))
                if stage == "arm"
                else None
            ),
        )
        log_directory = output / "logs"

        def command_succeeded(index: int, command: list[str]) -> None:
            receipt["completed_commands"].append(
                {"index": index, "command": command[3], "exit_status": 0}
            )
            _atomic_receipt(receipt_path, receipt)

        run_stage_commands(
            commands,
            checkpoint_root=checkpoint_root,
            deadline=deadline,
            log_directory=log_directory,
            on_success=command_succeeded,
        )
        protocol = module.validate_checkpoint_protocol(source_root, checkpoint_root)
        protocol_digest = content_digest(protocol)
        if stage != "prepare":
            validate_operator_approval(
                stage,
                approval_digest,
                review_input,
                protocol_digest,
                repository=identity["repository"],
                review_record_sha256=review_record_input,
            )
        receipt["checkpoint_protocol_digest"] = protocol_digest
        receipt["status"] = "succeeded"
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_receipt(receipt_path, receipt)
        return 0
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        receipt["error_type"] = type(error).__name__
        message = str(error)
        token = env.get("CHECKPOINT_GITHUB_TOKEN", "")
        if token:
            message = message.replace(token, "[redacted]")
        receipt["error_message"] = message[:500]
        if identity:
            for field, value in identity.items():
                receipt[field] = value
        if "input_artifacts" not in receipt or not isinstance(
            receipt["input_artifacts"], list
        ):
            receipt["input_artifacts"] = []
        try:
            _atomic_receipt(receipt_path, receipt)
        except OSError:
            pass
        raise


def main() -> None:
    try:
        execute_from_environment()
    except Exception as error:
        message = str(error)
        token = os.environ.get("CHECKPOINT_GITHUB_TOKEN", "")
        if token:
            message = message.replace(token, "[redacted]")
        print(
            f"checkpoint transport failed: {type(error).__name__}: {message}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
