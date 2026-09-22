"""Authenticated one-shot transport for the 4h PPO indicator smoke."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
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
REVIEW_SCHEMA = "ppo_4h_indicator_smoke_review_v1"
SOURCE_REVIEW_SCHEMA = "ppo_4h_indicator_source_review_v2"
SOURCE_REVIEW_MARKER = "<!-- ppo-4h-indicator-source-review-v2 -->\n"
REVIEWER_SURFACE = "github_pr_review_v2"
TRIGGER_MESSAGE = "run: execute 4h PPO indicator smoke"
MINIMUM_AVAILABLE_BYTES = 4 * 1024**3
DEADLINE_SECONDS = 300 * 60
_REVIEW_URL_RE = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+)/pull/(?P<pull>[1-9][0-9]*)#"
    r"pullrequestreview-(?P<review>[1-9][0-9]*)$"
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
        "reviewer_independence",
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
    if (
        not isinstance(findings, list)
        or any(not isinstance(item, str) or not item for item in findings)
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
    body = record.get("body")
    if (
        record.get("html_url") != review_url
        or not isinstance(body, str)
        or hashlib.sha256(body.encode("utf-8")).hexdigest() != review_sha
    ):
        raise ValueError("source review identity or bytes differ")
    if record.get("commit_id") != reviewed:
        raise ValueError("source review commit differs from reviewed code SHA")
    if record.get("state") not in {"COMMENTED", "APPROVED"}:
        raise ValueError("source review state does not authorize execution")
    reviewer_id = _github_principal_id(record, field="source review")
    author_id = _github_principal_id(pull, field="pull request author")
    if reviewer_id == author_id:
        raise ValueError(
            "source review is not independent from the pull request author"
        )

    source = _canonical_source_review(body)
    if source.get("reviewed_code_sha") != reviewed:
        raise ValueError("source review payload binds a different code SHA")
    if source.get("static_contract_digest") != expected_static:
        raise ValueError("source review payload binds a different static contract")
    if source.get("reviewer_independence") != "ESTABLISHED":
        raise ValueError("source review independence is not established")
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

    for field in (
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
        smoke.run_smoke(source, output)
    return 0


def main() -> int:
    return execute_from_environment()


if __name__ == "__main__":
    raise SystemExit(main())
