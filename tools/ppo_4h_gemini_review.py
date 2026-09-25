"""Trusted result-blind Gemini reviewer for the 4h PPO indicator smoke."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PACKET_SCHEMA = "ppo_4h_gemini_review_packet_v1"
ATTESTATION_SCHEMA = "ppo_4h_gemini_reviewer_run_v1"
EXECUTION_BRANCH = "research/ppo-4h-indicator-smoke-execution"
BASE_BRANCH = "main"
GEMINI_PROVIDER = "google_gemini"
PACKET_FILES = (
    ".github/workflows/ci.yml",
    "docs/architecture/research-assurance.md",
    "docs/research/current-status.md",
    "tools/ppo_4h_indicator_smoke_actions.py",
    "trade_rl/evaluation/ppo_4h_indicator_smoke.py",
    "trade_rl/data/features/price_channels.py",
    "trade_rl/strategies/rl/ppo.py",
    "tests/architecture/test_ppo_4h_indicator_smoke_workflow.py",
    "tests/evaluation/test_ppo_4h_indicator_smoke.py",
    "tests/tools/test_ppo_4h_indicator_smoke_actions.py",
)
_REQUIRED_SOFTWARE_JOBS = ("Lean Core", "PPO Runtime", "Human Guide")
_REVIEW_TAG_RE = re.compile(
    r"^review/ppo-4h-indicator-smoke-v(?P<version>[1-9][0-9]*)$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SAFE_MODEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_API_BYTES = 8 * 1024 * 1024
_MAX_PACKET_BYTES = 4 * 1024 * 1024


class ReviewTransportError(RuntimeError):
    """Secret-free transport failure for the trusted reviewer."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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


def _environment_positive_int(raw: str | None, *, field: str) -> int:
    if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{field} must be a positive integer")
    return _positive_int(int(raw), field=field)


def _repo_path(repository: str) -> str:
    if not isinstance(repository, str) or _SAFE_REPO_RE.fullmatch(repository) is None:
        raise ValueError("GITHUB_REPOSITORY is malformed")
    owner, name = repository.split("/", 1)
    return f"{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(name, safe='')}"


def require_review_tag(review_tag: str) -> int:
    if not isinstance(review_tag, str):
        raise ValueError("review tag is malformed")
    match = _REVIEW_TAG_RE.fullmatch(review_tag)
    if match is None:
        raise ValueError("review tag is malformed")
    return int(match.group("version"))


def build_result_blind_packet(
    root: Path,
    *,
    repository: str,
    reviewed_code_sha: str,
    review_tag: str,
    review_tag_object_sha: str,
) -> dict[str, Any]:
    _repo_path(repository)
    reviewed = _require_sha(reviewed_code_sha, field="reviewed code SHA")
    require_review_tag(review_tag)
    tag_object = _require_sha(review_tag_object_sha, field="review tag object SHA")
    files: list[dict[str, str]] = []
    total = 0
    for relative in PACKET_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"packet source is missing or unsafe: {relative}")
        raw = path.read_bytes()
        total += len(raw)
        if total > _MAX_PACKET_BYTES:
            raise ValueError("result-blind packet exceeds size budget")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"packet source is not UTF-8: {relative}") from None
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "text": text,
            }
        )
    return {
        "schema": PACKET_SCHEMA,
        "repository": repository,
        "reviewed_code_sha": reviewed,
        "review_tag": review_tag,
        "review_tag_object_sha": tag_object,
        "result_blind": True,
        "files": files,
    }


def validate_software_ci(
    run: object,
    jobs: object,
    reviewed_code_sha: str,
) -> tuple[int, int]:
    reviewed = _require_sha(reviewed_code_sha, field="reviewed code SHA")
    if not isinstance(run, dict):
        raise ValueError("software verification run is malformed")
    if (
        run.get("name") != "CI"
        or run.get("event") != "pull_request"
        or run.get("head_sha") != reviewed
        or run.get("status") != "completed"
    ):
        raise ValueError("software verification is not exact-head PR CI")
    run_id = _positive_int(run.get("id"), field="software CI run id")
    attempt = _positive_int(run.get("run_attempt"), field="software CI run attempt")
    if not isinstance(jobs, list):
        raise ValueError("software verification jobs are malformed")
    by_name: dict[str, dict[str, Any]] = {}
    for item in jobs:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            by_name[item["name"]] = item
    for name in _REQUIRED_SOFTWARE_JOBS:
        item = by_name.get(name)
        if (
            item is None
            or item.get("status") != "completed"
            or item.get("conclusion") != "success"
        ):
            raise ValueError(f"software verification job is not Green: {name}")
    return run_id, attempt


def _required_string_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} is malformed")
    return list(value)


def parse_gemini_response(response: object) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("Gemini response is malformed")
    model = response.get("modelVersion")
    response_id = response.get("responseId")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Gemini model provenance is missing")
    if not isinstance(response_id, str) or not response_id.strip():
        raise ValueError("Gemini response identity is missing")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("Gemini response candidate roster is malformed")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise ValueError("Gemini response candidate is malformed")
    content = candidate.get("content")
    if not isinstance(content, dict):
        raise ValueError("Gemini response content is malformed")
    parts = content.get("parts")
    if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict):
        raise ValueError("Gemini response parts are malformed")
    raw_text = parts[0].get("text")
    if not isinstance(raw_text, str):
        raise ValueError("Gemini structured review is missing")
    try:
        review = json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError("Gemini structured review is invalid JSON") from None
    expected = {
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
    if not isinstance(review, dict) or set(review) != expected:
        raise ValueError("Gemini structured review shape is unsupported")
    if review.get("g0") not in {"PASS", "FAIL", "NOT_ESTABLISHED"}:
        raise ValueError("Gemini G0 result is malformed")
    if review.get("g1") not in {"PASS", "FAIL", "NOT_ESTABLISHED"}:
        raise ValueError("Gemini G1 result is malformed")
    if review.get("g2") not in {"PASS", "EVIDENCE_BOUND", "FAIL", "NOT_ESTABLISHED"}:
        raise ValueError("Gemini G2 result is malformed")
    if review.get("disposition") not in {
        "G0_G1_CLEAR_G2_EVIDENCE_BOUND",
        "BLOCKED",
    }:
        raise ValueError("Gemini disposition is malformed")
    findings = review.get("blocking_findings")
    if not isinstance(findings, list) or any(
        not isinstance(item, str) or not item.strip() for item in findings
    ):
        raise ValueError("Gemini blocking findings are malformed")
    for field in (
        "strongest_counterexample",
        "claim_downgrade",
    ):
        value = review.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Gemini {field} is malformed")
    for field in ("missing_evidence", "machine_oracles", "what_this_cannot_prove"):
        review[field] = _required_string_list(review.get(field), field=f"Gemini {field}")
    return {
        "reviewer_provider": GEMINI_PROVIDER,
        "reviewer_model": model,
        "reviewer_response_id": response_id,
        **review,
    }


def build_attestation(
    *,
    repository: str,
    repository_id: int,
    reviewed_code_sha: str,
    review_tag: str,
    review_tag_object_sha: str,
    trusted_workflow_sha: str,
    trusted_workflow_ref: str,
    reviewer_run_id: int,
    reviewer_run_attempt: int,
    ci_run_id: int,
    ci_run_attempt: int,
    packet_sha256: str,
    parsed_review: dict[str, Any],
) -> dict[str, Any]:
    _repo_path(repository)
    reviewed = _require_sha(reviewed_code_sha, field="reviewed code SHA")
    require_review_tag(review_tag)
    tag_object = _require_sha(review_tag_object_sha, field="review tag object SHA")
    workflow_sha = _require_sha(trusted_workflow_sha, field="trusted workflow SHA")
    packet_digest = _require_sha256(packet_sha256, field="review packet SHA-256")
    if not isinstance(trusted_workflow_ref, str) or not trusted_workflow_ref.endswith(
        "@refs/heads/main"
    ):
        raise ValueError("trusted workflow ref is not default-branch bound")
    if parsed_review.get("reviewer_provider") != GEMINI_PROVIDER:
        raise ValueError("reviewer provider is not Google Gemini")
    fields = {
        key: parsed_review[key]
        for key in (
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
        )
    }
    return {
        "schema": ATTESTATION_SCHEMA,
        "repository": repository,
        "repository_id": _positive_int(repository_id, field="repository id"),
        "reviewed_code_sha": reviewed,
        "review_tag": review_tag,
        "review_tag_object_sha": tag_object,
        "trusted_workflow_sha": workflow_sha,
        "trusted_workflow_ref": trusted_workflow_ref,
        "reviewer_run_id": _positive_int(reviewer_run_id, field="reviewer run id"),
        "reviewer_run_attempt": _positive_int(
            reviewer_run_attempt, field="reviewer run attempt"
        ),
        "ci_run_id": _positive_int(ci_run_id, field="software CI run id"),
        "ci_run_attempt": _positive_int(ci_run_attempt, field="software CI run attempt"),
        "packet_sha256": packet_digest,
        "result_blind": True,
        "reviewer_context": "trusted_default_branch_read_only",
        "python_version": sys.version.split()[0],
        **fields,
    }


def _api_json(
    url: str,
    *,
    token: str,
    method: str = "GET",
    payload: object | None = None,
    headers: dict[str, str] | None = None,
) -> object:
    request_headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "trade-rl-ppo-gemini-reviewer",
    }
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    if headers:
        request_headers.update(headers)
    data = None if payload is None else _canonical_json_bytes(payload)
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise ReviewTransportError("remote API returned an unexpected status")
            raw = response.read(_MAX_API_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise ReviewTransportError(
            f"remote API request failed with HTTP {error.code}"
        ) from None
    except (OSError, urllib.error.URLError, TimeoutError):
        raise ReviewTransportError("remote API request could not be completed") from None
    if len(raw) > _MAX_API_BYTES:
        raise ReviewTransportError("remote API response exceeds size limit")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ReviewTransportError("remote API returned malformed JSON") from None


def _github_api(repository: str, suffix: str, *, token: str) -> object:
    return _api_json(
        f"https://api.github.com/repos/{_repo_path(repository)}/{suffix.lstrip('/')}",
        token=token,
        headers={"X-GitHub-Api-Version": "2022-11-28"},
    )


def _resolve_review_tag(
    repository: str,
    review_tag: str,
    *,
    token: str,
    expected_reviewed_sha: str,
) -> str:
    require_review_tag(review_tag)
    reviewed = _require_sha(expected_reviewed_sha, field="reviewed code SHA")
    encoded = urllib.parse.quote(f"tags/{review_tag}", safe="/")
    ref = _github_api(repository, f"git/ref/{encoded}", token=token)
    if not isinstance(ref, dict):
        raise ValueError("review tag ref is malformed")
    obj = ref.get("object")
    if (
        ref.get("ref") != f"refs/tags/{review_tag}"
        or not isinstance(obj, dict)
        or obj.get("type") != "tag"
    ):
        raise ValueError("review tag must be an annotated tag")
    tag_object_sha = _require_sha(obj.get("sha"), field="review tag object SHA")
    tag_object = _github_api(repository, f"git/tags/{tag_object_sha}", token=token)
    if not isinstance(tag_object, dict) or tag_object.get("tag") != review_tag:
        raise ValueError("review tag object identity differs")
    target = tag_object.get("object")
    if not isinstance(target, dict) or target.get("type") != "commit":
        raise ValueError("review tag object does not target a commit")
    if _require_sha(target.get("sha"), field="review tag commit SHA") != reviewed:
        raise ValueError("review tag does not bind the reviewed code SHA")
    return tag_object_sha


def _require_current_main_contained(
    repository: str,
    reviewed_code_sha: str,
    *,
    token: str,
) -> str:
    reviewed = _require_sha(reviewed_code_sha, field="reviewed code SHA")
    branch = _github_api(repository, f"branches/{BASE_BRANCH}", token=token)
    if not isinstance(branch, dict) or not isinstance(branch.get("commit"), dict):
        raise ValueError("current main branch record is malformed")
    main_sha = _require_sha(branch["commit"].get("sha"), field="current main SHA")
    comparison = _github_api(
        repository,
        f"compare/{main_sha}...{reviewed}",
        token=token,
    )
    if (
        not isinstance(comparison, dict)
        or comparison.get("status") not in {"ahead", "identical"}
        or comparison.get("behind_by") != 0
    ):
        raise ValueError("reviewed code does not contain current main")
    return main_sha


def _require_execution_pull(
    repository: str,
    reviewed_code_sha: str,
    *,
    token: str,
) -> int:
    owner = repository.split("/", 1)[0]
    query = urllib.parse.urlencode(
        {
            "state": "open",
            "head": f"{owner}:{EXECUTION_BRANCH}",
            "base": BASE_BRANCH,
            "per_page": "100",
        }
    )
    pulls = _github_api(repository, f"pulls?{query}", token=token)
    if not isinstance(pulls, list):
        raise ValueError("execution pull inventory is malformed")
    matches: list[dict[str, Any]] = []
    for pull in pulls:
        if not isinstance(pull, dict):
            continue
        head = pull.get("head")
        base = pull.get("base")
        if (
            pull.get("state") == "open"
            and pull.get("draft") is True
            and isinstance(head, dict)
            and head.get("ref") == EXECUTION_BRANCH
            and head.get("sha") == reviewed_code_sha
            and isinstance(head.get("repo"), dict)
            and head["repo"].get("full_name") == repository
            and isinstance(base, dict)
            and base.get("ref") == BASE_BRANCH
        ):
            matches.append(pull)
    if len(matches) != 1:
        raise ValueError("exact reviewed code is not owned by one open Draft execution PR")
    return _positive_int(matches[0].get("number"), field="execution pull number")


def _find_software_ci(
    repository: str,
    reviewed_code_sha: str,
    *,
    token: str,
) -> tuple[int, int]:
    query = urllib.parse.urlencode(
        {
            "head_sha": reviewed_code_sha,
            "event": "pull_request",
            "per_page": "100",
        }
    )
    payload = _github_api(repository, f"actions/runs?{query}", token=token)
    if not isinstance(payload, dict) or not isinstance(payload.get("workflow_runs"), list):
        raise ValueError("software CI inventory is malformed")
    runs = sorted(
        (run for run in payload["workflow_runs"] if isinstance(run, dict)),
        key=lambda item: int(item.get("id", 0)),
        reverse=True,
    )
    for run in runs:
        if (
            run.get("name") != "CI"
            or run.get("event") != "pull_request"
            or run.get("head_sha") != reviewed_code_sha
            or run.get("status") != "completed"
        ):
            continue
        run_id = _positive_int(run.get("id"), field="software CI run id")
        jobs_payload = _github_api(
            repository,
            f"actions/runs/{run_id}/jobs?per_page=100",
            token=token,
        )
        if not isinstance(jobs_payload, dict):
            continue
        jobs = jobs_payload.get("jobs")
        try:
            return validate_software_ci(run, jobs, reviewed_code_sha)
        except ValueError:
            continue
    raise ValueError("no exact-head software verification run has all required Green jobs")


def _gemini_schema() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "g0": {"type": "string", "enum": ["PASS", "FAIL", "NOT_ESTABLISHED"]},
            "g1": {"type": "string", "enum": ["PASS", "FAIL", "NOT_ESTABLISHED"]},
            "g2": {
                "type": "string",
                "enum": ["PASS", "EVIDENCE_BOUND", "FAIL", "NOT_ESTABLISHED"],
            },
            "disposition": {
                "type": "string",
                "enum": ["G0_G1_CLEAR_G2_EVIDENCE_BOUND", "BLOCKED"],
            },
            "blocking_findings": string_array,
            "strongest_counterexample": {"type": "string"},
            "missing_evidence": string_array,
            "claim_downgrade": {"type": "string"},
            "machine_oracles": string_array,
            "what_this_cannot_prove": string_array,
        },
        "required": [
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
        ],
        "additionalProperties": False,
    }


def _review_prompt(packet: dict[str, Any]) -> str:
    return (
        "You are the independent result-blind G0-G2 reviewer for a development-only "
        "4h PPO indicator smoke. Review only the exact packet below. Do not infer or "
        "request economic outputs, P&L, winner data, unused/final data, or live results. "
        "Adversarially test the research question, mechanism, causality/availability, "
        "fit/evaluation scope, risk/execution/accounting semantics, machine oracles, "
        "and claim boundary. PASS only when the exact packet supports the bounded "
        "development-smoke authorization. Otherwise set disposition=BLOCKED and list "
        "specific blocking findings. Your output must follow the supplied JSON schema.\n\n"
        + _canonical_json_bytes(packet).decode("utf-8")
    )


def _call_gemini(
    packet: dict[str, Any],
    *,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required")
    if not isinstance(model, str) or _SAFE_MODEL_RE.fullmatch(model) is None:
        raise ValueError("GEMINI_MODEL is malformed")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model, safe='')}:generateContent"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": _review_prompt(packet)}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseJsonSchema": _gemini_schema(),
        },
    }
    value = _api_json(
        endpoint,
        token="",
        method="POST",
        payload=payload,
        headers={"x-goog-api-key": api_key},
    )
    if not isinstance(value, dict):
        raise ValueError("Gemini API response is malformed")
    return value


def _write_canonical(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))


def _is_authorizing(review: dict[str, Any]) -> bool:
    return (
        review.get("g0") == "PASS"
        and review.get("g1") == "PASS"
        and review.get("g2") in {"PASS", "EVIDENCE_BOUND"}
        and review.get("disposition") == "G0_G1_CLEAR_G2_EVIDENCE_BOUND"
        and review.get("blocking_findings") == []
    )


def run(environment: dict[str, str] | None = None) -> int:
    env = dict(os.environ if environment is None else environment)
    repository = env.get("GITHUB_REPOSITORY", "")
    repository_id = _environment_positive_int(
        env.get("GITHUB_REPOSITORY_ID"), field="repository id"
    )
    token = env.get("GITHUB_TOKEN", "")
    if not token:
        raise ValueError("GITHUB_TOKEN is required")
    review_tag = env.get("REVIEW_TAG", "")
    reviewed = _require_sha(env.get("REVIEWED_SHA"), field="reviewed code SHA")
    workflow_sha = _require_sha(
        env.get("TRUSTED_WORKFLOW_SHA"), field="trusted workflow SHA"
    )
    workflow_ref = env.get("GITHUB_WORKFLOW_REF", "")
    reviewer_run_id = _environment_positive_int(
        env.get("GITHUB_RUN_ID"), field="reviewer run id"
    )
    reviewer_attempt = _environment_positive_int(
        env.get("GITHUB_RUN_ATTEMPT"), field="reviewer run attempt"
    )
    target_root = Path(env.get("TARGET_ROOT", "target"))
    output = Path(env.get("REVIEW_OUTPUT", "output"))
    if output.exists() or output.is_symlink():
        raise FileExistsError("review output already exists")

    tag_object_sha = _resolve_review_tag(
        repository,
        review_tag,
        token=token,
        expected_reviewed_sha=reviewed,
    )
    _require_current_main_contained(repository, reviewed, token=token)
    _require_execution_pull(repository, reviewed, token=token)
    ci_run_id, ci_attempt = _find_software_ci(repository, reviewed, token=token)
    packet = build_result_blind_packet(
        target_root,
        repository=repository,
        reviewed_code_sha=reviewed,
        review_tag=review_tag,
        review_tag_object_sha=tag_object_sha,
    )
    packet_digest = hashlib.sha256(_canonical_json_bytes(packet)).hexdigest()
    raw_response = _call_gemini(
        packet,
        api_key=env.get("GEMINI_API_KEY", ""),
        model=env.get("GEMINI_MODEL", ""),
    )
    parsed = parse_gemini_response(raw_response)
    attestation = build_attestation(
        repository=repository,
        repository_id=repository_id,
        reviewed_code_sha=reviewed,
        review_tag=review_tag,
        review_tag_object_sha=tag_object_sha,
        trusted_workflow_sha=workflow_sha,
        trusted_workflow_ref=workflow_ref,
        reviewer_run_id=reviewer_run_id,
        reviewer_run_attempt=reviewer_attempt,
        ci_run_id=ci_run_id,
        ci_run_attempt=ci_attempt,
        packet_sha256=packet_digest,
        parsed_review=parsed,
    )

    output.mkdir(parents=True, exist_ok=False)
    _write_canonical(output / "review-packet.json", packet)
    _write_canonical(output / "gemini-response.json", raw_response)
    _write_canonical(output / "reviewer-attestation.json", attestation)
    (output / "reviewer-disposition.txt").write_text(
        f"{attestation['disposition']}\n",
        encoding="utf-8",
    )
    return 0 if _is_authorizing(attestation) else 2


def main() -> int:
    try:
        return run()
    except Exception as error:
        print(f"trusted Gemini review failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
