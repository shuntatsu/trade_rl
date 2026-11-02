"""GET-only single-run Actions artifact observation, without lease/cleanup authority.

GitHub lists artifacts by run, not by run attempt. A stable double read of the
current listing cannot recover expired/deleted artifact history, authenticate
producers, verify artifact bytes, or prove complete repository Task/workflow work.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, NamedTuple, Protocol

_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_RUN_STATES = frozenset(
    {"requested", "waiting", "pending", "queued", "in_progress", "completed"}
)
_CONCLUSIONS = frozenset(
    {
        "success",
        "failure",
        "cancelled",
        "skipped",
        "neutral",
        "timed_out",
        "action_required",
        "stale",
    }
)
_PAGE_SIZE = 100
_MAX_ARTIFACTS = 10_000


class ArtifactReadApi(Protocol):
    def _request_json(
        self, method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> Any: ...


class _Run(NamedTuple):
    run_id: int
    attempt: int
    branch: str
    sha: str
    status: str
    conclusion: str | None
    updated_at: str
    repository_id: int
    head_repository_id: int


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a GitHub response object")
    return value


def _positive(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive native integer")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _repo(api: ArtifactReadApi, expected: str) -> tuple[str, int, str]:
    raw = _mapping(api._request_json("GET", ""), "repository identity")
    name = _text(raw.get("full_name"), "repository identity")
    if name.casefold() != expected.casefold():
        raise ValueError("repository identity differs from requested repository")
    return name.casefold(), _positive(raw.get("id"), "repository id"), _text(
        raw.get("node_id"), "repository node id"
    )


def _run(api: ArtifactReadApi, run_id: int, repository_id: int) -> _Run:
    raw = _mapping(
        api._request_json("GET", f"actions/runs/{run_id}"), "workflow run"
    )
    if _positive(raw.get("id"), "workflow run id") != run_id:
        raise ValueError("workflow run id changed")
    attempt = _positive(raw.get("run_attempt"), "run attempt")
    branch = _text(raw.get("head_branch"), "run head branch")
    sha = raw.get("head_sha")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise ValueError("run head SHA must be an exact Git SHA")
    status = _text(raw.get("status"), "run status")
    if status not in _RUN_STATES:
        raise ValueError("run status is unknown")
    conclusion = raw.get("conclusion")
    if status == "completed":
        if conclusion not in _CONCLUSIONS:
            raise ValueError("completed run conclusion is unknown")
    elif conclusion is not None:
        raise ValueError("unfinished run cannot have a conclusion")
    run_repo = _mapping(raw.get("repository"), "run repository")
    if _positive(run_repo.get("id"), "run repository id") != repository_id:
        raise ValueError("run repository differs from requested repository")
    head_repo = _mapping(raw.get("head_repository"), "run head repository")
    head_repo_id = _positive(head_repo.get("id"), "run head repository id")
    return _Run(
        run_id,
        attempt,
        branch,
        sha,
        status,
        conclusion,
        _text(raw.get("updated_at"), "run updated_at"),
        repository_id,
        head_repo_id,
    )


def _artifact(raw: Any, run: _Run) -> dict[str, Any]:
    item = _mapping(raw, "artifact")
    artifact_id = _positive(item.get("id"), "artifact id")
    node_id = _text(item.get("node_id"), "artifact node_id")
    name = _text(item.get("name"), "artifact name")
    size = item.get("size_in_bytes")
    if type(size) is not int or size < 0:
        raise ValueError("artifact size_in_bytes is invalid")
    expired = item.get("expired")
    if type(expired) is not bool:
        raise ValueError("artifact expired status is invalid")
    digest = item.get("digest")
    if digest is not None and (
        not isinstance(digest, str) or not _DIGEST.fullmatch(digest)
    ):
        raise ValueError("artifact digest is malformed")
    origin = _mapping(item.get("workflow_run"), "artifact workflow identity")
    if (
        _positive(origin.get("id"), "artifact workflow run id") != run.run_id
        or _positive(origin.get("repository_id"), "artifact repository id")
        != run.repository_id
        or _positive(origin.get("head_repository_id"), "artifact head repository id")
        != run.head_repository_id
        or origin.get("head_branch") != run.branch
        or origin.get("head_sha") != run.sha
    ):
        raise ValueError("artifact workflow identity differs from pinned run")
    return {
        "id": artifact_id,
        "node_id": node_id,
        "name": name,
        "size_in_bytes": size,
        "expired": expired,
        "digest": digest,
    }


def _scan_artifacts(api: ArtifactReadApi, run: _Run) -> list[dict[str, Any]]:
    endpoint = f"actions/runs/{run.run_id}/artifacts"
    artifacts: list[dict[str, Any]] = []
    seen: set[int] = set()
    expected_total: int | None = None
    page = 1
    while True:
        raw = _mapping(
            api._request_json(
                "GET",
                endpoint,
                query={"per_page": str(_PAGE_SIZE), "page": str(page)},
            ),
            "workflow artifact page",
        )
        count = raw.get("total_count")
        if type(count) is not int or not 0 <= count <= _MAX_ARTIFACTS:
            raise ValueError("artifact total_count is missing or exceeds limit")
        if expected_total is None:
            expected_total = count
        elif count != expected_total:
            raise ValueError("artifact total_count changed between pages")
        rows = raw.get("artifacts")
        remaining = count - len(artifacts)
        if (
            not isinstance(rows, list)
            or remaining < 0
            or len(rows) != min(_PAGE_SIZE, remaining)
        ):
            raise ValueError("artifact pagination is incomplete or inconsistent")
        for value in rows:
            item = _artifact(value, run)
            if item["id"] in seen:
                raise ValueError("duplicate artifact id")
            seen.add(item["id"])
            artifacts.append(item)
        if len(artifacts) == count:
            return sorted(artifacts, key=lambda item: item["id"])
        page += 1


def observe_workflow_artifacts(
    api: ArtifactReadApi, *, expected_repository: str, run_id: int
) -> dict[str, Any]:
    """Read two complete run-scoped artifact listings and reject observed drift.

    total_count=0 means only that the API returned no *current* artifacts;
    it does not prove none were produced earlier or deleted. Even identical
    observations are not atomic, do not bind artifacts to run attempts, and
    cannot authorize an expired-lease decision or repository cleanup.
    """
    if not isinstance(expected_repository, str) or not _REPO.fullmatch(
        expected_repository
    ):
        raise ValueError("expected_repository must be owner/name")
    run_id = _positive(run_id, "run_id")
    repo = _repo(api, expected_repository)
    run = _run(api, run_id, repo[1])
    first = _scan_artifacts(api, run)
    second = _scan_artifacts(api, run)
    if first != second:
        raise ValueError("artifact roster changed during observation")
    if _run(api, run_id, repo[1]) != run:
        raise ValueError("workflow run changed during artifact observation")
    if _repo(api, expected_repository) != repo:
        raise ValueError("repository changed during artifact observation")
    return {
        "schema": "github_workflow_artifact_observation_v1",
        "repository": expected_repository,
        "repository_id": repo[1],
        "run_id": run_id,
        "run_attempt_observed": run.attempt,
        "head_branch": run.branch,
        "head_sha": run.sha,
        "status": run.status,
        "conclusion": run.conclusion,
        "artifact_count": len(first),
        "artifacts": first,
        "scope": "single_run_all_attempts_current_artifacts",
        "artifact_metadata_double_read": True,
        "artifact_attempt_binding_verified": False,
        "artifact_bytes_verified": False,
        "artifact_history_complete": False,
        "workflow_activity_checked": False,
        "task_registry_complete": False,
        "lease_reassignment_authorized": False,
        "cleanup_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Explicit single-run GET-only audit; requires no mutation permission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="shuntatsu/trade_rl")
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args(argv)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("GH_TOKEN or GITHUB_TOKEN is required for artifact audit")

    from tools.branch_hygiene import GitHubApi

    reader = GitHubApi(
        repository=args.repository, token=token, api_url="https://api.github.com"
    )
    print(
        json.dumps(
            observe_workflow_artifacts(
                reader, expected_repository=args.repository, run_id=args.run_id
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = ["observe_workflow_artifacts"]


if __name__ == "__main__":
    raise SystemExit(main())
