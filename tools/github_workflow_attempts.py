"""Read-only, single-run GitHub Actions attempt observation; never cleanup authority.

The ordinary run-scoped jobs endpoint may return an empty list even when the
attempt-scoped endpoint contains jobs. Always inspect the exact run attempt.
A coherent observation of one run does NOT establish a complete repository
workflow roster or allow Task enrollment, reconciliation, or branch deletion.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, NamedTuple, Protocol

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_STATES = frozenset(
    {
        "requested",
        "waiting",
        "pending",
        "queued",
        "in_progress",
        "completed",
    }
)
_JOB_STATES = _RUN_STATES
_PAGE_SIZE = 100
_MAX_JOBS = 10_000
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


class _RunIdentity(NamedTuple):
    run_id: int
    attempt: int
    branch: str
    head_sha: str
    status: str
    conclusion: str | None
    updated_at: str


class WorkflowReadApi(Protocol):
    """The narrow GET-only surface needed for one run-attempt observation."""

    def _request_json(
        self,
        method: str,
        suffix: str,
        *,
        query: dict[str, str] | None = None,
    ) -> Any: ...


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive native integer")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 40-hex SHA")
    return value


def _state_and_conclusion(
    value: dict[str, Any],
    *,
    name: str,
) -> tuple[str, str | None]:
    status = _text(value.get("status"), f"{name} status")
    allowed = _RUN_STATES if name == "run" else _JOB_STATES
    if status not in allowed:
        raise ValueError(f"{name} status is unknown")
    conclusion = value.get("conclusion")
    if status == "completed":
        if _text(conclusion, f"{name} conclusion") not in _CONCLUSIONS:
            raise ValueError(f"{name} conclusion is unknown")
    elif conclusion is not None:
        raise ValueError(f"{name} unfinished status has a conclusion")
    return status, conclusion


def _run_identity(raw: Any, *, run_id: int) -> _RunIdentity:
    run = _mapping(raw, "workflow run")
    if _positive_int(run.get("id"), "run id") != run_id:
        raise ValueError("workflow run id changed")
    attempt = _positive_int(run.get("run_attempt"), "run attempt")
    branch = _text(run.get("head_branch"), "workflow head branch")
    head = _sha(run.get("head_sha"), "workflow head SHA")
    status, conclusion = _state_and_conclusion(run, name="run")
    updated = _text(run.get("updated_at"), "workflow update time")
    return _RunIdentity(run_id, attempt, branch, head, status, conclusion, updated)


def _job_identity(
    raw: Any,
    *,
    run_id: int,
    attempt: int,
    branch: str,
    head: str,
) -> dict[str, Any]:
    job = _mapping(raw, "workflow job")
    job_id = _positive_int(job.get("id"), "job id")
    if _positive_int(job.get("run_id"), "job run_id") != run_id:
        raise ValueError("job belongs to another run")
    if _positive_int(job.get("run_attempt"), "job run_attempt") != attempt:
        raise ValueError("job belongs to another run attempt")
    if _sha(job.get("head_sha"), "job head SHA") != head:
        raise ValueError("job head SHA differs from workflow run")
    if _text(job.get("head_branch"), "job head branch") != branch:
        raise ValueError("job head branch differs from workflow run")
    name = _text(job.get("name"), "job name")
    status, conclusion = _state_and_conclusion(job, name="job")
    return {"id": job_id, "name": name, "status": status, "conclusion": conclusion}


def observe_workflow_attempt(reader: WorkflowReadApi, *, run_id: int) -> dict[str, Any]:
    """Verify one exact run attempt with bounded full pagination and a run readback.

    Does not query unscoped ``/runs/{id}/jobs`` or infer the absence of other
    runs, Tasks, branches, PRs, or artifacts from this single observation.
    Any ambiguous, partial, changing, or unavailable input raises rather than
    returning a safe/no-active-work conclusion.
    """

    run_id = _positive_int(run_id, "run id")
    endpoint = f"actions/runs/{run_id}"
    identity = _run_identity(reader._request_json("GET", endpoint), run_id=run_id)
    _, attempt, branch, head, status, conclusion, _ = identity
    attempt_endpoint = f"{endpoint}/attempts/{attempt}/jobs"
    jobs: list[dict[str, Any]] = []
    seen: set[int] = set()
    expected_total: int | None = None
    page = 1
    while True:
        raw_page = _mapping(
            reader._request_json(
                "GET",
                attempt_endpoint,
                query={"per_page": str(_PAGE_SIZE), "page": str(page)},
            ),
            "attempt jobs page",
        )
        total = raw_page.get("total_count")
        if type(total) is not int or not 1 <= total <= _MAX_JOBS:
            raise ValueError("attempt job total_count must be bounded and nonzero")
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise ValueError("attempt job total_count changed between pages")
        records = raw_page.get("jobs")
        remaining = total - len(jobs)
        if not isinstance(records, list) or len(records) != min(_PAGE_SIZE, remaining):
            raise ValueError("attempt job pagination is incomplete or inconsistent")
        for raw_job in records:
            item = _job_identity(
                raw_job, run_id=run_id, attempt=attempt, branch=branch, head=head
            )
            if item["id"] in seen:
                raise ValueError("duplicate workflow job id")
            seen.add(item["id"])
            jobs.append(item)
        if len(jobs) == total:
            break
        page += 1

    # This is a race detector, not a remote snapshot transaction.
    if _run_identity(reader._request_json("GET", endpoint), run_id=run_id) != identity:
        raise ValueError("workflow run changed during attempt observation")
    if status == "completed" and any(item["status"] != "completed" for item in jobs):
        raise ValueError("completed run contains unfinished attempt jobs")
    if status == "completed" and conclusion == "success" and any(
        item["conclusion"] not in {"success", "skipped", "neutral"} for item in jobs
    ):
        raise ValueError("successful run contains failed or cancelled attempt jobs")
    return {
        "schema": "github_workflow_attempt_observation_v1",
        "run_id": run_id,
        "run_attempt": attempt,
        "head_branch": branch,
        "head_sha": head,
        "status": status,
        "conclusion": conclusion,
        "job_count": len(jobs),
        "jobs": sorted(jobs, key=lambda item: item["id"]),
        "job_source": "attempt_scoped_only",
        "scope": "single_run_attempt",
        "workflow_activity_checked": False,
        "task_registry_complete": False,
        "cleanup_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Run an explicit read-only audit of exactly one GitHub Actions attempt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="shuntatsu/trade_rl")
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args(argv)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("GH_TOKEN or GITHUB_TOKEN is required for GitHub workflow audit")

    # Keep the mutable GitHub client implementation out of pure coordination code.
    # This entrypoint only calls its GET method, never write/cleanup APIs.
    from tools.branch_hygiene import GitHubApi

    reader = GitHubApi(
        repository=args.repository, token=token, api_url="https://api.github.com"
    )
    report = observe_workflow_attempt(reader, run_id=args.run_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = ["observe_workflow_attempt"]


if __name__ == "__main__":
    raise SystemExit(main())
