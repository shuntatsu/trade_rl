"""Explicit, single-branch pruning of fully merged and retained PR source refs.

This is deliberately separate from bulk branch hygiene. No automatic cleanup,
Task registration, research archival, or remote deletion is authorized by a
merely reachable commit. A requested ref must be the exact final HEAD of an
already merged local PR, reachable from both main and durable retention, past
observation grace and absent from current open PRs and unfinished workflows.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from tools.branch_hygiene import (
    RETENTION_BRANCH,
    BranchInfo,
    GitHubApi,
    open_pull_request_branch_names,
)

_REPOSITORY = "shuntatsu/trade_rl"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_PRESERVED = (
    "provenance/",
    "research/",
    "seal/",
    "freeze/",
    "run/",
    "evidence/",
    "study/",
    "archive/",
    "release/",
    "artifact/",
)
_RETENTION_PREFIXES = (
    "Observe non-anchor branch tips before cleanup grace period",
    "Archive unique non-anchor branch tips before ref cleanup",
)
_GRACE = timedelta(hours=1)


@dataclass(frozen=True)
class VerifiedMergedPrune:
    """Bounded evidence for a redundant PR branch, not Task cleanup authority."""

    branch: str
    expected_sha: str
    main_sha: str
    retention_sha: str
    merged_pr_number: int
    deleted: bool = False


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a GitHub response object")
    return value


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not _SHA_RE.fullmatch(value)
        or value == "0" * 40
    ):
        raise ValueError(f"{field} must be a nonzero lowercase 40-hex Git SHA")
    return value


def _request_inputs(
    repository: str,
    branch: str,
    expected_sha: str,
    merged_pr_number: int,
    retention_record_sha: str,
    now: datetime,
) -> None:
    if repository != _REPOSITORY:
        raise ValueError("repository must be the explicitly authorized trade_rl")
    if (
        not isinstance(branch, str)
        or not _BRANCH_RE.fullmatch(branch)
        or branch in {"main", RETENTION_BRANCH}
        or branch.startswith(_PRESERVED)
        or "//" in branch
        or ".." in branch
        or any(part.startswith(".") or part.endswith(".lock") for part in branch.split("/"))
    ):
        raise ValueError("branch is not eligible for merged-PR pruning")
    _sha(expected_sha, "expected branch SHA")
    _sha(retention_record_sha, "retention record SHA")
    if type(merged_pr_number) is not int or merged_pr_number < 1:
        raise ValueError("merged PR number must be a positive native integer")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be an aware datetime")


def _branches(api: GitHubApi) -> dict[str, BranchInfo]:
    items = api.branches()
    if not items:
        raise ValueError("branch inventory is incomplete")
    seen: dict[str, BranchInfo] = {}
    for item in items:
        if not isinstance(item, BranchInfo):
            raise ValueError("branch inventory has invalid entry")
        if item.name in seen:
            raise ValueError("branch inventory contains duplicate names")
        _sha(item.sha, "branch inventory SHA")
        if type(item.protected) is not bool:
            raise ValueError("branch inventory protection is unknown")
        seen[item.name] = item
    return seen


def _require_ancestor(api: GitHubApi, older: str, newer: str, label: str) -> None:
    result = _mapping(
        api._request_json("GET", f"compare/{older}...{newer}"), label
    )
    ahead, behind = result.get("ahead_by"), result.get("behind_by")
    merge_base = _mapping(result.get("merge_base_commit"), "merge base")
    if (
        type(ahead) is not int
        or type(behind) is not int
        or ahead < 0
        or behind != 0
        or merge_base.get("sha") != older
        or result.get("status") not in ("ahead", "identical")
    ):
        raise ValueError(f"{label} ancestor condition not verified")


def _require_merged_pr(
    api: GitHubApi, branch: str, sha: str, number: int
) -> None:
    pr = _mapping(api._request_json("GET", f"pulls/{number}"), "merged PR")
    head = _mapping(pr.get("head"), "merged PR head")
    base = _mapping(pr.get("base"), "merged PR base")
    head_repo = _mapping(head.get("repo"), "merged PR head repository")
    base_repo = _mapping(base.get("repo"), "merged PR base repository")
    if (
        type(pr.get("number")) is not int
        or pr["number"] != number
        or pr.get("state") != "closed"
        or pr.get("merged") is not True
        or not isinstance(pr.get("merged_at"), str)
        or not pr["merged_at"]
        or not _SHA_RE.fullmatch(str(pr.get("merge_commit_sha")))
        or head.get("ref") != branch
        or head.get("sha") != sha
        or head_repo.get("full_name") != _REPOSITORY
        or base.get("ref") != "main"
        or base_repo.get("full_name") != _REPOSITORY
    ):
        raise ValueError("merged PR must be closed and bind the exact local source HEAD")


def _require_retention_record(
    api: GitHubApi,
    *,
    record_sha: str,
    branch: str,
    sha: str,
    now: datetime,
) -> None:
    record = _mapping(
        api._request_json("GET", f"git/commits/{record_sha}"),
        "retention record",
    )
    message = record.get("message")
    parents = record.get("parents")
    committer = _mapping(record.get("committer"), "retention committer")
    if (
        record.get("sha") != record_sha
        or not isinstance(message, str)
        or not message.startswith(_RETENTION_PREFIXES)
        or f"{branch}\t{sha}" not in message.splitlines()
        or not isinstance(parents, list)
        or len(parents) < 2
        or not all(isinstance(p, dict) for p in parents)
        or sha not in [p.get("sha") for p in parents[1:]]
    ):
        raise ValueError("retention record must bind the exact branch tip as a parent")
    when_text = committer.get("date")
    if not isinstance(when_text, str):
        raise ValueError("retention record has no timestamp")
    try:
        observed = datetime.fromisoformat(when_text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("retention record timestamp is invalid") from error
    if observed.tzinfo is None or now - observed < _GRACE:
        raise ValueError("retention observation grace period not elapsed")


def _observe(
    api: GitHubApi,
    *,
    branch: str,
    expected_sha: str,
    merged_pr_number: int,
    retention_record_sha: str,
    now: datetime,
) -> VerifiedMergedPrune:
    default = api.repository_default_branch()
    if default != "main":
        raise ValueError("main default branch differs from policy")
    by_name = _branches(api)
    source = by_name.get(branch)
    main = by_name.get(default)
    retention = by_name.get(RETENTION_BRANCH)
    if source is None or source.sha != expected_sha:
        raise ValueError("branch tip changed or is missing")
    if source.protected or main is None or retention is None:
        raise ValueError("protected branch or missing main/retention anchor")
    if source.name in open_pull_request_branch_names(api.open_pull_request_refs()):
        raise ValueError("branch remains referenced by an open PR")
    if branch in api.active_workflow_branch_names():
        raise ValueError("branch remains referenced by unfinished workflow")
    _require_merged_pr(api, branch, expected_sha, merged_pr_number)
    _require_ancestor(api, expected_sha, main.sha, "main ancestor")
    _require_ancestor(api, expected_sha, retention.sha, "retention tip")
    _require_ancestor(api, retention_record_sha, retention.sha, "retention record")
    _require_retention_record(
        api,
        record_sha=retention_record_sha,
        branch=branch,
        sha=expected_sha,
        now=now,
    )
    return VerifiedMergedPrune(
        branch=branch,
        expected_sha=expected_sha,
        main_sha=main.sha,
        retention_sha=retention.sha,
        merged_pr_number=merged_pr_number,
    )


def require_origin_checkout(repository: str) -> None:
    """Never push a branch deletion to an unrelated working-tree origin."""

    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=False,
    )
    allowed = {
        f"https://github.com/{repository}",
        f"https://github.com/{repository}.git",
    }
    if result.returncode != 0 or result.stdout.strip() not in allowed:
        raise RuntimeError("origin must be the exact authenticated HTTPS repository")


def prune_verified_merged_branch(
    api: GitHubApi,
    *,
    repository: str,
    branch: str,
    expected_sha: str,
    merged_pr_number: int,
    retention_record_sha: str,
    now: datetime,
    apply: bool = False,
) -> VerifiedMergedPrune:
    """Preview by default; explicitly apply one exact-SHA leased branch deletion."""

    _request_inputs(
        repository, branch, expected_sha, merged_pr_number, retention_record_sha, now
    )
    first = _observe(
        api,
        branch=branch,
        expected_sha=expected_sha,
        merged_pr_number=merged_pr_number,
        retention_record_sha=retention_record_sha,
        now=now,
    )
    if not apply:
        return first

    require_origin_checkout(repository)
    second = _observe(
        api,
        branch=branch,
        expected_sha=expected_sha,
        merged_pr_number=merged_pr_number,
        retention_record_sha=retention_record_sha,
        now=now,
    )
    if first != second:
        raise ValueError("main or retention snapshot changed during verification")

    # GitHubApi uses git push --atomic --force-with-lease on this exact SHA.
    # There is no unconditional ref deletion or automatic retry on conflict.
    api.delete_branches_with_leases((BranchInfo(branch, expected_sha, False),))
    after = _branches(api)
    if branch in after:
        raise RuntimeError("deleted branch is still present after readback")
    if (
        after.get("main") is None
        or after["main"].sha != first.main_sha
        or after.get(RETENTION_BRANCH) is None
        or after[RETENTION_BRANCH].sha != first.retention_sha
    ):
        raise RuntimeError("main or retention changed after deletion; audit required")
    return replace(first, deleted=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explicit single merged-PR branch prune; preview is the default."
    )
    parser.add_argument("--repository", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--merged-pr-number", type=int, required=True)
    parser.add_argument("--retention-record-sha", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--api-url", default="https://api.github.com")
    parser.add_argument(
        "--token", default=os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    )
    args = parser.parse_args(argv)
    if not args.token:
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required")
    api = GitHubApi(repository=args.repository, token=args.token, api_url=args.api_url)
    result = prune_verified_merged_branch(
        api,
        repository=args.repository,
        branch=args.branch,
        expected_sha=args.expected_sha,
        merged_pr_number=args.merged_pr_number,
        retention_record_sha=args.retention_record_sha,
        now=datetime.now(timezone.utc),
        apply=args.apply,
    )
    state = "DELETED" if result.deleted else "ELIGIBLE PREVIEW (no deletion)"
    print(
        f"{state}: {result.branch}@{result.expected_sha} "
        f"PR #{result.merged_pr_number} main={result.main_sha} "
        f"retention={result.retention_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
