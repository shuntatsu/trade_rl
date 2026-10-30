"""Read-only GitHub Issue / PR / branch inventory, never cleanup authority."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from tools.agent_repo.coordination.github_state import parse_status_comment
from tools.branch_hygiene import GitHubApi

_SHA = re.compile(r"^[0-9a-f]{40}$")
_TASK_MARKER = "<!-- agent-coordination:"


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{name} must be an exact commit SHA")
    return value


def _number(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def classify_ancestry(compare: dict[str, Any]) -> str:
    """Interpret the GitHub compare endpoint for PR_HEAD...CURRENT_MAIN."""
    ahead = compare.get("ahead_by")
    behind = compare.get("behind_by")
    if (
        isinstance(ahead, bool)
        or isinstance(behind, bool)
        or not isinstance(ahead, int)
        or not isinstance(behind, int)
        or ahead < 0
        or behind < 0
    ):
        raise ValueError("compare ahead/behind counts must be nonnegative integers")
    if ahead == 0 and behind == 0:
        return "identical"
    if ahead > 0 and behind == 0:
        return "head_in_main"
    if ahead == 0 and behind > 0:
        return "main_in_head"
    return "diverged"


class GitHubWorkApi(GitHubApi):
    """Network adapter outside the network-free agent_repo package."""

    def __init__(self, *, repository: str, token: str) -> None:
        super().__init__(
            repository=repository, token=token, api_url="https://api.github.com"
        )
        self.repository = repository

    def open_pulls(self) -> list[Any]:
        return self._list_pages("pulls", state="open")

    def open_issues(self) -> list[Any]:
        return self._list_pages("issues", state="open")

    def issue_comments(self, issue_number: int) -> list[Any]:
        return self._list_pages(f"issues/{issue_number}/comments")

    def compare_head_to_main(self, head_sha: str, main_sha: str) -> str:
        head = _sha(head_sha, "head_sha")
        main = _sha(main_sha, "main_sha")
        payload = self._request_json("GET", f"compare/{head}...{main}")
        return classify_ancestry(_mapping(payload, "compare"))


def build_inventory(api: GitHubWorkApi, *, compare_prs: bool = False) -> dict[str, Any]:
    """Make a deterministic, non-authoritative snapshot. Never return delete eligibility."""
    default = api.repository_default_branch()
    branches = api.branches()
    main_candidates = [branch for branch in branches if branch.name == default]
    if len(main_candidates) != 1:
        raise RuntimeError("default branch is absent or ambiguous")
    main_sha = _sha(main_candidates[0].sha, "main SHA")

    open_names = {default, "provenance/branch-retention"}
    pulls: list[dict[str, Any]] = []
    for item in api.open_pulls():
        pr = _mapping(item, "open PR")
        number = _number(pr.get("number"), "PR number")
        head = _mapping(pr.get("head"), "PR head")
        base = _mapping(pr.get("base"), "PR base")
        base_repo = _mapping(base.get("repo"), "PR base repository")
        if base_repo.get("full_name") != api.repository:
            raise ValueError("PR base points at a different repository")
        base_name = base.get("ref")
        if not isinstance(base_name, str) or not base_name:
            raise ValueError("PR base branch missing")
        open_names.add(base_name)
        head_sha = _sha(head.get("sha"), "PR head SHA")
        head_repo = head.get("repo")
        if isinstance(head_repo, dict) and head_repo.get("full_name") == api.repository:
            head_name = head.get("ref")
            if not isinstance(head_name, str) or not head_name:
                raise ValueError("local PR head branch missing")
            open_names.add(head_name)
        pulls.append(
            {
                "number": number,
                "head_sha": head_sha,
                "base": base_name,
                "relation_to_main": (
                    api.compare_head_to_main(head_sha, main_sha)
                    if compare_prs
                    else "not_checked"
                ),
            }
        )

    task_branches: set[str] = set()
    task_ids: set[str] = set()
    open_issues = 0
    for item in api.open_issues():
        issue = _mapping(item, "open Issue")
        if "pull_request" in issue:
            continue
        issue_number = _number(issue.get("number"), "Issue number")
        open_issues += 1
        for entry in api.issue_comments(issue_number):
            comment = _mapping(entry, "Issue comment")
            body = comment.get("body")
            if not isinstance(body, str):
                raise ValueError("Issue comment body is missing")
            if _TASK_MARKER not in body:
                continue
            try:
                record = parse_status_comment(body)
            except ValueError as exc:
                raise RuntimeError(
                    f"Issue #{issue_number} contains an invalid Coordinator projection"
                ) from exc
            if record.task_id in task_ids:
                raise RuntimeError(f"duplicate Coordinator Task ID: {record.task_id}")
            task_ids.add(record.task_id)
            if record.lease_branch is not None:
                task_branches.add(record.lease_branch)

    observed_names = set(open_names) | task_branches
    branch_rows = []
    for branch in sorted(branches, key=lambda b: b.name):
        if branch.name == default:
            reason = "default"
        elif branch.name == "provenance/branch-retention":
            reason = "retention"
        elif branch.protected:
            reason = "github_protected"
        elif branch.name in open_names:
            reason = "open_pr_ref"
        elif branch.name in task_branches:
            reason = "coordinator_projection"
        else:
            reason = "unclassified"
        branch_rows.append(
            {"name": branch.name, "sha": _sha(branch.sha, "branch SHA"), "reason": reason}
        )

    return {
        "schema": "github_work_inventory_v1",
        "repository": api.repository,
        "main_sha": main_sha,
        "open_issue_count": open_issues,
        "open_pr_count": len(pulls),
        "task_projection_count": len(task_ids),
        "task_scope": "open_issue_comments_only",
        "task_registry_complete": False,
        "workflow_activity_checked": False,
        "cleanup_authorized": False,
        "pull_requests": sorted(pulls, key=lambda pr: pr["number"]),
        "branches": branch_rows,
        "unclassified_branches": sorted(
            b.name
            for b in branches
            if b.name not in observed_names and not b.protected
        ),
        "warning": (
            "This is a read-only, incomplete Task projection. Closed Issue leases, "
            "unregistered work, active workflow runs, and coordinator ownership "
            "have not been independently verified. Never use as deletion authority."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="shuntatsu/trade_rl")
    parser.add_argument("--compare-prs", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("GH_TOKEN or GITHUB_TOKEN is required for GitHub inventory")
    api = GitHubWorkApi(repository=args.repository, token=token)
    print(json.dumps(build_inventory(api, compare_prs=args.compare_prs), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
