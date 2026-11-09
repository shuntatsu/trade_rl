"""Read-only GitHub Issue / PR / branch inventory, never cleanup authority."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from tools.agent_repo.coordination.github_state import parse_status_comment
from tools.branch_hygiene import BranchInfo, GitHubApi
from tools.github_workflow_ref_guard import stable_workflow_branch_refs

_SHA = re.compile(r"^[0-9a-f]{40}$")
_TASK_MARKER = "<!-- agent-coordination:"
_MAX_TASK_ISSUE_COMMENTS = 1000


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


def _issue_identity(value: Any, repository: str) -> tuple[int, int, str, str, int, str]:
    """Fix Issue identity, current state and comment count for later readback."""

    issue = _mapping(value, "open Issue")
    number = _number(issue.get("number"), "Issue number")
    issue_id = _number(issue.get("id"), "Issue id")
    node_id = issue.get("node_id")
    repository_url = issue.get("repository_url")
    if (
        not isinstance(repository_url, str)
        or repository_url.casefold()
        != f"https://api.github.com/repos/{repository}".casefold()
    ):
        raise ValueError("open Issue repository identity differs")
    state = issue.get("state")
    count = issue.get("comments")
    updated_at = issue.get("updated_at")
    if not isinstance(node_id, str) or not node_id.strip() or state != "open":
        raise ValueError("open Issue identity or state is invalid")
    if type(count) is not int or count < 0 or count > _MAX_TASK_ISSUE_COMMENTS:
        raise ValueError("open Issue comments count is invalid or unbounded")
    if not isinstance(updated_at, str) or not updated_at.strip():
        raise ValueError("open Issue updated_at is missing")
    return (number, issue_id, node_id, state, count, updated_at)


def _comment_snapshot(
    api: GitHubWorkApi, issue_number: int, expected_count: int
) -> tuple[tuple[int, int, str, str], ...]:
    """Require exact comment cardinality and unchanged identity, author and body."""

    comments = api.issue_comments(issue_number)
    if not isinstance(comments, list) or len(comments) != expected_count:
        raise RuntimeError(f"Issue #{issue_number} comments count differs from Issue")
    identities: list[tuple[int, int, str, str]] = []
    seen: set[int] = set()
    for raw in comments:
        comment = _mapping(raw, "Issue comment")
        comment_id = _number(comment.get("id"), "Issue comment id")
        if comment_id in seen:
            raise RuntimeError(f"Issue #{issue_number} contains duplicate comment IDs")
        seen.add(comment_id)
        publisher = _mapping(comment.get("user"), "Issue comment publisher")
        publisher_id = _number(publisher.get("id"), "Issue comment publisher id")
        body = comment.get("body")
        updated_at = comment.get("updated_at")
        if (
            not isinstance(body, str)
            or not isinstance(updated_at, str)
            or not updated_at.strip()
        ):
            raise ValueError("Issue comment body or updated_at is malformed")
        identities.append((comment_id, publisher_id, body, updated_at))
    return tuple(identities)


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
        comparison_status = "identical"
        relationship = "identical"
    elif ahead > 0 and behind == 0:
        comparison_status = "ahead"
        relationship = "head_in_main"
    elif ahead == 0 and behind > 0:
        comparison_status = "behind"
        relationship = "main_in_head"
    else:
        comparison_status = "diverged"
        relationship = "diverged"
    if compare.get("status") != comparison_status:
        raise ValueError("GitHub compare status conflicts with ahead/behind counts")
    return relationship


def _pull_identity(
    value: Any, repository: str
) -> tuple[int, str, str, str, str | None]:
    """Stable reference identity, ignoring mutable title/updated_at metadata."""
    pr = _mapping(value, "open PR")
    number = _number(pr.get("number"), "PR number")
    head = _mapping(pr.get("head"), "PR head")
    base = _mapping(pr.get("base"), "PR base")
    base_repo = _mapping(base.get("repo"), "PR base repository")
    base_full_name = base_repo.get("full_name")
    if (
        not isinstance(base_full_name, str)
        or base_full_name.casefold() != repository.casefold()
    ):
        raise ValueError("PR base points at a different repository")
    base_name = base.get("ref")
    if not isinstance(base_name, str) or not base_name:
        raise ValueError("PR base branch missing")
    base_sha = _sha(base.get("sha"), "PR base SHA")
    head_sha = _sha(head.get("sha"), "PR head SHA")
    head_name: str | None = None
    head_repo = head.get("repo")
    if (
        isinstance(head_repo, dict)
        and isinstance(head_repo.get("full_name"), str)
        and head_repo["full_name"].casefold() == repository.casefold()
    ):
        raw_name = head.get("ref")
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("local PR head branch missing")
        head_name = raw_name
    return number, head_sha, base_name, base_sha, head_name


class GitHubWorkApi:
    """Read-only public adapter; the underlying shared client is not a security boundary."""

    def __init__(self, *, repository: str, token: str) -> None:
        self._reader = GitHubApi(
            repository=repository, token=token, api_url="https://api.github.com"
        )
        self.repository = repository

    def repository_default_branch(self) -> str:
        return self._reader.repository_default_branch()

    def branches(self) -> tuple[BranchInfo, ...]:
        return self._reader.branches()

    def open_pulls(self) -> list[Any]:
        return self._reader._list_pages("pulls", state="open")

    def open_issues(self) -> list[Any]:
        return self._reader._list_pages("issues", state="open")

    def issue_comments(self, issue_number: int) -> list[Any]:
        _number(issue_number, "Issue number")
        items: list[Any] = []
        # A full 1000-comment list needs one terminal empty page to prove that
        # the caller was not silently given a truncated pagination result.
        for page in range(1, _MAX_TASK_ISSUE_COMMENTS // 100 + 2):
            response = self._reader._request_json(
                "GET",
                f"issues/{issue_number}/comments",
                query={"per_page": "100", "page": str(page)},
            )
            if not isinstance(response, list) or len(response) > 100:
                raise RuntimeError("Issue comment page has invalid size or shape")
            if page > _MAX_TASK_ISSUE_COMMENTS // 100:
                if response:
                    raise RuntimeError("Issue comment enumeration exceeds limit")
                return items
            items.extend(response)
            if len(response) < 100:
                return items
        raise RuntimeError("Issue comment enumeration is incomplete")

    def unfinished_workflow_branch_names(self) -> frozenset[str]:
        return self._reader.active_workflow_branch_names()

    def compare_head_to_main(self, head_sha: str, main_sha: str) -> str:
        head = _sha(head_sha, "head_sha")
        main = _sha(main_sha, "main_sha")
        payload = self._reader._request_json("GET", f"compare/{head}...{main}")
        return classify_ancestry(_mapping(payload, "compare"))


def build_inventory(
    api: GitHubWorkApi,
    *,
    compare_prs: bool = False,
    include_workflow_refs: bool = False,
) -> dict[str, Any]:
    """Make a deterministic, non-authoritative snapshot. Never return delete eligibility."""
    default = api.repository_default_branch()
    branches = api.branches()
    main_candidates = [branch for branch in branches if branch.name == default]
    if len(main_candidates) != 1:
        raise RuntimeError("default branch is absent or ambiguous")
    main_sha = _sha(main_candidates[0].sha, "main SHA")

    branch_tips = {branch.name: _sha(branch.sha, "branch SHA") for branch in branches}
    if len(branch_tips) != len(branches):
        raise RuntimeError("branch list contains duplicate names")

    open_names = {default, "provenance/branch-retention"}
    pulls: list[dict[str, Any]] = []
    pull_identities: list[tuple[int, str, str, str, str | None]] = []
    seen_pr_numbers: set[int] = set()
    for item in api.open_pulls():
        number, head_sha, base_name, base_sha, head_name = _pull_identity(
            item, api.repository
        )
        if number in seen_pr_numbers:
            raise RuntimeError(f"duplicate open PR number: {number}")
        seen_pr_numbers.add(number)
        pull_identities.append((number, head_sha, base_name, base_sha, head_name))
        open_names.add(base_name)
        if head_name is not None:
            if branch_tips.get(head_name) != head_sha:
                raise RuntimeError(f"local PR head tip is absent or changed: #{number}")
            open_names.add(head_name)
        # A forked or unavailable head is not comparable as a local branch tip.
        # Unknown ancestry remains explicit; this inventory never permits cleanup.
        relation_to_main = "not_checked"
        if compare_prs:
            relation_to_main = (
                api.compare_head_to_main(head_sha, main_sha)
                if head_name is not None
                else "nonlocal_head_uncompared"
            )
        pulls.append(
            {
                "number": number,
                "head_sha": head_sha,
                "head_ref": head_name,
                "base": base_name,
                "base_sha": base_sha,
                "relation_to_main": relation_to_main,
            }
        )

    task_branches: set[str] = set()
    task_ids: set[str] = set()
    open_issues = 0
    issue_identities: dict[int, tuple[int, int, str, str, int, str]] = {}
    issue_comments: dict[int, tuple[tuple[int, int, str, str], ...]] = {}
    seen_issue_ids: set[int] = set()
    seen_issue_node_ids: set[str] = set()
    seen_comment_ids: set[int] = set()
    for item in api.open_issues():
        issue = _mapping(item, "open Issue")
        if "pull_request" in issue:
            continue
        identity = _issue_identity(issue, api.repository)
        issue_number = identity[0]
        if issue_number in issue_identities:
            raise RuntimeError(f"duplicate open Issue number: {issue_number}")
        if identity[1] in seen_issue_ids or identity[2] in seen_issue_node_ids:
            raise RuntimeError("open Issue identity is duplicated across numbers")
        seen_issue_ids.add(identity[1])
        seen_issue_node_ids.add(identity[2])
        issue_identities[issue_number] = identity
        open_issues += 1
        comments = _comment_snapshot(api, issue_number, identity[4])
        for comment_id, _author, _body, _timestamp in comments:
            if comment_id in seen_comment_ids:
                raise RuntimeError("duplicate Issue comment ID across open Issues")
            seen_comment_ids.add(comment_id)
        issue_comments[issue_number] = comments
        for _id, _publisher, body, _updated_at in comments:
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

    # The second/third Issue reads and full comment reread catch ordinary
    # changes before publication, but are not an atomic GitHub snapshot.
    if sorted(pull_identities) != sorted(
        _pull_identity(item, api.repository) for item in api.open_pulls()
    ):
        raise RuntimeError("open PR refs changed during inventory")
    expected_issues = sorted(issue_identities.values())

    def reread_issue_identities() -> list[tuple[int, int, str, str, int, str]]:
        rows = [
            _issue_identity(issue, api.repository)
            for issue in (_mapping(item, "open Issue") for item in api.open_issues())
            if "pull_request" not in issue
        ]
        if len(rows) != len(set(row[0] for row in rows)):
            raise RuntimeError("duplicate open Issue number during readback")
        return sorted(rows)

    if reread_issue_identities() != expected_issues:
        raise RuntimeError("open Issue roster or metadata changed during inventory")
    for issue_number, first in sorted(issue_comments.items()):
        second = _comment_snapshot(api, issue_number, issue_identities[issue_number][4])
        if first != second:
            raise RuntimeError(
                f"Issue #{issue_number} comments changed during inventory"
            )
    if reread_issue_identities() != expected_issues:
        raise RuntimeError("open Issue roster or metadata changed during inventory")
    if tuple(sorted(api.branches(), key=lambda b: b.name)) != tuple(
        sorted(branches, key=lambda b: b.name)
    ):
        raise RuntimeError("branch tips or protection changed during inventory")

    workflow_refs: tuple[str, ...] = ()
    if include_workflow_refs:
        workflow_refs = stable_workflow_branch_refs(
            api, known_branch_names=frozenset(branch_tips)
        )
        if tuple(sorted(api.branches(), key=lambda b: b.name)) != tuple(
            sorted(branches, key=lambda b: b.name)
        ):
            raise RuntimeError("branch tips or protection changed during workflow scan")

    observed_names = set(open_names) | task_branches | set(workflow_refs)
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
            reason = "unverified_task_comment"
        elif branch.name in workflow_refs:
            reason = "unfinished_workflow_ref"
        else:
            reason = "unclassified"
        branch_rows.append(
            {
                "name": branch.name,
                "sha": _sha(branch.sha, "branch SHA"),
                "reason": reason,
            }
        )

    report: dict[str, Any] = {
        "schema": "github_work_inventory_v1",
        "repository": api.repository,
        "main_sha": main_sha,
        "open_issue_count": open_issues,
        "open_pr_count": len(pulls),
        "task_projection_count": len(task_ids),
        "task_projection_trusted": False,
        "task_scope": "open_issue_comments_only",
        "pr_base_sha_semantics": "historical_ref_not_current_branch_tip",
        "task_registry_complete": False,
        "workflow_activity_checked": False,
        "cleanup_authorized": False,
        "pull_requests": sorted(pulls, key=lambda pr: pr["number"]),
        "branches": branch_rows,
        "unclassified_branches": sorted(
            b.name for b in branches if b.name not in observed_names and not b.protected
        ),
        "warning": (
            "This is a read-only, incomplete Task projection. The author of a "
            "Coordinator-shaped Issue comment has not been authenticated. Closed "
            "Issue leases, unregistered work, active workflow runs, and coordinator "
            "ownership remain unverified; a stable readback is not an atomic "
            "snapshot. Never use as deletion authority."
        ),
    }
    if include_workflow_refs:
        report["unfinished_workflow_branches"] = list(workflow_refs)
        report["unfinished_workflow_branch_names_checked"] = True
        report["warning"] += (
            " Optional double-read checks unfinished workflow branch names only; "
            "run attempts, jobs, artifacts, closed Task leases and authenticated "
            "registry completeness are still unverified."
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="shuntatsu/trade_rl")
    parser.add_argument("--compare-prs", action="store_true")
    parser.add_argument("--include-workflow-refs", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("GH_TOKEN or GITHUB_TOKEN is required for GitHub inventory")
    api = GitHubWorkApi(repository=args.repository, token=token)
    print(
        json.dumps(
            build_inventory(
                api,
                compare_prs=args.compare_prs,
                include_workflow_refs=args.include_workflow_refs,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
