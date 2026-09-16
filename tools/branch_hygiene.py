from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PRESERVED_PREFIXES = ("research/", "seal/", "freeze/", "run/")
TRANSIENT_PREFIXES = ("verify/", "automation/", "tmp/")
RETENTION_BRANCH = "provenance/branch-retention"
RETENTION_BATCH_SIZE = 20
DELETE_BATCH_SIZE = 25
API_VERSION = "2022-11-28"


@dataclass(frozen=True)
class BranchInfo:
    name: str
    sha: str
    protected: bool


@dataclass(frozen=True)
class OpenPullRequestRefs:
    head_name: str | None
    head_sha: str | None
    base_name: str
    base_sha: str


@dataclass(frozen=True)
class BranchDecision:
    branch: BranchInfo
    action: str
    reason: str


class GitHubApi:
    def __init__(self, *, repository: str, token: str, api_url: str) -> None:
        owner, separator, name = repository.partition("/")
        if not separator or not owner or not name or "/" in name:
            raise ValueError("repository must use owner/name form")
        self._repository = repository
        self._owner = owner
        self._name = name
        self._token = token
        self._api_url = api_url.rstrip("/")

    def _url(self, suffix: str, query: dict[str, str] | None = None) -> str:
        owner = urllib.parse.quote(self._owner, safe="")
        name = urllib.parse.quote(self._name, safe="")
        url = f"{self._api_url}/repos/{owner}/{name}/{suffix.lstrip('/')}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        return url

    def _request_json(
        self,
        method: str,
        suffix: str,
        *,
        query: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        data = None
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token}",
            "User-Agent": "trade-rl-branch-hygiene",
            "X-GitHub-Api-Version": API_VERSION,
        }
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self._url(suffix, query),
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GitHub API {method} {suffix} failed: HTTP {exc.code}: {detail}"
            ) from exc
        if not payload:
            return None
        return json.loads(payload.decode("utf-8"))

    def _list_pages(self, suffix: str, *, state: str | None = None) -> list[Any]:
        items: list[Any] = []
        page = 1
        while True:
            query = {"per_page": "100", "page": str(page)}
            if state is not None:
                query["state"] = state
            payload = self._request_json("GET", suffix, query=query)
            if not isinstance(payload, list):
                raise RuntimeError(f"GitHub API {suffix} did not return a list")
            items.extend(payload)
            if len(payload) < 100:
                return items
            page += 1

    def repository_default_branch(self) -> str:
        payload = self._request_json("GET", "")
        if not isinstance(payload, dict):
            raise RuntimeError("repository metadata is not an object")
        default_branch = payload.get("default_branch")
        if not isinstance(default_branch, str) or not default_branch:
            raise RuntimeError("repository default_branch is missing")
        return default_branch

    def branches(self) -> tuple[BranchInfo, ...]:
        result: list[BranchInfo] = []
        for item in self._list_pages("branches"):
            if not isinstance(item, dict):
                raise RuntimeError("branch entry is not an object")
            commit = item.get("commit")
            if not isinstance(commit, dict):
                raise RuntimeError("branch commit entry is not an object")
            name = item.get("name")
            sha = commit.get("sha")
            protected = item.get("protected")
            if (
                not isinstance(name, str)
                or not isinstance(sha, str)
                or not isinstance(protected, bool)
            ):
                raise RuntimeError("branch entry has an invalid shape")
            result.append(BranchInfo(name=name, sha=sha, protected=protected))
        return tuple(result)

    def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
        result: list[OpenPullRequestRefs] = []
        for item in self._list_pages("pulls", state="open"):
            if not isinstance(item, dict):
                raise RuntimeError("pull request entry is not an object")
            head = item.get("head")
            base = item.get("base")
            if not isinstance(head, dict) or not isinstance(base, dict):
                raise RuntimeError("pull request ref entry is not an object")

            base_repo = base.get("repo")
            if not isinstance(base_repo, dict):
                raise RuntimeError("pull request base repository is missing")
            if base_repo.get("full_name") != self._repository:
                raise RuntimeError("pull request base repository does not match target")
            base_name = base.get("ref")
            base_sha = base.get("sha")
            if not isinstance(base_name, str) or not isinstance(base_sha, str):
                raise RuntimeError("pull request base ref has an invalid shape")

            head_name: str | None = None
            head_sha: str | None = None
            head_repo = head.get("repo")
            if (
                isinstance(head_repo, dict)
                and head_repo.get("full_name") == self._repository
            ):
                raw_head_name = head.get("ref")
                raw_head_sha = head.get("sha")
                if not isinstance(raw_head_name, str) or not isinstance(
                    raw_head_sha, str
                ):
                    raise RuntimeError("pull request head ref has an invalid shape")
                head_name = raw_head_name
                head_sha = raw_head_sha

            result.append(
                OpenPullRequestRefs(
                    head_name=head_name,
                    head_sha=head_sha,
                    base_name=base_name,
                    base_sha=base_sha,
                )
            )
        return tuple(result)

    def active_workflow_branch_names(self) -> frozenset[str]:
        names: set[str] = set()
        for status in ("in_progress", "queued"):
            page = 1
            while True:
                payload = self._request_json(
                    "GET",
                    "actions/runs",
                    query={
                        "status": status,
                        "per_page": "100",
                        "page": str(page),
                    },
                )
                if not isinstance(payload, dict):
                    raise RuntimeError("workflow run list is not an object")
                runs = payload.get("workflow_runs")
                if not isinstance(runs, list):
                    raise RuntimeError("workflow_runs is not a list")
                for run in runs:
                    if not isinstance(run, dict):
                        raise RuntimeError("workflow run entry is not an object")
                    head_branch = run.get("head_branch")
                    if isinstance(head_branch, str) and head_branch:
                        names.add(head_branch)
                if len(runs) < 100:
                    break
                page += 1
        return frozenset(names)

    def git_commit_tree_sha(self, commit_sha: str) -> str:
        encoded = urllib.parse.quote(commit_sha, safe="")
        payload = self._request_json("GET", f"git/commits/{encoded}")
        if not isinstance(payload, dict):
            raise RuntimeError("git commit lookup is not an object")
        tree = payload.get("tree")
        if not isinstance(tree, dict):
            raise RuntimeError("git commit tree is missing")
        tree_sha = tree.get("sha")
        if not isinstance(tree_sha, str) or not tree_sha:
            raise RuntimeError("git commit tree SHA is missing")
        return tree_sha

    def create_retention_commit(
        self,
        *,
        tree_sha: str,
        parent_shas: Sequence[str],
        message: str,
    ) -> str:
        payload = self._request_json(
            "POST",
            "git/commits",
            body={"message": message, "tree": tree_sha, "parents": list(parent_shas)},
        )
        if not isinstance(payload, dict):
            raise RuntimeError("created git commit is not an object")
        sha = payload.get("sha")
        if not isinstance(sha, str) or not sha:
            raise RuntimeError("created git commit SHA is missing")
        return sha

    def create_branch_ref(self, name: str, sha: str) -> None:
        self._request_json(
            "POST",
            "git/refs",
            body={"ref": f"refs/heads/{name}", "sha": sha},
        )

    def update_branch_ref(self, name: str, sha: str) -> None:
        encoded = urllib.parse.quote(name, safe="/")
        self._request_json(
            "PATCH",
            f"git/refs/heads/{encoded}",
            body={"sha": sha, "force": False},
        )

    def delete_branches_with_leases(self, branches: Sequence[BranchInfo]) -> None:
        if not branches:
            return
        credential = base64.b64encode(
            f"x-access-token:{self._token}".encode("utf-8")
        ).decode("ascii")
        command = [
            "git",
            "-c",
            f"http.https://github.com/.extraheader=AUTHORIZATION: basic {credential}",
            "push",
            "--atomic",
            "--porcelain",
        ]
        command.extend(
            f"--force-with-lease=refs/heads/{branch.name}:{branch.sha}"
            for branch in branches
        )
        command.append("origin")
        command.extend(f":refs/heads/{branch.name}" for branch in branches)
        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(f"conditional branch deletion failed: {detail}")


def is_preserved_branch(name: str) -> bool:
    return name == RETENTION_BRANCH or name.startswith(PRESERVED_PREFIXES)


def is_transient_branch(name: str) -> bool:
    return name.startswith(TRANSIENT_PREFIXES)


def open_pull_request_branch_names(
    refs: Sequence[OpenPullRequestRefs],
) -> frozenset[str]:
    names = {ref.base_name for ref in refs}
    names.update(ref.head_name for ref in refs if ref.head_name is not None)
    return frozenset(names)


def anchor_branch_names(
    branches: Sequence[BranchInfo],
    *,
    default_branch: str,
    open_pull_request_refs: Sequence[OpenPullRequestRefs],
    active_workflow_branches: frozenset[str],
) -> frozenset[str]:
    open_names = open_pull_request_branch_names(open_pull_request_refs)
    return frozenset(
        branch.name
        for branch in branches
        if branch.name == default_branch
        or branch.protected
        or branch.name in open_names
        or branch.name in active_workflow_branches
        or is_preserved_branch(branch.name)
    )


def anchor_shas(
    branches: Sequence[BranchInfo],
    *,
    default_branch: str,
    open_pull_request_refs: Sequence[OpenPullRequestRefs],
    active_workflow_branches: frozenset[str],
) -> frozenset[str]:
    names = anchor_branch_names(
        branches,
        default_branch=default_branch,
        open_pull_request_refs=open_pull_request_refs,
        active_workflow_branches=active_workflow_branches,
    )
    shas = {branch.sha for branch in branches if branch.name in names}
    shas.update(ref.base_sha for ref in open_pull_request_refs)
    shas.update(
        ref.head_sha for ref in open_pull_request_refs if ref.head_sha is not None
    )
    if not shas:
        raise RuntimeError("no durable anchor commits were found")
    return frozenset(shas)


def reachable_commits(shas: Iterable[str]) -> frozenset[str]:
    ordered = sorted(set(shas))
    if not ordered:
        raise ValueError("at least one anchor SHA is required")
    process = subprocess.run(
        ["git", "rev-list", "--stdin"],
        input="".join(f"{sha}\n" for sha in ordered),
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"git rev-list failed: {process.stderr.strip()}")
    return frozenset(line for line in process.stdout.splitlines() if line)


def plan_cleanup(
    branches: Sequence[BranchInfo],
    *,
    default_branch: str,
    open_pull_request_refs: Sequence[OpenPullRequestRefs],
    active_workflow_branches: frozenset[str],
    reachable_from_anchors: frozenset[str],
) -> tuple[BranchDecision, ...]:
    open_names = open_pull_request_branch_names(open_pull_request_refs)
    decisions: list[BranchDecision] = []
    for branch in sorted(branches, key=lambda item: item.name):
        if branch.name == default_branch:
            decision = BranchDecision(branch, "keep", "default-branch")
        elif branch.protected:
            decision = BranchDecision(branch, "keep", "protected")
        elif branch.name in open_names:
            decision = BranchDecision(branch, "keep", "open-pr-ref")
        elif branch.name in active_workflow_branches:
            decision = BranchDecision(branch, "keep", "active-workflow")
        elif is_preserved_branch(branch.name):
            decision = BranchDecision(branch, "keep", "provenance")
        elif branch.sha in reachable_from_anchors:
            decision = BranchDecision(branch, "delete", "tip-reachable-from-anchor")
        elif is_transient_branch(branch.name):
            decision = BranchDecision(
                branch, "archive-delete", "transient-unique-tip"
            )
        else:
            decision = BranchDecision(branch, "keep", "unique-unmerged-tip")
        decisions.append(decision)
    return tuple(decisions)


def chunks(items: Sequence[BranchInfo], size: int) -> Iterable[Sequence[BranchInfo]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def retention_message(branches: Sequence[BranchInfo]) -> str:
    lines = [
        "Archive transient branch tips before ref cleanup",
        "",
        "Each mapping below preserves the deleted remote branch name and exact tip SHA.",
        "Restore with: git branch <name> <sha>",
        "",
    ]
    lines.extend(f"{branch.name}\t{branch.sha}" for branch in branches)
    return "\n".join(lines)


def archive_transient_tips(
    api: GitHubApi,
    *,
    default_branch: str,
    current_by_name: dict[str, BranchInfo],
    branches: Sequence[BranchInfo],
) -> str | None:
    if not branches:
        retention = current_by_name.get(RETENTION_BRANCH)
        return retention.sha if retention is not None else None

    default = current_by_name.get(default_branch)
    if default is None:
        raise RuntimeError("default branch disappeared before archival")
    retention = current_by_name.get(RETENTION_BRANCH)
    parent_sha = retention.sha if retention is not None else default.sha
    tree_sha = api.git_commit_tree_sha(parent_sha)
    retention_exists = retention is not None

    for batch in chunks(tuple(branches), RETENTION_BATCH_SIZE):
        unique_tip_shas = tuple(dict.fromkeys(branch.sha for branch in batch))
        commit_sha = api.create_retention_commit(
            tree_sha=tree_sha,
            parent_shas=(parent_sha, *unique_tip_shas),
            message=retention_message(batch),
        )
        if retention_exists:
            api.update_branch_ref(RETENTION_BRANCH, commit_sha)
        else:
            api.create_branch_ref(RETENTION_BRANCH, commit_sha)
            retention_exists = True
        parent_sha = commit_sha

    return parent_sha


def apply_cleanup(
    api: GitHubApi,
    decisions: Sequence[BranchDecision],
    *,
    default_branch: str,
) -> tuple[str, ...]:
    first_snapshot = {branch.name: branch for branch in api.branches()}
    first_open_names = open_pull_request_branch_names(api.open_pull_request_refs())
    first_active_names = api.active_workflow_branch_names()

    archive_candidates: list[BranchInfo] = []
    for decision in decisions:
        if decision.action != "archive-delete":
            continue
        expected = decision.branch
        current = first_snapshot.get(expected.name)
        if (
            expected.name not in first_open_names
            and expected.name not in first_active_names
            and current is not None
            and not current.protected
            and current.sha == expected.sha
        ):
            archive_candidates.append(expected)

    retention_sha = archive_transient_tips(
        api,
        default_branch=default_branch,
        current_by_name=first_snapshot,
        branches=archive_candidates,
    )

    current_by_name = {branch.name: branch for branch in api.branches()}
    if archive_candidates:
        retention = current_by_name.get(RETENTION_BRANCH)
        if retention is None or retention.sha != retention_sha:
            raise RuntimeError("retention branch did not reach the expected archive commit")

    archived_names = {branch.name for branch in archive_candidates}
    delete_candidates: list[BranchInfo] = []
    for decision in decisions:
        if decision.action not in {"delete", "archive-delete"}:
            continue
        expected = decision.branch
        if decision.action == "archive-delete" and expected.name not in archived_names:
            continue
        current = current_by_name.get(expected.name)
        if current is None or current.protected or current.sha != expected.sha:
            continue
        delete_candidates.append(expected)

    deleted: list[str] = []
    for batch in chunks(tuple(delete_candidates), DELETE_BATCH_SIZE):
        open_names = open_pull_request_branch_names(api.open_pull_request_refs())
        active_names = api.active_workflow_branch_names()
        safe_batch = tuple(
            branch
            for branch in batch
            if branch.name not in open_names and branch.name not in active_names
        )
        if not safe_batch:
            continue
        api.delete_branches_with_leases(safe_batch)
        deleted.extend(branch.name for branch in safe_batch)

    remaining_names = {branch.name for branch in api.branches()}
    unexpectedly_remaining = sorted(set(deleted) & remaining_names)
    if unexpectedly_remaining:
        joined = ", ".join(unexpectedly_remaining)
        raise RuntimeError(f"deleted branches still present after cleanup: {joined}")
    return tuple(deleted)


def render_summary(
    decisions: Sequence[BranchDecision], *, deleted: Sequence[str], apply: bool
) -> str:
    direct_candidates = [
        item.branch.name for item in decisions if item.action == "delete"
    ]
    archive_candidates = [
        item.branch.name for item in decisions if item.action == "archive-delete"
    ]
    unique = [
        item.branch.name
        for item in decisions
        if item.action == "keep" and item.reason == "unique-unmerged-tip"
    ]
    anchors = [
        item.branch.name
        for item in decisions
        if item.action == "keep" and item.reason != "unique-unmerged-tip"
    ]
    lines = [
        "## Branch hygiene",
        "",
        f"- mode: {'apply' if apply else 'dry-run'}",
        f"- branches inspected: {len(decisions)}",
        f"- durable anchors kept: {len(anchors)}",
        f"- unique non-transient branches kept: {len(unique)}",
        f"- direct deletion candidates: {len(direct_candidates)}",
        f"- archive-then-delete candidates: {len(archive_candidates)}",
        f"- branches deleted: {len(deleted)}",
        f"- retention branch: `{RETENTION_BRANCH}`",
        "",
    ]
    if deleted:
        lines.extend(["### Deleted", ""])
        lines.extend(f"- `{name}`" for name in deleted[:100])
        if len(deleted) > 100:
            lines.append(f"- … and {len(deleted) - 100} more")
        lines.append("")
    if unique:
        lines.extend(["### Kept unique non-transient branches", ""])
        lines.extend(f"- `{name}`" for name in unique[:100])
        if len(unique) > 100:
            lines.append(f"- … and {len(unique) - 100} more")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete redundant remote branches and archive unique transient tips "
            "before deleting their refs."
        )
    )
    parser.add_argument(
        "--repository", required=True, help="GitHub repository in owner/name form"
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--summary", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.token:
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required")
    api = GitHubApi(repository=args.repository, token=args.token, api_url=args.api_url)
    default_branch = api.repository_default_branch()
    branches = api.branches()
    open_refs = api.open_pull_request_refs()
    active_branches = api.active_workflow_branch_names()
    anchors = anchor_shas(
        branches,
        default_branch=default_branch,
        open_pull_request_refs=open_refs,
        active_workflow_branches=active_branches,
    )
    reachable = reachable_commits(anchors)
    decisions = plan_cleanup(
        branches,
        default_branch=default_branch,
        open_pull_request_refs=open_refs,
        active_workflow_branches=active_branches,
        reachable_from_anchors=reachable,
    )
    deleted = (
        apply_cleanup(api, decisions, default_branch=default_branch)
        if args.apply
        else ()
    )
    summary = render_summary(decisions, deleted=deleted, apply=args.apply)
    print(summary)
    if args.summary is not None:
        args.summary.write_text(summary + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
