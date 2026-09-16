from __future__ import annotations

import argparse
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
API_VERSION = "2022-11-28"


@dataclass(frozen=True)
class BranchInfo:
    name: str
    sha: str
    protected: bool


@dataclass(frozen=True)
class OpenPullRequestHead:
    name: str
    sha: str


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
        allow_not_found: bool = False,
    ) -> Any:
        request = urllib.request.Request(
            self._url(suffix, query),
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "trade-rl-branch-hygiene",
                "X-GitHub-Api-Version": API_VERSION,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            if allow_not_found and exc.code == 404:
                return None
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

    def open_pull_request_heads(self) -> tuple[OpenPullRequestHead, ...]:
        result: list[OpenPullRequestHead] = []
        for item in self._list_pages("pulls", state="open"):
            if not isinstance(item, dict):
                raise RuntimeError("pull request entry is not an object")
            head = item.get("head")
            if not isinstance(head, dict):
                raise RuntimeError("pull request head is not an object")
            head_repo = head.get("repo")
            if not isinstance(head_repo, dict):
                continue
            if head_repo.get("full_name") != self._repository:
                continue
            name = head.get("ref")
            sha = head.get("sha")
            if not isinstance(name, str) or not isinstance(sha, str):
                raise RuntimeError("pull request head has an invalid shape")
            result.append(OpenPullRequestHead(name=name, sha=sha))
        return tuple(result)

    def current_branch(self, name: str) -> BranchInfo | None:
        encoded = urllib.parse.quote(name, safe="")
        payload = self._request_json(
            "GET", f"branches/{encoded}", allow_not_found=True
        )
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise RuntimeError("branch lookup is not an object")
        commit = payload.get("commit")
        branch_name = payload.get("name")
        protected = payload.get("protected")
        if not isinstance(commit, dict):
            raise RuntimeError("branch lookup commit is not an object")
        sha = commit.get("sha")
        if (
            not isinstance(branch_name, str)
            or not isinstance(sha, str)
            or not isinstance(protected, bool)
        ):
            raise RuntimeError("branch lookup has an invalid shape")
        return BranchInfo(name=branch_name, sha=sha, protected=protected)

    def delete_branch(self, name: str) -> None:
        encoded = urllib.parse.quote(name, safe="/")
        self._request_json("DELETE", f"git/refs/heads/{encoded}")


def is_preserved_branch(name: str) -> bool:
    return name.startswith(PRESERVED_PREFIXES)


def anchor_branch_names(
    branches: Sequence[BranchInfo],
    *,
    default_branch: str,
    open_pull_request_heads: Sequence[OpenPullRequestHead],
) -> frozenset[str]:
    open_names = {head.name for head in open_pull_request_heads}
    return frozenset(
        branch.name
        for branch in branches
        if branch.name == default_branch
        or branch.protected
        or branch.name in open_names
        or is_preserved_branch(branch.name)
    )


def anchor_shas(
    branches: Sequence[BranchInfo],
    *,
    default_branch: str,
    open_pull_request_heads: Sequence[OpenPullRequestHead],
) -> frozenset[str]:
    names = anchor_branch_names(
        branches,
        default_branch=default_branch,
        open_pull_request_heads=open_pull_request_heads,
    )
    shas = {branch.sha for branch in branches if branch.name in names}
    shas.update(head.sha for head in open_pull_request_heads)
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
    open_pull_request_heads: Sequence[OpenPullRequestHead],
    reachable_from_anchors: frozenset[str],
) -> tuple[BranchDecision, ...]:
    open_names = {head.name for head in open_pull_request_heads}
    decisions: list[BranchDecision] = []
    for branch in sorted(branches, key=lambda item: item.name):
        if branch.name == default_branch:
            decision = BranchDecision(branch, "keep", "default-branch")
        elif branch.protected:
            decision = BranchDecision(branch, "keep", "protected")
        elif branch.name in open_names:
            decision = BranchDecision(branch, "keep", "open-pr-head")
        elif is_preserved_branch(branch.name):
            decision = BranchDecision(branch, "keep", "research-provenance")
        elif branch.sha in reachable_from_anchors:
            decision = BranchDecision(branch, "delete", "tip-reachable-from-anchor")
        else:
            decision = BranchDecision(branch, "keep", "unique-unmerged-tip")
        decisions.append(decision)
    return tuple(decisions)


def apply_cleanup(
    api: GitHubApi,
    decisions: Sequence[BranchDecision],
) -> tuple[str, ...]:
    open_names = {head.name for head in api.open_pull_request_heads()}
    deleted: list[str] = []
    for decision in decisions:
        if decision.action != "delete":
            continue
        expected = decision.branch
        if expected.name in open_names:
            continue
        current = api.current_branch(expected.name)
        if current is None:
            continue
        if current.protected or current.sha != expected.sha:
            continue
        api.delete_branch(expected.name)
        deleted.append(expected.name)
    return tuple(deleted)


def render_summary(
    decisions: Sequence[BranchDecision], *, deleted: Sequence[str], apply: bool
) -> str:
    delete_candidates = [item.branch.name for item in decisions if item.action == "delete"]
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
        f"- unique/unmerged branches kept: {len(unique)}",
        f"- safe deletion candidates: {len(delete_candidates)}",
        f"- branches deleted: {len(deleted)}",
        "",
    ]
    if deleted:
        lines.extend(["### Deleted", ""])
        lines.extend(f"- `{name}`" for name in deleted[:100])
        if len(deleted) > 100:
            lines.append(f"- … and {len(deleted) - 100} more")
        lines.append("")
    if unique:
        lines.extend(["### Kept because the tip is not reachable from a durable anchor", ""])
        lines.extend(f"- `{name}`" for name in unique[:100])
        if len(unique) > 100:
            lines.append(f"- … and {len(unique) - 100} more")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete only remote branches whose tip is already reachable from a durable anchor."
    )
    parser.add_argument("--repository", required=True, help="GitHub repository in owner/name form")
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
    open_heads = api.open_pull_request_heads()
    anchors = anchor_shas(
        branches,
        default_branch=default_branch,
        open_pull_request_heads=open_heads,
    )
    reachable = reachable_commits(anchors)
    decisions = plan_cleanup(
        branches,
        default_branch=default_branch,
        open_pull_request_heads=open_heads,
        reachable_from_anchors=reachable,
    )
    deleted = apply_cleanup(api, decisions) if args.apply else ()
    summary = render_summary(decisions, deleted=deleted, apply=args.apply)
    print(summary)
    if args.summary is not None:
        args.summary.write_text(summary + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
