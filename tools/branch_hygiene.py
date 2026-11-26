from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeGuard

API_VERSION = "2022-11-28"
_RETIREMENT_ORIGIN_URLS = frozenset(
    {
        "https://github.com/shuntatsu/trade_rl",
        "https://github.com/shuntatsu/trade_rl.git",
    }
)


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
class MergedPullRequestSource:
    name: str
    sha: str


@dataclass(frozen=True)
class ConsolidatedPullRequestSource:
    """A closed exact source introduced by an authenticated normal main merge."""

    name: str
    sha: str
    source_pr_number: int
    carrier_pr_number: int


@dataclass(frozen=True)
class BranchDecision:
    branch: BranchInfo
    action: str
    reason: str


class _WorkflowRosterChanged(RuntimeError):
    """A paginated workflow-status roster was inconsistent during observation."""


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
        base_url = f"{self._api_url}/repos/{owner}/{name}"
        path = suffix.lstrip("/")
        url = f"{base_url}/{path}" if path else base_url
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

    def merged_pr_source_refs(
        self, *, include_nested: bool = False
    ) -> tuple[MergedPullRequestSource, ...]:
        """Enumerate exact merged local PR final heads (including nested only on request).

        Normal retirement continues to require direct-main merged PRs. A scoped,
        archive-pinned one-shot may include PRs merged first into an intermediate
        branch, but canonical planning still demands that the exact final source
        SHA be reachable from current main.

        A newly created branch at the main tip is main-reachable but not
        necessarily finished. Only a matching merged PR can retire that ref.
        """

        result: list[MergedPullRequestSource] = []
        for item in self._list_pages("pulls", state="closed"):
            if not isinstance(item, dict) or item.get("state") != "closed":
                raise RuntimeError("closed pull request roster is malformed")
            merged_at = item.get("merged_at")
            if merged_at is None:
                continue
            if not isinstance(merged_at, str) or not merged_at:
                raise RuntimeError("merged pull request timestamp is malformed")
            head = item.get("head")
            base = item.get("base")
            if not isinstance(head, dict) or not isinstance(base, dict):
                raise RuntimeError("merged PR head/base is malformed")
            base_repo = base.get("repo")
            if not isinstance(base_repo, dict):
                raise RuntimeError("merged PR base repository is missing")
            if base_repo.get("full_name") != self._repository or (
                not include_nested and base.get("ref") != "main"
            ):
                continue
            head_repo = head.get("repo")
            if head_repo is None:
                # Deleted heads cannot match a currently existing source ref.
                continue
            if not isinstance(head_repo, dict):
                raise RuntimeError("merged PR head repository is malformed")
            if head_repo.get("full_name") != self._repository:
                continue
            name, sha = head.get("ref"), head.get("sha")
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(sha, str)
                or not re.fullmatch(r"[0-9a-f]{40}", sha)
            ):
                raise RuntimeError("merged PR source ref is malformed")
            result.append(MergedPullRequestSource(name, sha))
        return tuple(result)

    def consolidated_pr_source_refs(
        self, branches: Sequence[BranchInfo], *, main_sha: str
    ) -> tuple[ConsolidatedPullRequestSource, ...]:
        """Prove closed constituent tips entered main through a merged carrier PR.

        Never infer retirement from a title, a close reason, or patch equality.
        The first main first-parent commit containing the exact source must be
        the actual two-parent merge of a local merged PR targeting main.
        """
        if self._repository != "shuntatsu/trade_rl" or not _is_nonzero_oid(main_sha):
            raise ValueError("consolidated retirement requires trade_rl/main identity")
        # Read both source and carrier identities from one closed roster.
        # Re-run this proof for each bounded deletion batch.
        branches_by_name = {branch.name: branch.sha for branch in branches}
        if len(branches_by_name) != len(branches):
            raise ValueError("duplicate branch inventory")
        seen: set[int] = set()
        sources: list[tuple[str, str, int]] = []
        carriers: dict[tuple[str, str], int] = {}
        for item in self._list_pages("pulls", state="closed"):
            if not isinstance(item, dict) or item.get("state") != "closed":
                raise RuntimeError("closed pull request roster is malformed")
            number = item.get("number")
            if type(number) is not int or number <= 0:
                raise RuntimeError("closed pull request number is malformed")
            if number in seen:
                raise RuntimeError("duplicate closed pull request number")
            seen.add(number)
            head, base = item.get("head"), item.get("base")
            if not isinstance(head, dict) or not isinstance(base, dict):
                raise RuntimeError("closed pull request refs are malformed")
            head_repo, base_repo = head.get("repo"), base.get("repo")
            if head_repo is None:
                continue
            if not isinstance(head_repo, dict) or not isinstance(base_repo, dict):
                raise RuntimeError("closed pull request repository is malformed")
            if (
                head_repo.get("full_name") != self._repository
                or base_repo.get("full_name") != self._repository
            ):
                continue
            name, sha, base_name = head.get("ref"), head.get("sha"), base.get("ref")
            if (
                not isinstance(name, str)
                or not name
                or not _is_nonzero_oid(sha)
                or not isinstance(base_name, str)
                or not base_name
                or "merged_at" not in item
            ):
                raise RuntimeError("closed local PR source ref is malformed")
            merged_at = item["merged_at"]
            if merged_at is not None and (
                not isinstance(merged_at, str) or not merged_at
            ):
                raise RuntimeError("merged pull request timestamp is malformed")
            if merged_at is not None and base_name == "main":
                merge_sha = item.get("merge_commit_sha")
                if not _is_nonzero_oid(merge_sha):
                    raise RuntimeError("merged main PR commit identity is malformed")
                # GitHub may mark multiple constituent PRs merged at one
                # commit. Only the row matching the actual second parent
                # can serve as the carrier; stable ties need no inference.
                key = (merge_sha, sha)
                carriers[key] = min(number, carriers.get(key, number))
            elif branches_by_name.get(name) == sha:
                sources.append((name, sha, number))
        return _consolidated_source_proofs(main_sha, sources, carriers)

    def active_workflow_branch_names(self) -> frozenset[str]:
        """Return only a complete roster; never combine pages across retries."""
        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                return self._active_workflow_branch_names_once()
            except _WorkflowRosterChanged:
                if attempt == max_attempts - 1:
                    raise
                # GitHub status transitions can shift total_count while paging.
                # Restart every status/page; total retry delay is bounded at 25.4 s.
                # Persistent inconsistency still aborts *before* any ref deletion.
                time.sleep(0.2 * (2**attempt))
        raise AssertionError("unreachable workflow roster retry state")

    def _active_workflow_branch_names_once(self) -> frozenset[str]:
        names: set[str] = set()
        for status in ("requested", "waiting", "pending", "queued", "in_progress"):
            page = 1
            expected_total: int | None = None
            seen_run_ids: set[int] = set()
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
                count = payload.get("total_count")
                if not isinstance(runs, list):
                    raise RuntimeError("workflow_runs is not a list")
                if type(count) is not int or count < 0:
                    raise RuntimeError("workflow run total_count is missing or invalid")
                if count >= 1000:
                    raise RuntimeError(
                        "workflow run search meets GitHub's 1000-result limit"
                    )
                if expected_total is None:
                    expected_total = count
                elif count != expected_total:
                    raise _WorkflowRosterChanged(
                        "workflow run total_count changed between pages"
                    )
                remaining = count - (page - 1) * 100
                if remaining < 0 or len(runs) != min(100, remaining):
                    raise _WorkflowRosterChanged(
                        "workflow run pagination is incomplete or exceeds total_count"
                    )
                for run in runs:
                    if not isinstance(run, dict):
                        raise RuntimeError("workflow run entry is not an object")
                    run_id = run.get("id")
                    if type(run_id) is not int or run_id <= 0:
                        raise RuntimeError("workflow run id is missing or invalid")
                    if run_id in seen_run_ids:
                        raise RuntimeError("duplicate workflow run id within status")
                    seen_run_ids.add(run_id)
                    if run.get("status") != status:
                        raise RuntimeError(
                            "workflow run status differs from requested status"
                        )
                    head_branch = run.get("head_branch")
                    if not isinstance(head_branch, str) or not head_branch.strip():
                        raise RuntimeError(
                            "unfinished workflow run is missing its head_branch"
                        )
                    names.add(head_branch)
                # A full page cannot prove exhaustion, even if total_count agrees.
                if len(seen_run_ids) == count and len(runs) < 100:
                    break
                page += 1
        return frozenset(names)

    def delete_branches_with_leases(self, branches: Sequence[BranchInfo]) -> None:
        if not branches:
            return
        if self._repository != "shuntatsu/trade_rl":
            raise ValueError("branch deletion is restricted to trade_rl")
        # Revalidate the exact effective Git configuration at each batch.
        # The child environment replaces GIT_CONFIG_COUNT, so inspecting the
        # inherited environment alone could hide a different push destination.
        env = _retirement_push_env(self._token)
        require_retirement_origin(env=env)
        command = [
            "git",
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
            env=env,
        )
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(f"conditional branch deletion failed: {detail}")


def _retirement_push_env(token: str) -> dict[str, str]:
    credential = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode(
        "ascii"
    )
    # Keep the reversible credential out of argv. Use this same environment
    # for destination validation and for the subsequent push.
    return {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {credential}",
    }


def open_pull_request_branch_names(
    refs: Sequence[OpenPullRequestRefs],
) -> frozenset[str]:
    names = {ref.base_name for ref in refs}
    names.update(ref.head_name for ref in refs if ref.head_name is not None)
    return frozenset(names)


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


def _is_nonzero_oid(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{40}", value) is not None
        and value != "0" * 40
    )


def _git_history(*args: str) -> str:
    return subprocess.check_output(["git", "--no-replace-objects", *args], text=True)


def _consolidated_source_proofs(
    main_sha: str,
    sources: Sequence[tuple[str, str, int]],
    carriers: dict[tuple[str, str], int],
) -> tuple[ConsolidatedPullRequestSource, ...]:
    """Find the first main commit introducing each closed source's history.

    Binary search is valid because each first-parent step contains its
    predecessor's history. A source already on the main first-parent chain
    is never eligible via this alternate proof (e.g. an abandoned no-op PR).
    """
    if not sources:
        return ()
    if _git_history("rev-parse", "--is-shallow-repository").strip() != "false":
        raise RuntimeError("complete Git history required for consolidated retirement")
    history = [
        tuple(line.split())
        for line in _git_history(
            "log", "--first-parent", "--reverse", "--format=%H %P", main_sha
        ).splitlines()
    ]
    if (
        not history
        or len(history[0]) != 1
        or history[-1][0] != main_sha
        or any(not _is_nonzero_oid(oid) for row in history for oid in row)
        or any(
            len(row) < 2 or row[1] != previous[0]
            for previous, row in zip(history, history[1:])
        )
    ):
        raise RuntimeError(
            "incomplete first-parent history for consolidated retirement"
        )
    reachable = frozenset(_git_history("rev-list", main_sha).splitlines())
    first_parent = {row[0] for row in history}
    proof_by_sha: dict[str, int | None] = {}

    for _, sha, _ in sources:
        if sha in proof_by_sha or sha not in reachable or sha in first_parent:
            continue
        low, high = 0, len(history) - 1
        while low < high:
            middle = (low + high) // 2
            result = subprocess.run(
                [
                    "git",
                    "--no-replace-objects",
                    "merge-base",
                    "--is-ancestor",
                    sha,
                    history[middle][0],
                ],
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                high = middle
            elif result.returncode == 1:
                low = middle + 1
            else:
                raise RuntimeError("could not establish source commit ancestry")
        row = history[low]
        # A normal merge brings in exactly its second parent's additional
        # ancestry. Its actual merge OID and second parent must be the PR's
        # merge_commit_sha and final head, respectively.
        proof_by_sha[sha] = carriers.get((row[0], row[2])) if len(row) == 3 else None
    return tuple(
        ConsolidatedPullRequestSource(name, sha, number, carrier_number)
        for name, sha, number in sources
        if (carrier_number := proof_by_sha.get(sha)) is not None
    )


# Routine retirement never depends on the separate Coordinator enrollment,
# because an exact ancestor of current main cannot contain unique code history.
# Divergent sources, authoritative research refs and any currently used PR/run
# stay outside this path. No alternate retention/grace planner is active.
# M1 merge handoff requires durable source branch refs even when their
# exact final head is already reachable from main. This exception is scoped
# to the M1 branch namespace plus its Windows test-only companion.
M1_SOURCE_REF_PREFIX = "fix/m1-"
M1_ADDITIONAL_SOURCE_REFS = frozenset({"fix/windows-job-child-pid-marker-20261105"})


RETIRED_MERGED_PRESERVED_PREFIXES = (
    "provenance/",
    "research/",
    "infra/",
    "seal/",
    "freeze/",
    "run/",
    "evidence/",
    "study/",
    "archive/",
    "release/",
    "artifact/",
    "coordination/",
    "task/",
    "lease/",
)


def plan_merged_retirement(
    branches: Sequence[BranchInfo],
    *,
    default_branch: str,
    open_pull_request_refs: Sequence[OpenPullRequestRefs],
    active_workflow_branches: frozenset[str],
    reachable_from_main: frozenset[str],
    merged_pr_sources: Sequence[MergedPullRequestSource] = (),
    consolidated_pr_sources: Sequence[ConsolidatedPullRequestSource] = (),
) -> tuple[BranchDecision, ...]:
    """Retire exact integrated PR sources, never unclaimed branches."""

    names: set[str] = set()
    mains = 0
    for branch in branches:
        if (
            not isinstance(branch.name, str)
            or not branch.name
            or branch.name in names
            or not isinstance(branch.sha, str)
            or not re.fullmatch(r"[0-9a-f]{40}", branch.sha)
            or type(branch.protected) is not bool
        ):
            raise ValueError("invalid or duplicate branch inventory")
        names.add(branch.name)
        mains += branch.name == default_branch
    if mains != 1 or not reachable_from_main:
        raise ValueError(
            "exactly one default branch and complete main history required"
        )

    open_names = open_pull_request_branch_names(open_pull_request_refs)
    merged_sources = {(source.name, source.sha) for source in merged_pr_sources}
    consolidated_sources = {
        (source.name, source.sha) for source in consolidated_pr_sources
    }
    decisions = []
    for branch in sorted(branches, key=lambda entry: entry.name):
        if branch.name == default_branch:
            reason = "default-branch"
        elif branch.protected:
            reason = "protected"
        elif branch.name in open_names:
            reason = "open-pr-ref"
        elif branch.name in active_workflow_branches:
            reason = "active-workflow"
        elif (
            branch.name.startswith(M1_SOURCE_REF_PREFIX)
            or branch.name in M1_ADDITIONAL_SOURCE_REFS
        ):
            reason = "m1-source-preservation"
        elif branch.name.startswith(RETIRED_MERGED_PRESERVED_PREFIXES):
            reason = "reserved-research-or-provenance"
        elif branch.sha not in reachable_from_main:
            reason = "not-in-main-history"
        elif (branch.name, branch.sha) in merged_sources:
            decisions.append(
                BranchDecision(branch, "delete", "exact-merged-pr-in-main")
            )
            continue
        elif (branch.name, branch.sha) in consolidated_sources:
            decisions.append(
                BranchDecision(branch, "delete", "exact-closed-pr-consolidated-in-main")
            )
            continue
        else:
            reason = "no-exact-merged-pr-source"
        decisions.append(BranchDecision(branch, "keep", reason))
    return tuple(decisions)


def _name_summary(category: str, names: set[str]) -> str:
    """Bound diagnostic size without hiding the total number of differing names."""
    sample = [
        ascii(name[:120] + ("..." if len(name) > 120 else ""))
        for name in sorted(names)[:5]
    ]
    return f"{category}={len(names)} [{', '.join(sample)}]"


def _readback_differences(
    expected: dict[str, BranchInfo], actual: dict[str, BranchInfo]
) -> str:
    shared = expected.keys() & actual.keys()
    return "; ".join(
        (
            _name_summary("missing", expected.keys() - actual.keys()),
            _name_summary("unexpected", actual.keys() - expected.keys()),
            _name_summary(
                "sha_changed",
                {name for name in shared if expected[name].sha != actual[name].sha},
            ),
            _name_summary(
                "protection_changed",
                {
                    name
                    for name in shared
                    if expected[name].protected != actual[name].protected
                },
            ),
        )
    )


def _unique_branch_readback(branches: Sequence[BranchInfo]) -> dict[str, BranchInfo]:
    """Reject ambiguous paginated rosters before comparing their contents."""
    by_name: dict[str, BranchInfo] = {}
    duplicates: set[str] = set()
    for branch in branches:
        if branch.name in by_name:
            duplicates.add(branch.name)
        by_name[branch.name] = branch
    if duplicates:
        raise RuntimeError(
            "duplicate branch name in retirement readback: "
            + _name_summary("duplicate_names", duplicates)
        )
    return by_name


def apply_merged_retirement(
    api: GitHubApi,
    decisions: Sequence[BranchDecision],
    *,
    default_branch: str,
    expected_main_sha: str,
    reachable_from_main: frozenset[str],
    max_deletes: int = 100,
    batch_size: int = 10,
    include_nested_merged_prs: bool = False,
    include_consolidated_prs: bool = False,
) -> tuple[str, ...]:
    """Delete only exact, revalidated integrated PR main ancestors using atomic SHA leases.

    Before *each* batch a fresh PR, workflow and branch inventory must still
    authorize every candidate. After each batch every removed ref is read back.
    A stale snapshot aborts rather than applying an old authorization.
    """

    if api._repository != "shuntatsu/trade_rl" or default_branch != "main":
        raise ValueError("merged retirement is restricted to trade_rl/main")
    if type(include_nested_merged_prs) is not bool:
        raise ValueError("nested PR evidence switch must be boolean")
    if type(include_consolidated_prs) is not bool:
        raise ValueError("consolidated PR evidence switch must be boolean")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", expected_main_sha)
        or type(max_deletes) is not int
        or not 1 <= max_deletes <= 100
        or type(batch_size) is not int
        or not 1 <= batch_size <= 20
        or expected_main_sha not in reachable_from_main
    ):
        raise ValueError("invalid retirement bounds or main evidence")

    expected = {entry.branch.name: entry.branch for entry in decisions}
    if len(expected) != len(decisions):
        raise ValueError("duplicate planned branches")
    main = expected.get(default_branch)
    if main is None or main.sha != expected_main_sha:
        raise ValueError("main changed before retirement")

    queue = [entry.branch for entry in decisions if entry.action == "delete"]
    if any(entry.sha not in reachable_from_main for entry in queue):
        raise ValueError("unmerged branch in retirement plan")
    queue = queue[:max_deletes]
    # Closed/merged PR identities are stable: no need to re-download the entire
    # historical 1k+ PR roster before every bounded branch batch.
    if include_nested_merged_prs and queue:
        merged_sources = api.merged_pr_source_refs(include_nested=True)
    else:
        merged_sources = api.merged_pr_source_refs() if queue else ()
    deleted: list[str] = []
    while queue:
        before = _unique_branch_readback(api.branches())
        if before != expected or before.get(default_branch) != main:
            raise RuntimeError(
                "branch or main changed before retirement: "
                + _readback_differences(expected, before)
            )
        # A closed constituent PR can be reopened. Obtain its exact source
        # and carrier evidence again for this batch, not from the preview.
        consolidated = (
            api.consolidated_pr_source_refs(
                tuple(before.values()), main_sha=expected_main_sha
            )
            if include_consolidated_prs
            else ()
        )
        open_refs = api.open_pull_request_refs()
        running = api.active_workflow_branch_names()
        fresh = plan_merged_retirement(
            tuple(before.values()),
            default_branch=default_branch,
            open_pull_request_refs=open_refs,
            active_workflow_branches=running,
            reachable_from_main=reachable_from_main,
            merged_pr_sources=merged_sources,
            consolidated_pr_sources=consolidated,
        )
        eligible = {entry.branch for entry in fresh if entry.action == "delete"}
        batch = queue[:batch_size]
        if any(branch not in eligible for branch in batch):
            raise RuntimeError("PR/workflow/protection changed before retirement")
        if include_consolidated_prs:
            # The extra history and closed-PR proof can take longer than
            # ordinary retirement. Do not carry its initial ref observation
            # across that work into a deletion without another readback.
            final_refs = _unique_branch_readback(api.branches())
            if final_refs != before:
                raise RuntimeError(
                    "branch or main changed during consolidated proof: "
                    + _readback_differences(before, final_refs)
                )
        # The git transport performs one atomic push with one expected SHA
        # lease per ref: concurrent source writes cannot be silently discarded.
        api.delete_branches_with_leases(tuple(batch))
        after = _unique_branch_readback(api.branches())
        expected_after = {
            name: branch
            for name, branch in before.items()
            if name not in {candidate.name for candidate in batch}
        }
        if after != expected_after:
            raise RuntimeError(
                "post-delete readback differs from exact expected refs: "
                + _readback_differences(expected_after, after)
            )
        deleted.extend(branch.name for branch in batch)
        expected = after
        queue = queue[len(batch) :]
    return tuple(deleted)


def require_retirement_origin(*, env: dict[str, str] | None = None) -> None:
    """Do not let an exact-SHA lease target an unrelated checkout or remote."""

    # A standalone preflight must inspect the same effective Git settings
    # as the eventual transport; the empty token is never used for network I/O.
    if env is None:
        env = _retirement_push_env("")
    for argv in (
        ("git", "remote", "get-url", "--all", "origin"),
        ("git", "remote", "get-url", "--push", "--all", "origin"),
    ):
        actual = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stdout.splitlines()
        if len(actual) != 1 or actual[0] not in _RETIREMENT_ORIGIN_URLS:
            raise RuntimeError(
                "merged retirement requires the exact trade_rl HTTPS origin"
            )


def render_merged_retirement_summary(
    decisions: Sequence[BranchDecision], *, deleted: Sequence[str], apply: bool
) -> str:
    candidates = [item for item in decisions if item.action == "delete"]
    remaining = len(candidates) - len(deleted)
    return "\n".join(
        (
            "## Main-merged branch retirement",
            f"- mode: {'apply' if apply else 'dry-run'}",
            f"- inspected: {len(decisions)}",
            f"- eligible main ancestors at observation: {len(candidates)}",
            f"- deleted and verified: {len(deleted)}",
            f"- remaining eligible (may require a fresh check): {remaining}",
            "- unmerged, open-PR, active-workflow, protected and research refs are retained",
            *(f"- retired: `{name}`" for name in deleted),
            "",
        )
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "--apply is disabled without --retire-merged. "
            "Inspect merged-PR main-ancestor refs using exact SHA checks."
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
    parser.add_argument(
        "--apply",
        action="store_true",
        help="disabled without --retire-merged (main ancestor refs only)",
    )
    parser.add_argument("--retire-merged", action="store_true")
    parser.add_argument(
        "--include-consolidated",
        action="store_true",
        help="also verify closed constituent sources via exact normal main merge ancestry",
    )
    parser.add_argument("--max-deletes", type=int, default=100)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.retire_merged:
        raise RuntimeError(
            "obsolete retention planner disabled: bulk apply is disabled; use --retire-merged "
            "for a read-only preview or --retire-merged --apply for exact-SHA retirement"
        )
    if not args.token:
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required")
    api = GitHubApi(repository=args.repository, token=args.token, api_url=args.api_url)
    default_branch = api.repository_default_branch()
    branches = api.branches()
    open_refs = api.open_pull_request_refs()
    active_branches = api.active_workflow_branch_names()
    if args.repository != "shuntatsu/trade_rl" or default_branch != "main":
        raise RuntimeError("merged branch retirement is limited to trade_rl/main")
    main_refs = [branch for branch in branches if branch.name == default_branch]
    if len(main_refs) != 1:
        raise RuntimeError("cannot identify unique main")
    # The workflow fetches the complete main history before this check.
    local_main = subprocess.run(
        ["git", "rev-parse", f"refs/remotes/origin/{default_branch}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if local_main != main_refs[0].sha:
        raise RuntimeError("local main history differs from current GitHub ref")
    reachable_from_main = reachable_commits((local_main,))
    merged_sources = api.merged_pr_source_refs()
    consolidated_sources = (
        api.consolidated_pr_source_refs(branches, main_sha=local_main)
        if args.include_consolidated
        else ()
    )
    decisions = plan_merged_retirement(
        branches,
        default_branch=default_branch,
        open_pull_request_refs=open_refs,
        active_workflow_branches=active_branches,
        reachable_from_main=reachable_from_main,
        merged_pr_sources=merged_sources,
        consolidated_pr_sources=consolidated_sources,
    )
    if args.apply:
        require_retirement_origin()
    deleted = (
        apply_merged_retirement(
            api,
            decisions,
            default_branch=default_branch,
            expected_main_sha=local_main,
            reachable_from_main=reachable_from_main,
            max_deletes=args.max_deletes,
            include_consolidated_prs=args.include_consolidated,
        )
        if args.apply
        else ()
    )
    summary = render_merged_retirement_summary(
        decisions, deleted=deleted, apply=args.apply
    )
    print(summary)
    if args.summary is not None:
        args.summary.write_text(summary + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
