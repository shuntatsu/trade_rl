"""Safely remove stale linked git worktrees under ``<repo>/.claude/worktrees``.

Default mode is a read-only plan.  ``--apply`` removes only worktrees that are
classified ``REMOVABLE``: clean (ignored regenerable caches excepted), merged
into ``origin/main``, not an open PR head or branch, not locked, not
mid-operation, free of nested worktrees, symlinks/junctions and non-cache
ignored data, and idle.
The tool never forces a removal, never deletes remote branches and never
touches the stash.  Local branches are deleted only with
``--delete-local-branches`` and only via ``git branch -d``.

Failure behaviour: any uncertainty *before* removal starts (``gh`` failure,
unreadable ``origin/main``, git errors) exits 2 and nothing is removed.  A
failure for one worktree during ``--apply`` is reported, the others are still
processed, and the exit code is 1; an unexpected error after removals began
reports the actual removed count and also exits 1.
If the rename probe cannot restore a directory, the run stops at once (exit 1),
skips ``git worktree prune`` and prints the probe path for manual restore.
While such a leftover ``*.hygiene-probe-*`` directory exists next to an in-scope
worktree, ``--apply`` refuses to start (exit 2); the read-only plan still works.

Limits (heuristics, not guarantees): only a git lock, the worktree the tool
runs in, and the idle window identify live sessions; on Windows a rename probe
additionally detects open handles.  Other sessions are not detected.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

ORIGIN_REF = "origin/main"
SCOPE_PARTS = (".claude", "worktrees")
PR_LIST_LIMIT = 200
DEFAULT_MIN_IDLE_HOURS = 24.0
PROTECTED_BRANCHES = frozenset({"main", "master"})
# Regenerable caches that never count as "ignored data".
REGENERABLE_DIR_NAMES = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
# Gitdir state files/dirs meaning a rebase/merge/cherry-pick/revert/bisect is open.
OPERATION_MARKERS = (
    "rebase-merge",
    "rebase-apply",
    "sequencer",
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "BISECT_LOG",
)
EXAMPLE_PATHS = 3

REMOVABLE = "REMOVABLE"
KEEP = "KEEP"

_SHA = re.compile(r"[0-9a-f]{40}")


class WorktreeHygieneError(RuntimeError):
    """Fail-closed error: nothing may be removed after this is raised."""


class ProbeRestoreError(WorktreeHygieneError):
    """The rename probe could not put a worktree directory back; stop everything."""


@dataclass(frozen=True)
class Worktree:
    path: Path
    head: str | None
    branch: str | None
    locked: bool
    prunable: bool
    bare: bool
    is_main: bool


@dataclass(frozen=True)
class OpenPullRequest:
    number: int
    branch: str
    head_sha: str


@dataclass(frozen=True)
class WorktreeDecision:
    worktree: Worktree
    action: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class ApplyResult:
    path: Path
    outcome: str  # "removed" | "kept" | "failed"
    reason: str
    detail: str = ""
    branch_result: str | None = None


PrHeadsProvider = Callable[[], Sequence[OpenPullRequest]]


def _run(args: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            list(args),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=env,
        )
    except OSError as exc:
        raise WorktreeHygieneError(f"cannot run {args[0]}: {exc}") from exc


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=cwd)


def _git_ok(cwd: Path, *args: str) -> str:
    process = _git(cwd, *args)
    if process.returncode != 0:
        raise WorktreeHygieneError(
            f"git {' '.join(args)} failed: {process.stderr.strip()}"
        )
    return process.stdout


def _norm(path: str | Path) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


def _is_within(child: str | Path, parent: str | Path) -> bool:
    child_norm = _norm(child)
    parent_norm = _norm(parent).rstrip("\\/")
    return child_norm != parent_norm and child_norm.startswith(parent_norm + os.sep)


def parse_worktree_list(text: str) -> list[Worktree]:
    """Parse ``git worktree list --porcelain``; the first entry is the main one."""
    worktrees: list[Worktree] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            fields[key] = value
        if "worktree" not in fields:
            continue
        branch = fields.get("branch")
        if branch is not None and branch.startswith("refs/heads/"):
            branch = branch[len("refs/heads/") :]
        head = fields.get("HEAD")
        worktrees.append(
            Worktree(
                path=Path(os.path.realpath(fields["worktree"])),
                head=head if head is not None and _SHA.fullmatch(head) else None,
                branch=branch,
                locked="locked" in fields,
                prunable="prunable" in fields,
                bare="bare" in fields,
                is_main=not worktrees,
            )
        )
    return worktrees


def list_worktrees(cwd: Path) -> list[Worktree]:
    worktrees = parse_worktree_list(_git_ok(cwd, "worktree", "list", "--porcelain"))
    if not worktrees:
        raise WorktreeHygieneError("git reported no worktrees")
    return worktrees


def current_toplevel(cwd: Path) -> Path:
    return Path(os.path.realpath(_git_ok(cwd, "rev-parse", "--show-toplevel").strip()))


def fetch_origin_main(repo_root: Path) -> None:
    process = _git(repo_root, "fetch", "origin", "main")
    if process.returncode != 0:
        raise WorktreeHygieneError(
            f"git fetch origin main failed: {process.stderr.strip()}"
        )


def resolve_origin_main(repo_root: Path) -> str:
    process = _git(
        repo_root,
        "rev-parse",
        "--verify",
        "--quiet",
        f"refs/remotes/{ORIGIN_REF}^{{commit}}",
    )
    sha = process.stdout.strip()
    if process.returncode != 0 or not _SHA.fullmatch(sha):
        raise WorktreeHygieneError(
            f"cannot resolve {ORIGIN_REF}; run with --fetch or fetch origin first"
        )
    return sha


def gh_open_pull_requests(repo_root: Path) -> list[OpenPullRequest]:
    """Return every open PR head via ``gh``; any doubt raises (fail closed)."""
    process = _run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            str(PR_LIST_LIMIT),
            "--json",
            "number,headRefName,headRefOid",
        ],
        cwd=repo_root,
    )
    if process.returncode != 0:
        raise WorktreeHygieneError(f"gh pr list failed: {process.stderr.strip()}")
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise WorktreeHygieneError(f"gh pr list returned invalid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise WorktreeHygieneError("gh pr list did not return a JSON list")
    if len(payload) >= PR_LIST_LIMIT:
        raise WorktreeHygieneError(
            f"gh pr list returned {len(payload)} PRs (limit {PR_LIST_LIMIT}); "
            "the open-PR roster may be truncated"
        )
    pulls: list[OpenPullRequest] = []
    for item in payload:
        if not isinstance(item, dict):
            raise WorktreeHygieneError("gh pr list returned a malformed entry")
        number = item.get("number")
        branch = item.get("headRefName")
        sha = item.get("headRefOid")
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or not isinstance(branch, str)
            or not isinstance(sha, str)
            or not _SHA.fullmatch(sha.lower())
        ):
            raise WorktreeHygieneError("gh pr list returned a malformed entry")
        pulls.append(OpenPullRequest(number, branch, sha.lower()))
    return pulls


@dataclass(frozen=True)
class _Status:
    tracked: list[str]
    untracked: list[str]
    ignored: list[str]  # ignored entries beyond the regenerable-cache allowlist


def _is_regenerable(entry: str) -> bool:
    name = entry.strip().strip('"').rstrip("/").rsplit("/", 1)[-1]
    return name in REGENERABLE_DIR_NAMES or name.endswith(".pyc")


def _status(worktree: Path) -> _Status:
    output = _git_ok(
        worktree,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--ignored=matching",
    )
    tracked: list[str] = []
    untracked: list[str] = []
    ignored: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        if line.startswith("!! "):
            if not _is_regenerable(line[3:]):
                ignored.append(line[3:].strip().strip('"'))
        elif line.startswith("??"):
            untracked.append(line)
        else:
            tracked.append(line)
    return _Status(tracked, untracked, ignored)


def _ignored_detail(ignored: Sequence[str]) -> str:
    shown = ", ".join(ignored[:EXAMPLE_PATHS])
    more = len(ignored) - EXAMPLE_PATHS
    return f"{len(ignored)} ignored entr(ies) would be deleted: {shown}" + (
        f", +{more} more" if more > 0 else ""
    )


def _is_ancestor(repo_root: Path, sha: str, base_sha: str) -> bool:
    process = _git(repo_root, "merge-base", "--is-ancestor", sha, base_sha)
    if process.returncode == 0:
        return True
    if process.returncode == 1:
        return False
    raise WorktreeHygieneError(f"merge-base failed: {process.stderr.strip()}")


def _operation_in_progress(worktree: Path) -> str | None:
    gitdir = Path(_git_ok(worktree, "rev-parse", "--absolute-git-dir").strip())
    found = [name for name in OPERATION_MARKERS if (gitdir / name).exists()]
    return ", ".join(found) if found else None


@dataclass(frozen=True)
class _Scan:
    nested: Path | None  # first nested .git (file or directory)
    link: Path | None  # first symlink / junction / reparse point
    newest: float | None  # newest mtime of any file or directory


_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _is_link(path: str, info: os.stat_result) -> bool:
    """Symlink, Windows junction or any other reparse point (never followed)."""
    if stat.S_ISLNK(info.st_mode):
        return True
    if getattr(info, "st_file_attributes", 0) & _REPARSE_POINT:
        return True
    return os.path.islink(path) or os.path.isjunction(path)


def _scan_tree(worktree: Path) -> _Scan:
    """Walk the whole worktree (ignored dirs included, root ``.git`` excluded).

    Stops at the first nested ``.git`` or link so a KEEP verdict is cheap.  Links
    are detected with ``lstat`` and never descended into: Git for Windows follows
    junctions during ``git worktree remove`` and would delete their targets.
    """

    def fail(error: OSError) -> None:
        raise error

    newest: float | None = None
    root = _norm(worktree)
    for dirpath, dirnames, filenames in os.walk(worktree, onerror=fail):
        names = [*dirnames, *filenames]
        if _norm(dirpath) == root:
            names = [name for name in names if name != ".git"]
            dirnames[:] = [name for name in dirnames if name != ".git"]
        elif ".git" in names:
            return _Scan(Path(dirpath) / ".git", None, newest)
        for entry in (".", *names):
            path = os.path.join(dirpath, entry)
            info = os.lstat(path)
            if entry != "." and _is_link(path, info):
                return _Scan(None, Path(path), newest)
            if newest is None or info.st_mtime > newest:
                newest = info.st_mtime
    return _Scan(None, None, newest)


def _registered_nested(
    worktree: Worktree, registered: Sequence[Worktree]
) -> Path | None:
    for other in registered:
        if _norm(other.path) != _norm(worktree.path) and _is_within(
            other.path, worktree.path
        ):
            return other.path
    return None


def _classify(
    worktree: Worktree,
    *,
    repo_root: Path,
    scope_dir: Path,
    current: Path,
    registered: Sequence[Worktree],
    origin_main_sha: str,
    pr_by_sha: dict[str, OpenPullRequest],
    pr_by_branch: dict[str, OpenPullRequest],
    min_idle_seconds: float,
    allow_ignored_data: bool,
    now: float,
) -> tuple[str, str]:
    """Return ``(reason, detail)``; reason ``removable`` means REMOVABLE."""
    if worktree.is_main:
        return "main-worktree", "the main worktree is never touched"
    if worktree.bare or not _is_within(worktree.path, scope_dir):
        return "outside-scope", f"not inside {SCOPE_PARTS[0]}/{SCOPE_PARTS[1]}"
    if _norm(worktree.path) == _norm(current):
        return "current-worktree", "the tool is running in this worktree"
    if worktree.locked:
        return "locked", "git worktree lock is set"
    if worktree.prunable or not worktree.path.is_dir():
        return "missing-directory", "directory is gone; `git worktree prune` handles it"
    if worktree.head is None:
        return "head-unreadable", "no HEAD commit reported"
    try:
        operation = _operation_in_progress(worktree.path)
        status = _status(worktree.path)
    except WorktreeHygieneError as exc:
        return "status-failed", str(exc)
    if operation:
        return "operation-in-progress", f"git state present: {operation}"
    if status.tracked:
        return "dirty-tracked", f"{len(status.tracked)} tracked change(s)"
    if status.untracked:
        return "untracked-files", f"{len(status.untracked)} untracked file(s)"
    try:
        merged = _is_ancestor(repo_root, worktree.head, origin_main_sha)
    except WorktreeHygieneError as exc:
        return "merge-check-failed", str(exc)
    if not merged:
        return "unmerged-commits", f"HEAD is not an ancestor of {ORIGIN_REF}"
    pull = pr_by_sha.get(worktree.head)
    if pull is not None:
        return "open-pr-head", f"HEAD is the head of open PR #{pull.number}"
    if worktree.branch is not None and worktree.branch in pr_by_branch:
        pull = pr_by_branch[worktree.branch]
        return "open-pr-branch", f"branch is the head branch of open PR #{pull.number}"
    # Cheap checks first; the full-tree walk only runs for still-eligible worktrees.
    nested = _registered_nested(worktree, registered)
    if nested is not None:
        return "contains-nested-worktree", f"registered worktree at {nested}"
    if status.ignored and not allow_ignored_data:
        return "ignored-data", _ignored_detail(status.ignored)
    try:
        scan = _scan_tree(worktree.path)
    except OSError as exc:
        return "scan-failed", str(exc)
    if scan.nested is not None:
        return "contains-nested-worktree", f"nested git entry at {scan.nested}"
    if scan.link is not None:
        return "contains-link", f"symlink/junction/reparse point at {scan.link}"
    newest = scan.newest
    if newest is not None and now - newest < min_idle_seconds:
        idle_hours = max(now - newest, 0.0) / 3600.0
        return "recently-modified", f"newest file change {idle_hours:.1f}h ago"
    if newest is None:
        return "removable", "clean, merged, empty"
    idle_hours = max(now - newest, 0.0) / 3600.0
    return "removable", f"clean, merged into {ORIGIN_REF}, idle {idle_hours:.1f}h"


def plan_worktrees(
    repo_root: Path,
    current: Path,
    *,
    open_pulls: Sequence[OpenPullRequest],
    origin_main_sha: str,
    min_idle_hours: float = DEFAULT_MIN_IDLE_HOURS,
    allow_ignored_data: bool = False,
    now: float | None = None,
) -> list[WorktreeDecision]:
    scope_dir = repo_root.joinpath(*SCOPE_PARTS)
    pr_by_sha = {pull.head_sha: pull for pull in open_pulls}
    pr_by_branch = {pull.branch: pull for pull in open_pulls}
    clock = time.time() if now is None else now
    registered = list_worktrees(repo_root)
    decisions: list[WorktreeDecision] = []
    for worktree in registered:
        reason, detail = _classify(
            worktree,
            repo_root=repo_root,
            scope_dir=scope_dir,
            current=current,
            registered=registered,
            origin_main_sha=origin_main_sha,
            pr_by_sha=pr_by_sha,
            pr_by_branch=pr_by_branch,
            min_idle_seconds=min_idle_hours * 3600.0,
            allow_ignored_data=allow_ignored_data,
            now=clock,
        )
        action = REMOVABLE if reason == "removable" else KEEP
        decisions.append(WorktreeDecision(worktree, action, reason, detail))
    return decisions


def _recheck(
    repo_root: Path, planned: Worktree, *, allow_ignored_data: bool
) -> tuple[str, str] | None:
    """Return a ``(reason, detail)`` keep verdict, or None if still removable."""
    registered = list_worktrees(repo_root)
    entry = next(
        (
            item
            for item in registered
            if _norm(item.path) == _norm(planned.path) and not item.is_main
        ),
        None,
    )
    if entry is None:
        return "recheck-vanished", "worktree is no longer registered"
    if entry.locked:
        return "recheck-locked", "worktree was locked after planning"
    if entry.head != planned.head:
        return "recheck-head-changed", f"HEAD moved to {entry.head}"
    operation = _operation_in_progress(planned.path)
    if operation:
        return "recheck-operation-in-progress", f"git state present: {operation}"
    status = _status(planned.path)
    changes = len(status.tracked) + len(status.untracked)
    if changes:
        return "recheck-dirty", f"{changes} change(s) appeared after planning"
    nested = _registered_nested(entry, registered)
    if nested is not None:
        return "recheck-nested-worktree", f"registered worktree at {nested}"
    if status.ignored and not allow_ignored_data:
        return "recheck-ignored-data", _ignored_detail(status.ignored)
    scan = _scan_tree(planned.path)
    if scan.nested is not None:
        return "recheck-nested-worktree", f"nested git entry at {scan.nested}"
    if scan.link is not None:
        return "recheck-contains-link", f"symlink/junction at {scan.link}"
    return None


def _probe_in_use(path: Path) -> str | None:
    """Rename the directory away and back; return why it is in use, if it is.

    On Windows an open handle or a process cwd inside the tree makes the rename
    fail, which prevents a half-completed ``git worktree remove``.  POSIX
    systems allow renaming busy directories, so there the probe detects nothing.
    """
    probe = path.with_name(f"{path.name}.hygiene-probe-{os.getpid()}")
    if os.path.lexists(probe):
        return f"probe path {probe} already exists"
    try:
        os.rename(path, probe)
    except OSError as exc:
        return str(exc)
    try:
        os.rename(probe, path)
    except OSError as exc:
        raise ProbeRestoreError(
            f"probe rename back failed; RESTORE MANUALLY: move {probe} back to "
            f"{path} ({exc})"
        ) from exc
    return None


PROBE_GLOB = "*.hygiene-probe-*"


def leftover_probes(
    repo_root: Path, decisions: Sequence[WorktreeDecision]
) -> list[Path]:
    """Probe directories left by an earlier failed restore, next to in-scope worktrees."""
    scope = repo_root.joinpath(*SCOPE_PARTS)
    parents = {scope}
    parents.update(
        item.worktree.path.parent
        for item in decisions
        if not item.worktree.is_main and _is_within(item.worktree.path, scope)
    )
    found = {
        match
        for parent in parents
        if parent.is_dir()
        for match in parent.glob(PROBE_GLOB)
    }
    return sorted(found)


def _delete_branch(repo_root: Path, branch: str) -> str:
    if branch in PROTECTED_BRANCHES:
        return f"skipped: protected branch {branch}"
    process = _git(repo_root, "branch", "-d", "--", branch)
    if process.returncode == 0:
        return f"deleted {branch}"
    message = (process.stderr or process.stdout).strip().replace("\n", " ")
    return f"refused by git branch -d: {message}"


def _remove_one(
    repo_root: Path,
    worktree: Worktree,
    *,
    delete_local_branches: bool,
    allow_ignored_data: bool,
) -> ApplyResult:
    verdict = _recheck(repo_root, worktree, allow_ignored_data=allow_ignored_data)
    if verdict is not None:
        return ApplyResult(worktree.path, "kept", verdict[0], verdict[1])
    in_use = _probe_in_use(worktree.path)
    if in_use is not None:
        return ApplyResult(worktree.path, "kept", "in-use", in_use)
    process = _git(repo_root, "worktree", "remove", str(worktree.path))
    if process.returncode != 0:
        return ApplyResult(
            worktree.path, "failed", "remove-failed", process.stderr.strip()
        )
    branch_result = None
    if delete_local_branches and worktree.branch is not None:
        branch_result = _delete_branch(repo_root, worktree.branch)
    return ApplyResult(worktree.path, "removed", "removed", "", branch_result)


def apply_removals(
    repo_root: Path,
    decisions: Sequence[WorktreeDecision],
    *,
    delete_local_branches: bool = False,
    allow_ignored_data: bool = False,
    results: list[ApplyResult] | None = None,
) -> list[ApplyResult]:
    """Remove REMOVABLE worktrees after a fresh per-worktree re-check.

    Outcomes are appended to ``results`` as they happen so a caller keeps the
    partial record if something unexpected escapes.
    """
    out = [] if results is None else results
    for decision in decisions:
        if decision.action != REMOVABLE:
            continue
        worktree = decision.worktree
        try:
            out.append(
                _remove_one(
                    repo_root,
                    worktree,
                    delete_local_branches=delete_local_branches,
                    allow_ignored_data=allow_ignored_data,
                )
            )
        except ProbeRestoreError:
            raise
        except (WorktreeHygieneError, OSError) as exc:
            out.append(ApplyResult(worktree.path, "failed", "error", str(exc)))
    return out


def prune_worktrees(repo_root: Path) -> str | None:
    """Run ``git worktree prune``; return an error message on failure."""
    process = _git(repo_root, "worktree", "prune")
    if process.returncode != 0:
        return process.stderr.strip() or "git worktree prune failed"
    return None


def _decision_dict(decision: WorktreeDecision) -> dict[str, object]:
    worktree = decision.worktree
    return {
        "path": str(worktree.path),
        "branch": worktree.branch,
        "head": worktree.head,
        "action": decision.action,
        "reason": decision.reason,
        "detail": decision.detail,
    }


def summarize(
    decisions: Sequence[WorktreeDecision],
    results: Sequence[ApplyResult],
    errors: Sequence[str],
) -> dict[str, object]:
    removed = sum(1 for item in results if item.outcome == "removed")
    failed = sum(1 for item in results if item.outcome == "failed")
    kept_after = sum(1 for item in results if item.outcome == "kept")
    plan_kept = sum(1 for item in decisions if item.action == KEEP)
    reasons = Counter(item.reason for item in decisions)
    return {
        "inspected": len(decisions),
        "removable": sum(1 for item in decisions if item.action == REMOVABLE),
        "removed": removed,
        "kept": plan_kept + kept_after,
        "failed": failed + len(errors),
        "reasons": dict(sorted(reasons.items())),
    }


def render_text(
    *,
    apply: bool,
    repo_root: Path,
    origin_main_sha: str,
    open_pr_count: int,
    min_idle_hours: float,
    allow_ignored_data: bool,
    decisions: Sequence[WorktreeDecision],
    results: Sequence[ApplyResult],
    errors: Sequence[str],
) -> str:
    lines = [
        "## Worktree hygiene",
        f"- mode: {'apply' if apply else 'dry-run'}",
        f"- repo root: {repo_root}",
        f"- {ORIGIN_REF}: {origin_main_sha}",
        f"- open PRs: {open_pr_count}",
        f"- min idle hours: {min_idle_hours:g}",
        f"- allow ignored data (DESTRUCTIVE): {'yes' if allow_ignored_data else 'no'}",
        "",
    ]
    for decision in decisions:
        worktree = decision.worktree
        lines.append(
            f"{decision.action:<9} [{decision.reason}] {worktree.path} "
            f"branch={worktree.branch or '-'} head={(worktree.head or '-')[:12]}"
            + (f" ({decision.detail})" if decision.detail else "")
        )
    if apply:
        lines.append("")
        for result in results:
            suffix = f" ({result.detail})" if result.detail else ""
            branch = f" branch: {result.branch_result}" if result.branch_result else ""
            lines.append(
                f"{result.outcome.upper():<9} [{result.reason}] {result.path}{suffix}{branch}"
            )
        lines.extend(f"FAILED    [error] {message}" for message in errors)
    summary = summarize(decisions, results, errors)
    reasons = summary["reasons"]
    assert isinstance(reasons, dict)
    lines.extend(
        [
            "",
            f"- inspected: {summary['inspected']}",
            f"- removable at observation: {summary['removable']}",
            f"- removed: {summary['removed']}",
            f"- kept: {summary['kept']}",
            f"- failed: {summary['failed']}",
            "- by reason: "
            + (", ".join(f"{key}={value}" for key, value in reasons.items()) or "none"),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan (default, read-only) or --apply removal of clean, merged, idle "
            f"linked worktrees under {'/'.join(SCOPE_PARTS)}."
        )
    )
    parser.add_argument(
        "--apply", action="store_true", help="remove REMOVABLE worktrees"
    )
    parser.add_argument(
        "--fetch", action="store_true", help="run `git fetch origin main` first"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--min-idle-hours",
        type=float,
        default=DEFAULT_MIN_IDLE_HOURS,
        help="keep worktrees with a file modified within this many hours",
    )
    parser.add_argument(
        "--delete-local-branches",
        action="store_true",
        help="with --apply, also `git branch -d` the removed worktree's branch",
    )
    parser.add_argument(
        "--allow-ignored-data",
        action="store_true",
        help=(
            "DESTRUCTIVE: also treat worktrees holding gitignored data beyond "
            "regenerable caches (artifacts, data, outputs, .env, ...) as "
            "removable; that data is deleted and cannot be recovered"
        ),
    )
    args = parser.parse_args(argv)
    if args.min_idle_hours < 0:
        parser.error("--min-idle-hours must be >= 0")
    if args.delete_local_branches and not args.apply:
        parser.error("--delete-local-branches requires --apply")
    return args


def main(
    argv: Sequence[str] | None = None,
    *,
    pr_heads_provider: PrHeadsProvider | None = None,
    cwd: Path | None = None,
    now: float | None = None,
) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(stdout_reconfigure):
        stdout_reconfigure(errors="backslashreplace")
    start = Path.cwd() if cwd is None else cwd
    try:
        repo_root = list_worktrees(start)[0].path
        current = current_toplevel(start)
        if args.fetch:
            fetch_origin_main(repo_root)
        origin_main_sha = resolve_origin_main(repo_root)
        provider = pr_heads_provider or (lambda: gh_open_pull_requests(repo_root))
        open_pulls = list(provider())
        decisions = plan_worktrees(
            repo_root,
            current,
            open_pulls=open_pulls,
            origin_main_sha=origin_main_sha,
            min_idle_hours=args.min_idle_hours,
            allow_ignored_data=args.allow_ignored_data,
            now=now,
        )
        if args.apply:
            leftovers = leftover_probes(repo_root, decisions)
            if leftovers:
                raise WorktreeHygieneError(
                    "refusing --apply: leftover rename-probe director(ies) from an "
                    "earlier failed restore; move each back to its worktree name "
                    "(or delete it if it is a stray) first: "
                    + ", ".join(str(path) for path in leftovers)
                )
    except (WorktreeHygieneError, OSError) as exc:
        print(f"worktree_hygiene: {exc}; nothing was removed", file=sys.stderr)
        return 2
    results: list[ApplyResult] = []
    errors: list[str] = []
    restore_failed = False
    if args.apply:
        try:
            apply_removals(
                repo_root,
                decisions,
                delete_local_branches=args.delete_local_branches,
                allow_ignored_data=args.allow_ignored_data,
                results=results,
            )
        except ProbeRestoreError as exc:
            errors.append(f"apply stopped, git worktree prune skipped: {exc}")
            restore_failed = True
        except (WorktreeHygieneError, OSError) as exc:
            errors.append(f"apply stopped: {exc}")
        if not restore_failed:
            try:
                prune_error = prune_worktrees(repo_root)
            except (WorktreeHygieneError, OSError) as exc:
                prune_error = str(exc)
            if prune_error:
                errors.append(f"git worktree prune: {prune_error}")
    summary = summarize(decisions, results, errors)
    if args.json:
        print(
            json.dumps(
                {
                    "mode": "apply" if args.apply else "dry-run",
                    "repo_root": str(repo_root),
                    "origin_main": origin_main_sha,
                    "open_prs": len(open_pulls),
                    "min_idle_hours": args.min_idle_hours,
                    "allow_ignored_data": args.allow_ignored_data,
                    "worktrees": [_decision_dict(item) for item in decisions],
                    "results": [
                        {
                            "path": str(item.path),
                            "outcome": item.outcome,
                            "reason": item.reason,
                            "detail": item.detail,
                            "branch_result": item.branch_result,
                        }
                        for item in results
                    ],
                    "errors": errors,
                    "summary": summary,
                },
                indent=2,
            )
        )
    else:
        print(
            render_text(
                apply=args.apply,
                repo_root=repo_root,
                origin_main_sha=origin_main_sha,
                open_pr_count=len(open_pulls),
                min_idle_hours=args.min_idle_hours,
                allow_ignored_data=args.allow_ignored_data,
                decisions=decisions,
                results=results,
                errors=errors,
            )
        )
    if errors:
        print(
            f"worktree_hygiene: {summary['removed']} worktree(s) removed before "
            f"the error(s) above: {'; '.join(errors)}",
            file=sys.stderr,
        )
    failed = summary["failed"]
    return 1 if isinstance(failed, int) and failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
