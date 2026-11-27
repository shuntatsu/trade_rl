"""Safely remove stale linked git worktrees under ``<repo>/.claude/worktrees``.

Default mode is a read-only plan.  ``--apply`` removes only worktrees that are
classified ``REMOVABLE`` (clean, merged into ``origin/main``, not an open PR
head, not locked, idle).  The tool never forces a removal, never deletes remote
branches and never touches the stash.  Local branches are deleted only with
``--delete-local-branches`` and only via ``git branch -d``.  Any uncertainty
(``gh`` failure, unreadable ``origin/main``, git errors) stops the run without
removing anything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

REMOVABLE = "REMOVABLE"
KEEP = "KEEP"

_SHA = re.compile(r"[0-9a-f]{40}")


class WorktreeHygieneError(RuntimeError):
    """Fail-closed error: nothing may be removed after this is raised."""


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


def _status_lines(worktree: Path) -> list[str]:
    output = _git_ok(worktree, "status", "--porcelain", "--untracked-files=all")
    return [line for line in output.splitlines() if line.strip()]


def _is_ancestor(repo_root: Path, sha: str, base_sha: str) -> bool:
    process = _git(repo_root, "merge-base", "--is-ancestor", sha, base_sha)
    if process.returncode == 0:
        return True
    if process.returncode == 1:
        return False
    raise WorktreeHygieneError(f"merge-base failed: {process.stderr.strip()}")


def latest_activity(worktree: Path) -> float | None:
    """Newest mtime among tracked/untracked-not-ignored files (``.git`` excluded)."""
    output = _git_ok(
        worktree, "ls-files", "-z", "--cached", "--others", "--exclude-standard"
    )
    newest: float | None = None
    for name in output.split("\0"):
        if not name or name == ".git" or name.startswith(".git/"):
            continue
        try:
            mtime = os.lstat(worktree / name).st_mtime
        except FileNotFoundError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    return newest


def _classify(
    worktree: Worktree,
    *,
    repo_root: Path,
    scope_dir: Path,
    current: Path,
    origin_main_sha: str,
    pr_by_sha: dict[str, OpenPullRequest],
    min_idle_seconds: float,
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
        lines = _status_lines(worktree.path)
    except WorktreeHygieneError as exc:
        return "status-failed", str(exc)
    tracked = [line for line in lines if not line.startswith("??")]
    untracked = [line for line in lines if line.startswith("??")]
    if tracked:
        return "dirty-tracked", f"{len(tracked)} tracked change(s)"
    if untracked:
        return "untracked-files", f"{len(untracked)} untracked file(s)"
    try:
        merged = _is_ancestor(repo_root, worktree.head, origin_main_sha)
    except WorktreeHygieneError as exc:
        return "merge-check-failed", str(exc)
    if not merged:
        return "unmerged-commits", f"HEAD is not an ancestor of {ORIGIN_REF}"
    pull = pr_by_sha.get(worktree.head)
    if pull is not None:
        return "open-pr-head", f"HEAD is the head of open PR #{pull.number}"
    try:
        newest = latest_activity(worktree.path)
    except (WorktreeHygieneError, OSError) as exc:
        return "activity-scan-failed", str(exc)
    if newest is not None and now - newest < min_idle_seconds:
        idle_hours = max(now - newest, 0.0) / 3600.0
        return "recently-modified", f"newest file change {idle_hours:.1f}h ago"
    if newest is None:
        return "removable", "clean, merged, no tracked files"
    idle_hours = max(now - newest, 0.0) / 3600.0
    return "removable", f"clean, merged into {ORIGIN_REF}, idle {idle_hours:.1f}h"


def plan_worktrees(
    repo_root: Path,
    current: Path,
    *,
    open_pulls: Sequence[OpenPullRequest],
    origin_main_sha: str,
    min_idle_hours: float = DEFAULT_MIN_IDLE_HOURS,
    now: float | None = None,
) -> list[WorktreeDecision]:
    scope_dir = repo_root.joinpath(*SCOPE_PARTS)
    pr_by_sha = {pull.head_sha: pull for pull in open_pulls}
    clock = time.time() if now is None else now
    decisions: list[WorktreeDecision] = []
    for worktree in list_worktrees(repo_root):
        reason, detail = _classify(
            worktree,
            repo_root=repo_root,
            scope_dir=scope_dir,
            current=current,
            origin_main_sha=origin_main_sha,
            pr_by_sha=pr_by_sha,
            min_idle_seconds=min_idle_hours * 3600.0,
            now=clock,
        )
        action = REMOVABLE if reason == "removable" else KEEP
        decisions.append(WorktreeDecision(worktree, action, reason, detail))
    return decisions


def _recheck(repo_root: Path, planned: Worktree) -> tuple[str, str] | None:
    """Return a ``(reason, detail)`` keep verdict, or None if still removable."""
    entry = next(
        (
            item
            for item in list_worktrees(repo_root)
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
    lines = _status_lines(planned.path)
    if lines:
        return "recheck-dirty", f"{len(lines)} change(s) appeared after planning"
    return None


def _delete_branch(repo_root: Path, branch: str) -> str:
    if branch in PROTECTED_BRANCHES:
        return f"skipped: protected branch {branch}"
    process = _git(repo_root, "branch", "-d", "--", branch)
    if process.returncode == 0:
        return f"deleted {branch}"
    message = (process.stderr or process.stdout).strip().replace("\n", " ")
    return f"refused by git branch -d: {message}"


def apply_removals(
    repo_root: Path,
    decisions: Sequence[WorktreeDecision],
    *,
    delete_local_branches: bool = False,
) -> list[ApplyResult]:
    """Remove REMOVABLE worktrees after a fresh per-worktree re-check."""
    results: list[ApplyResult] = []
    for decision in decisions:
        if decision.action != REMOVABLE:
            continue
        worktree = decision.worktree
        try:
            verdict = _recheck(repo_root, worktree)
        except WorktreeHygieneError as exc:
            results.append(
                ApplyResult(worktree.path, "failed", "recheck-failed", str(exc))
            )
            continue
        if verdict is not None:
            results.append(ApplyResult(worktree.path, "kept", verdict[0], verdict[1]))
            continue
        process = _git(repo_root, "worktree", "remove", str(worktree.path))
        if process.returncode != 0:
            results.append(
                ApplyResult(
                    worktree.path, "failed", "remove-failed", process.stderr.strip()
                )
            )
            continue
        branch_result = None
        if delete_local_branches and worktree.branch is not None:
            branch_result = _delete_branch(repo_root, worktree.branch)
        results.append(
            ApplyResult(worktree.path, "removed", "removed", "", branch_result)
        )
    return results


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
    prune_error: str | None,
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
        "failed": failed + (1 if prune_error else 0),
        "reasons": dict(sorted(reasons.items())),
    }


def render_text(
    *,
    apply: bool,
    repo_root: Path,
    origin_main_sha: str,
    open_pr_count: int,
    min_idle_hours: float,
    decisions: Sequence[WorktreeDecision],
    results: Sequence[ApplyResult],
    prune_error: str | None,
) -> str:
    lines = [
        "## Worktree hygiene",
        f"- mode: {'apply' if apply else 'dry-run'}",
        f"- repo root: {repo_root}",
        f"- {ORIGIN_REF}: {origin_main_sha}",
        f"- open PR heads: {open_pr_count}",
        f"- min idle hours: {min_idle_hours:g}",
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
        if prune_error:
            lines.append(f"FAILED    [prune-failed] git worktree prune: {prune_error}")
    summary = summarize(decisions, results, prune_error)
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
            now=now,
        )
        results: list[ApplyResult] = []
        prune_error: str | None = None
        if args.apply:
            results = apply_removals(
                repo_root,
                decisions,
                delete_local_branches=args.delete_local_branches,
            )
            prune_error = prune_worktrees(repo_root)
    except WorktreeHygieneError as exc:
        print(f"worktree_hygiene: {exc}; nothing was removed", file=sys.stderr)
        return 2
    summary = summarize(decisions, results, prune_error)
    if args.json:
        print(
            json.dumps(
                {
                    "mode": "apply" if args.apply else "dry-run",
                    "repo_root": str(repo_root),
                    "origin_main": origin_main_sha,
                    "open_pr_heads": len(open_pulls),
                    "min_idle_hours": args.min_idle_hours,
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
                    "prune_error": prune_error,
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
                decisions=decisions,
                results=results,
                prune_error=prune_error,
            )
        )
    failed = summary["failed"]
    return 1 if isinstance(failed, int) and failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
