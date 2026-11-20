"""One-shot retirement of archived, inactive branch refs.

No economic execution. Refusal is the default: every removed tip must be pinned in
an immutable archive snapshot and reachable from the archive commit. The owner
issues a one-time issue comment; an ordinary local invocation cannot delete.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.branch_hygiene import (
    M1_ADDITIONAL_SOURCE_REFS,
    M1_SOURCE_REF_PREFIX,
    BranchDecision,
    BranchInfo,
    GitHubApi,
    MergedPullRequestSource,
    OpenPullRequestRefs,
    apply_merged_retirement,
    open_pull_request_branch_names,
    plan_merged_retirement,
    reachable_commits,
    require_retirement_origin,
)

REPOSITORY = "shuntatsu/trade_rl"
ARCHIVE_BRANCH = "archive/branch-snapshot-20261120"
ARCHIVE_HEAD = "2be720a5ef837462e4132ebecf982d8929112b6c"
OWNER_ID = 148511409
TRIGGER = "/retire-archived-branches-20261120 " + ARCHIVE_HEAD
PRESERVE = (
    "archive/",
    "provenance/",
    "research/",
    "audit/",
    "infra/",
    "seal/",
    "freeze/",
    "run/",
    "evidence/",
    "study/",
    "release/",
    "artifact/",
)
# A stored archive is a recoverability mechanism, not proof that active/unmerged
# work is disposable. Retire only old *main-ancestor* work refs under this
# one-shot exception; separately review every branch with unique history.
MERGED_DISPOSABLE = ("codex/", "review/", "docs/", "dependabot/", "feat/", "fix/")
MAX_DELETES = 60
BATCH_SIZE = 8
_SHA = re.compile(r"^[0-9a-f]{40}$")


def _git(*args: str) -> str:
    p = subprocess.run(["git", *args], text=True, capture_output=True, check=True)
    return p.stdout.strip()


def _archive_manifest() -> dict[str, str]:
    if _git("rev-parse", "refs/remotes/origin/" + ARCHIVE_BRANCH) != ARCHIVE_HEAD:
        raise RuntimeError("archive ref moved or cannot be verified")
    payload = json.loads(
        _git(
            "show",
            "refs/remotes/origin/" + ARCHIVE_BRANCH + ":branch-tip-manifest.json",
        )
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "trade_rl_branch_tip_snapshot_v1"
        or payload.get("repository") != REPOSITORY
        or payload.get("archive_branch") != ARCHIVE_BRANCH
    ):
        raise RuntimeError("archive manifest identity differs")
    rows = payload.get("branches")
    if not isinstance(rows, list) or len(rows) != 85:
        raise RuntimeError("archive manifest does not cover the 85 recorded tips")
    out: dict[str, str] = {}
    for entry in rows:
        if not isinstance(entry, dict):
            raise RuntimeError("malformed archive tip")
        name, sha = entry.get("name"), entry.get("sha")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(sha, str)
            or _SHA.fullmatch(sha) is None
            or name in out
        ):
            raise RuntimeError("invalid or duplicate archived ref")
        out[name] = sha
    recorded_main = payload.get("main_head")
    if (
        not isinstance(recorded_main, str)
        or _SHA.fullmatch(recorded_main) is None
        or out.get("main") != recorded_main
    ):
        raise RuntimeError("archive main identity is inconsistent")
    return out


def _git_ancestor(tip: str, ancestor_of: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", tip, ancestor_of],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def plan(
    branches: Sequence[BranchInfo],
    *,
    archived: Mapping[str, str],
    main_reachable: frozenset[str],
    open_refs: Sequence[OpenPullRequestRefs],
    active_branches: frozenset[str],
    merged_pr_sources: Sequence[MergedPullRequestSource] = (),
) -> tuple[BranchInfo, ...]:
    """Select only frozen archived tips that pass canonical merged-PR source proof."""
    if not main_reachable or len({b.name for b in branches}) != len(branches):
        raise RuntimeError("incomplete or duplicate branch roster")
    refs = {b.name: b for b in branches}
    if (
        "main" not in refs
        or refs["main"].sha not in main_reachable
        or refs.get(ARCHIVE_BRANCH) is None
        or refs[ARCHIVE_BRANCH].sha != ARCHIVE_HEAD
    ):
        raise RuntimeError("main or the frozen archive is missing")
    canonical = plan_merged_retirement(
        branches,
        default_branch="main",
        open_pull_request_refs=open_refs,
        active_workflow_branches=active_branches,
        reachable_from_main=main_reachable,
        merged_pr_sources=merged_pr_sources,
    )
    eligible = {
        decision.branch for decision in canonical if decision.action == "delete"
    }
    open_names = open_pull_request_branch_names(open_refs)
    selected: list[BranchInfo] = []
    for b in sorted(branches, key=lambda item: item.name):
        if (
            b.name == "main"
            or b.protected
            or b.name in open_names
            or b.name in active_branches
            or b.name.startswith(PRESERVE)
            or b.name.startswith(M1_SOURCE_REF_PREFIX)
            or b.name in M1_ADDITIONAL_SOURCE_REFS
            or b not in eligible
            or archived.get(b.name) != b.sha
            or _SHA.fullmatch(b.sha) is None
        ):
            continue
        if b.sha not in main_reachable:
            continue
        if b.name.startswith(MERGED_DISPOSABLE):
            selected.append(b)
    if len(selected) > MAX_DELETES:
        raise RuntimeError("retirement count exceeded the one-shot safety cap")
    return tuple(selected)


def _require_owner_event() -> None:
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_ACTOR") != "shuntatsu"
        or os.environ.get("GITHUB_EVENT_NAME") != "issue_comment"
    ):
        raise RuntimeError(
            "one-shot retirement requires the owner issue-comment workflow"
        )
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise RuntimeError("GitHub event payload is missing")
    event: Any = json.loads(Path(event_path).read_text(encoding="utf-8"))
    if not isinstance(event, dict):
        raise RuntimeError("malformed GitHub event")
    issue, comment = event.get("issue"), event.get("comment")
    if not isinstance(issue, dict) or not isinstance(comment, dict):
        raise RuntimeError("event is not an issue comment")
    actor = comment.get("user")
    if (
        issue.get("number") != 1057
        or not isinstance(actor, dict)
        or actor.get("login") != "shuntatsu"
        or actor.get("id") != OWNER_ID
        or comment.get("body") != TRIGGER
    ):
        raise RuntimeError("missing exact owner-authored retirement command")


def _snapshot(
    api: GitHubApi, archived: Mapping[str, str]
) -> tuple[tuple[BranchInfo, ...], tuple[BranchInfo, ...], frozenset[str]]:
    branches = api.branches()
    refs = {b.name: b for b in branches}
    main = refs.get("main")
    archive = refs.get(ARCHIVE_BRANCH)
    if main is None or archive is None or archive.sha != ARCHIVE_HEAD:
        raise RuntimeError("main or archive was replaced")
    if main.sha != _git("rev-parse", "refs/remotes/origin/main"):
        raise RuntimeError("fresh GitHub main differs from checked-out history")
    main_reachable = reachable_commits((main.sha,))
    selected = plan(
        branches,
        archived=archived,
        main_reachable=main_reachable,
        open_refs=api.open_pull_request_refs(),
        active_branches=api.active_workflow_branch_names(),
        merged_pr_sources=api.merged_pr_source_refs(include_nested=True),
    )
    for b in selected:
        if not _git_ancestor(b.sha, ARCHIVE_HEAD):
            raise RuntimeError(
                "selected history is not reachable from archive: " + b.name
            )
    return branches, selected, main_reachable


def retire(
    api: GitHubApi, *, apply: bool
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    archived = _archive_manifest()
    before, selected, main_reachable = _snapshot(api, archived)
    selected_names = tuple(b.name for b in selected)
    if not apply or not selected:
        return len(before), selected_names, ()

    _require_owner_event()
    require_retirement_origin()
    start_main = next(b.sha for b in before if b.name == "main")
    if os.environ.get("GITHUB_SHA") != start_main:
        raise RuntimeError("workflow SHA is not the current main")
    # Delegate actual GitHub mutation and fresh per-batch PR/workflow/ref
    # validation to the *existing canonical* merged-PR executor. All candidates
    # have separately passed archive-SHA and archive-ancestry checks; only
    # final exact source heads of genuinely merged PRs are admitted. In this
    # pinned one-shot, an intermediate-base PR is acceptable only when its
    # exact source SHA is already reachable from current main.
    selected_set = set(selected)
    decisions = tuple(
        BranchDecision(
            branch=b,
            action="delete" if b in selected_set else "keep",
            reason="archived-exact-merged-pr"
            if b in selected_set
            else "not-in-one-shot-scope",
        )
        for b in before
    )
    deleted = apply_merged_retirement(
        api,
        decisions,
        default_branch="main",
        expected_main_sha=start_main,
        reachable_from_main=main_reachable,
        max_deletes=MAX_DELETES,
        batch_size=BATCH_SIZE,
        include_nested_merged_prs=True,
    )
    return len(before), selected_names, deleted


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GitHub Actions token is required")
    api = GitHubApi(
        repository=REPOSITORY, token=token, api_url="https://api.github.com"
    )
    initial, selected, deleted = retire(api, apply=args.apply)
    summary = "\n".join(
        [
            "## Archived one-shot branch retirement",
            f"- mode: {'apply' if args.apply else 'preview'}",
            f"- observed: {initial}",
            f"- selected: {len(selected)}",
            f"- deleted and verified: {len(deleted)}",
            f"- archive: {ARCHIVE_BRANCH} at {ARCHIVE_HEAD}",
            "- protected, open PR, active workflow, research, audit, M1 and provenance refs remain",
            *(f"- deleted: {name}" for name in deleted),
        ]
    )
    print(summary)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(summary + "\n")
    return 0


if __name__ == "__main__":
    sys_code = main()
    raise SystemExit(sys_code)
