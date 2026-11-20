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
    BranchInfo,
    GitHubApi,
    OpenPullRequestRefs,
    open_pull_request_branch_names,
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
    "fix/m1-",
)
MERGED_DISPOSABLE = ("codex/", "review/", "docs/", "dependabot/", "feat/", "fix/")
UNMERGED_DISPOSABLE = ("codex/", "review/", "docs/", "dependabot/")
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
    if out.get("main") != payload.get("main_head"):
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
) -> tuple[BranchInfo, ...]:
    """Pick only frozen archived tips, excluding all currently used/protected refs."""
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
    open_names = open_pull_request_branch_names(open_refs)
    selected: list[BranchInfo] = []
    for b in sorted(branches, key=lambda item: item.name):
        if (
            b.name == "main"
            or b.protected
            or b.name in open_names
            or b.name in active_branches
            or b.name.startswith(PRESERVE)
            or archived.get(b.name) != b.sha
            or _SHA.fullmatch(b.sha) is None
        ):
            continue
        merged = b.sha in main_reachable
        prefixes = MERGED_DISPOSABLE if merged else UNMERGED_DISPOSABLE
        if b.name.startswith(prefixes):
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
    before, selected, _ = _snapshot(api, archived)
    selected_names = tuple(b.name for b in selected)
    if not apply or not selected:
        return len(before), selected_names, ()

    _require_owner_event()
    require_retirement_origin()
    start_main = next(b.sha for b in before if b.name == "main")
    if os.environ.get("GITHUB_SHA") != start_main:
        raise RuntimeError("workflow SHA is not the current main")
    deleted: list[str] = []
    for start in range(0, len(selected), BATCH_SIZE):
        batch = selected[start : start + BATCH_SIZE]
        current = api.branches()
        refs = {b.name: b for b in current}
        if refs.get("main") is None or refs["main"].sha != start_main:
            raise RuntimeError("main moved before a retirement batch")
        if refs.get(ARCHIVE_BRANCH) is None or refs[ARCHIVE_BRANCH].sha != ARCHIVE_HEAD:
            raise RuntimeError("archive moved before a retirement batch")
        active = api.active_workflow_branch_names()
        open_refs = api.open_pull_request_refs()
        current_plan = set(
            plan(
                current,
                archived=archived,
                main_reachable=reachable_commits((start_main,)),
                open_refs=open_refs,
                active_branches=active,
            )
        )
        if any(refs.get(b.name) != b or b not in current_plan for b in batch):
            raise RuntimeError("branch state or ownership changed before deletion")
        if any(not _git_ancestor(b.sha, ARCHIVE_HEAD) for b in batch):
            raise RuntimeError("archive reachability changed before deletion")
        api.delete_branches_with_leases(batch)
        after = {b.name: b for b in api.branches()}
        if (
            any(b.name in after for b in batch)
            or after.get("main") != refs["main"]
            or after.get(ARCHIVE_BRANCH) != refs[ARCHIVE_BRANCH]
        ):
            raise RuntimeError("post-deletion GitHub readback is inconsistent")
        deleted.extend(b.name for b in batch)
    return len(before), selected_names, tuple(deleted)


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
