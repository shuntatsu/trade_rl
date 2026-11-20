"""One-time retirement of six exact frozen old fix refs already in current main.

No generic deletion by prefix or age. Every source has an exact recovery
snapshot in the frozen 85-tip archive, and is reachable from current main.
Owner Issue #1057 created-comment authority is required for application.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tools.archived_branch_retirement import (
    ARCHIVE_BRANCH,
    ARCHIVE_HEAD,
    REPOSITORY,
    _archive_manifest,
    _git_ancestor,
)
from tools.branch_hygiene import (
    BranchInfo,
    GitHubApi,
    open_pull_request_branch_names,
    reachable_commits,
    require_retirement_origin,
)

FROZEN_SOURCES = {
    "fix/issue810-crossasset-ancestor-link-20261111": "f56971b857db862c4adaa6682ee0dcf2ef36d625",
    "fix/issue1057-prune-record-shape-fence-20261103": "69d86011aade7fe3128fadf965496e58d5aecc92",
    "fix/issue1146-sd1-synthetic-boundary-regressions-20261111": "dffa2e9aed9b46287a676f84a3e7649184bcbce0",
    "fix/issue1153-binance-source-close-v2-20261111": "6a555dbb902312f1133f935356bdfe454f5b5a63",
    "fix/trackd-manifest-nofollow-20261111": "dd6e3ad9b4eb4d26c2466dbea1cc0d073cf7418e",
    "fix/trackd-raw-read-no-follow-20261111": "2dbea03274f113853f0dc4546e99515c32e435f3",
}
COMMAND = "/retire-six-archived-main-ancestors-20261120 " + ARCHIVE_HEAD
BATCH_SIZE = 3


def _git_main() -> str:
    return subprocess.run(
        ["git", "rev-parse", "refs/remotes/origin/main"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _require_owner_event() -> None:
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_ACTOR") != "shuntatsu"
        or os.environ.get("GITHUB_EVENT_NAME") != "issue_comment"
        or not os.environ.get("GITHUB_EVENT_PATH")
    ):
        raise RuntimeError("only exact owner-created Issue #1057 comment may run")
    payload: Any = json.loads(
        Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise RuntimeError("owner event payload malformed")
    issue, comment = payload.get("issue"), payload.get("comment")
    if (
        payload.get("action") != "created"
        or not isinstance(issue, dict)
        or type(issue.get("number")) is not int
        or issue.get("number") != 1057
        or not isinstance(comment, dict)
        or not isinstance(comment.get("user"), dict)
        or comment["user"].get("login") != "shuntatsu"
        or type(comment["user"].get("id")) is not int
        or comment["user"].get("id") != 148511409
        or comment.get("body") != COMMAND
    ):
        raise RuntimeError("missing exact owner-created Issue #1057 comment")


def _select(api: GitHubApi, branches: Sequence[BranchInfo]) -> tuple[BranchInfo, ...]:
    if api._repository != REPOSITORY or api.repository_default_branch() != "main":
        raise RuntimeError("only trusted trade_rl main may retire")
    refs = {b.name: b for b in branches}
    if len(refs) != len(branches):
        raise RuntimeError("duplicate ref roster")
    main, archive = refs.get("main"), refs.get(ARCHIVE_BRANCH)
    if main is None or archive is None or archive.sha != ARCHIVE_HEAD:
        raise RuntimeError("main or immutable archive missing/changed")
    if main.sha != _git_main():
        raise RuntimeError("main changed relative to fetched history")
    frozen = _archive_manifest()
    if (
        len(FROZEN_SOURCES) != 6
        or len(set(FROZEN_SOURCES.values())) != 6
        or any(
            not name.startswith("fix/")
            or name.startswith("fix/m1-")
            or frozen.get(name) != sha
            or re.fullmatch(r"[0-9a-f]{40}", sha) is None
            for name, sha in FROZEN_SOURCES.items()
        )
    ):
        raise RuntimeError("exact six source/manifest proof failed")
    reachable = reachable_commits((main.sha,))
    if main.sha not in reachable:
        raise RuntimeError("main reachable-commit evidence unavailable")
    open_names = open_pull_request_branch_names(api.open_pull_request_refs())
    running = api.active_workflow_branch_names()
    result: list[BranchInfo] = []
    for name, sha in sorted(FROZEN_SOURCES.items()):
        b = refs.get(name)
        if b is None:
            continue
        if (
            b.protected
            or b.sha != sha
            or name in open_names
            or name in running
            or b.sha not in reachable
        ):
            continue
        if not _git_ancestor(b.sha, ARCHIVE_HEAD):
            raise RuntimeError("immutable archive lost history of " + name)
        result.append(b)
    return tuple(result)


def retire(
    api: GitHubApi, *, apply: bool
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    before = api.branches()
    queue = _select(api, before)
    names = tuple(b.name for b in queue)
    if not apply or not queue:
        return len(before), names, ()
    _require_owner_event()
    require_retirement_origin()
    main_sha = next(b.sha for b in before if b.name == "main")
    if os.environ.get("GITHUB_SHA") != main_sha:
        raise RuntimeError("workflow main SHA stale")
    expected = {b.name: b for b in before}
    deleted: list[str] = []
    for start in range(0, len(queue), BATCH_SIZE):
        batch = queue[start : start + BATCH_SIZE]
        current = api.branches()
        refs = {b.name: b for b in current}
        if refs != expected:
            raise RuntimeError("remote branch roster drifted")
        if refs.get("main") is None or refs["main"].sha != main_sha:
            raise RuntimeError("main changed before batch")
        if refs.get(ARCHIVE_BRANCH) is None or refs[ARCHIVE_BRANCH].sha != ARCHIVE_HEAD:
            raise RuntimeError("frozen archive changed before batch")
        if not set(batch).issubset(_select(api, current)):
            raise RuntimeError("eligibility changed before deletion")
        api.delete_branches_with_leases(batch)
        for b in batch:
            expected.pop(b.name)
        if {b.name: b for b in api.branches()} != expected:
            raise RuntimeError("post-delete exact roster mismatch")
        deleted.extend(b.name for b in batch)
    return len(before), names, tuple(deleted)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("trusted workflow token missing")
    api = GitHubApi(
        repository=REPOSITORY, token=token, api_url="https://api.github.com"
    )
    initial, selected, deleted = retire(api, apply=args.apply)
    summary = "\n".join(
        (
            "## Exact six archived main-ancestor retirement",
            f"- mode: {'apply' if args.apply else 'preview'}",
            f"- observed: {initial}",
            f"- candidate count: {len(selected)}",
            f"- deleted and verified: {len(deleted)}",
            f"- recovery archive: {ARCHIVE_BRANCH}@{ARCHIVE_HEAD}",
            *(f"- deleted: {name}" for name in deleted),
        )
    )
    print(summary)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(summary + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
