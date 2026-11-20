"""Owner-only archival retirement of three superseded SD2 branch names.

Do not merge abandoned source into main. A frozen archive preserves exact original
commits, and an independently tested open successor PR must still include each
original commit. No generic name/age-based branch deletion is permitted.
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

EXACT = {
    "fix/issue1146-sd2-manifest-first-array-gate-20261113": "6131b5d0c54f86a5cc9d50f641c163ad0b8d6fa8",
    "review/issue1146-bounded-codeonly-base-20261114": "93d94145c7809c7fc0f1819a500f910b28eb2a0f",
    "review/issue1146-pre-budget-source-20261114": "3a3c959350fe31d4a05d9540d460c3187be3a772",
}
SUCCEEDING_PR = 1247
SUCCEEDING_REF = "fix/issue1146-sd2-descriptor-nofollow-20261114"
COMMAND = "/retire-three-superseded-sd2-refs-20261120 " + ARCHIVE_HEAD
OWNER_ID = 148511409
_SHA = re.compile(r"[a-f0-9]{40}")


def _require_owner_event() -> None:
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_ACTOR") != "shuntatsu"
        or os.environ.get("GITHUB_EVENT_NAME") != "issue_comment"
        or not os.environ.get("GITHUB_EVENT_PATH")
    ):
        raise RuntimeError("owner-only created comment required")
    obj: Any = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if not isinstance(obj, dict):
        raise RuntimeError("invalid event")
    issue, comment = obj.get("issue"), obj.get("comment")
    if (
        obj.get("action") != "created"
        or not isinstance(issue, dict)
        or type(issue.get("number")) is not int
        or issue.get("number") != 1057
        or not isinstance(comment, dict)
        or not isinstance(comment.get("user"), dict)
        or comment["user"].get("login") != "shuntatsu"
        or comment["user"].get("id") != OWNER_ID
        or comment.get("body") != COMMAND
    ):
        raise RuntimeError("incorrect owner issue comment")


def _git_main() -> str:
    return subprocess.run(
        ["git", "rev-parse", "refs/remotes/origin/main"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _proven_successor(api: GitHubApi, refs: Mapping[str, BranchInfo]) -> str:
    item: Any = api._request_json("GET", f"pulls/{SUCCEEDING_PR}")
    if not isinstance(item, dict):
        raise RuntimeError("successor PR response is invalid")
    head, base = item.get("head"), item.get("base")
    if not isinstance(head, dict) or not isinstance(base, dict):
        raise RuntimeError("successor PR identities are missing")
    head_repo, base_repo = head.get("repo"), base.get("repo")
    if not isinstance(head_repo, dict) or not isinstance(base_repo, dict):
        raise RuntimeError("successor repo identities are missing")
    if (
        item.get("number") != SUCCEEDING_PR
        or head.get("ref") != SUCCEEDING_REF
        or head_repo.get("full_name") != REPOSITORY
        or base_repo.get("full_name") != REPOSITORY
        or base.get("ref") != "main"
    ):
        raise RuntimeError("successor PR identity changed")
    if item.get("state") == "open":
        successor = refs.get(SUCCEEDING_REF)
        if successor is None or successor.sha != head.get("sha"):
            raise RuntimeError("open successor exact live branch SHA changed")
        if successor.protected:
            # Unneeded legacy ref retirement never changes protected successor.
            pass
        return successor.sha
    if item.get("state") == "closed" and item.get("merged_at"):
        # A normally merged successor must include the old source history in main.
        return refs["main"].sha
    raise RuntimeError("successor PR was abandoned or not merged")


def _select(api: GitHubApi, branches: Sequence[BranchInfo]) -> tuple[BranchInfo, ...]:
    if api._repository != REPOSITORY or api.repository_default_branch() != "main":
        raise RuntimeError("unexpected repo or default branch")
    by_name = {b.name: b for b in branches}
    if len(by_name) != len(branches):
        raise RuntimeError("duplicate branch roster")
    main, archive = by_name.get("main"), by_name.get(ARCHIVE_BRANCH)
    if main is None or archive is None or archive.sha != ARCHIVE_HEAD:
        raise RuntimeError("main or immutable archive ref changed")
    if main.sha != _git_main():
        raise RuntimeError("main changed against complete checkout")
    frozen = _archive_manifest()
    if len(EXACT) != 3 or len(set(EXACT.values())) != 3:
        raise RuntimeError("allowed source set changed")
    if any(
        frozen.get(name) != sha or _SHA.fullmatch(sha) is None
        for name, sha in EXACT.items()
    ):
        raise RuntimeError("immutable frozen SHA/name manifest differs")
    successor = _proven_successor(api, by_name)
    open_names = open_pull_request_branch_names(api.open_pull_request_refs())
    active = api.active_workflow_branch_names()
    current_main = reachable_commits((main.sha,))
    selected: list[BranchInfo] = []
    for name, sha in sorted(EXACT.items()):
        b = by_name.get(name)
        if b is None:
            continue
        if (
            b.protected
            or b.sha != sha
            or name in open_names
            or name in active
            or name in (ARCHIVE_BRANCH, "main", SUCCEEDING_REF)
        ):
            continue
        if not _git_ancestor(sha, ARCHIVE_HEAD):
            raise RuntimeError("archive no longer reaches source tip")
        if not (_git_ancestor(sha, successor) or sha in current_main):
            raise RuntimeError("successor/main lost the superseded source ancestry")
        selected.append(b)
    return tuple(selected)


def retire(
    api: GitHubApi, *, apply: bool
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    before = api.branches()
    queue = _select(api, before)
    selected_names = tuple(b.name for b in queue)
    if not apply or not queue:
        return len(before), selected_names, ()
    _require_owner_event()
    require_retirement_origin()
    main_sha = next(b.sha for b in before if b.name == "main")
    if os.environ.get("GITHUB_SHA") != main_sha:
        raise RuntimeError("workflow source main SHA changed")
    expected = {b.name: b for b in before}
    deleted: list[str] = []
    # Three exact refs: one atomic SHA-leased push with a complete post-delete roster.
    if {b.name: b for b in api.branches()} != expected:
        raise RuntimeError("branch roster changed before delete")
    if set(queue) - set(_select(api, tuple(expected.values()))):
        raise RuntimeError("PR, workflow, archive or successor changed before delete")
    if next(b.sha for b in api.branches() if b.name == "main") != main_sha:
        raise RuntimeError("main changed before deletion")
    api.delete_branches_with_leases(queue)
    for b in queue:
        expected.pop(b.name)
    if {b.name: b for b in api.branches()} != expected:
        raise RuntimeError("post-delete exact branch roster mismatch")
    deleted.extend(b.name for b in queue)
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
    initial, planned, deleted = retire(api, apply=args.apply)
    output = "\n".join(
        [
            "## Frozen superseded SD2 source ref retirement",
            f"- mode: {'apply' if args.apply else 'preview'}",
            f"- inspected: {initial}",
            f"- planned: {len(planned)}",
            f"- deleted and verified: {len(deleted)}",
            f"- immutable archive: {ARCHIVE_BRANCH}@{ARCHIVE_HEAD}",
            *(f"- retired: {name}" for name in deleted),
        ]
    )
    print(output)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
