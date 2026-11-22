"""Retire 13 exact closed SD2 reviewer references, preserving frozen archive history.

This is software-only.  The feature is inert until this module exists on main
and the exact owner-created Issue #1057 comment triggers its dedicated job.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.archived_obsolete_20261121 import (
    ARCHIVE_BRANCH,
    ARCHIVE_HEAD,
    OWNER_ID,
    OWNER_LOGIN,
    REPOSITORY,
    _archive_manifest,
    _closed_pr_proof,
    _git_main,
    _is_ancestor,
)
from tools.branch_hygiene import (
    BranchInfo,
    GitHubApi,
    open_pull_request_branch_names,
    require_retirement_origin,
)

ISSUE_NUMBER = 1057
COMMAND = "/retire-thirteen-archived-sd2-review-refs-20261121 " + ARCHIVE_HEAD

# Exact PR source names and final HEADs. Every PR is already CLOSED/unmerged.
PR_SOURCE: Mapping[str, tuple[str, int]] = {
    "feat/issue1146-sd2-runner-20261113": (
        "38375e402d265b5e9868edc521cede87cd29c8c1",
        1236,
    ),
    "review/issue1146-sd2-2014-nofollow-history-20261121": (
        "bbc27e022865d5b9b76987454128385f7c9540b9",
        1308,
    ),
    "review/issue1146-sd2-2020-history-20261121": (
        "f5f21c55bf62240f4edaa9191de83672ef82c13b",
        1309,
    ),
    "review/issue1146-sd2-3037-packets-20261120": (
        "752dc00cb9805f91266aeb29752267fc69a6acab",
        1292,
    ),
    "review/issue1146-sd2-3037-source-20261120": (
        "25add7058aea068e2465d5d0b474285aca5b74cb",
        1290,
    ),
    "review/issue1146-sd2-3037-tests-20261120": (
        "9830a891d08ee2646a6429d2c335530afcdf6062",
        1291,
    ),
    "review/issue1146-sd2-owner-docs-after1305-20261121": (
        "99f28b6e15293a779da4836aa99bc5b1e8091b17",
        1310,
    ),
    "review/issue1146-sd2-postm2-consistent-packet-20261120": (
        "f94ec17eab751ef0a5a7fec2e780e7f7a237f20a",
        1297,
    ),
}
# Review-comparison refs unused by any active PR. Some are exact main ancestors;
# two doc2 comparison refs are not in main but are fully archived separately.
REVIEW_BASES: Mapping[str, str] = {
    "review/issue1146-sd2-3037-base-20261120": (
        "670a3f32eeb7709168f382d4f2ee7c3da818d209"
    ),
    "review/issue1146-sd2-after-main1293-base-20261121": (
        "9ca71fa23badfce0f7b3da7ce13d7b789a8b87f4"
    ),
    "review/issue1146-sd2-doc2-base-20261121": (
        "ea5c0f3e2138e4db07b68905065566266c24499f"
    ),
    "review/issue1146-sd2-doc2-mirror-20261121": (
        "60c84d2240e4b332ae51356057e48afe40545061"
    ),
    "review/issue1146-sd2-owner-docs-after1305-base-20261121": (
        "d13cc66464e6efd5708cf056e65bd4002a8696d1"
    ),
}
MAIN_ANCESTOR_BASES = frozenset(
    {
        "review/issue1146-sd2-3037-base-20261120",
        "review/issue1146-sd2-after-main1293-base-20261121",
        "review/issue1146-sd2-owner-docs-after1305-base-20261121",
    }
)
EXACT_COUNT = 13


def _owner_command_event() -> None:
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_EVENT_NAME") != "issue_comment"
        or os.environ.get("GITHUB_ACTOR") != OWNER_LOGIN
        or not os.environ.get("GITHUB_EVENT_PATH")
    ):
        raise RuntimeError("exact created owner Issue #1057 comment is required")
    payload: Any = json.loads(
        Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise RuntimeError("GitHub event must be an object")
    issue, comment = payload.get("issue"), payload.get("comment")
    if (
        payload.get("action") != "created"
        or not isinstance(issue, dict)
        or type(issue.get("number")) is not int
        or issue["number"] != ISSUE_NUMBER
        or not isinstance(comment, dict)
        or not isinstance(comment.get("user"), dict)
        or comment["user"].get("id") != OWNER_ID
        or comment["user"].get("login") != OWNER_LOGIN
        or comment.get("body") != COMMAND
    ):
        raise RuntimeError("not the owner-created exact retired-review command")


def _select(
    api: GitHubApi,
    branches: Sequence[BranchInfo],
    *,
    archive: Mapping[str, str],
    open_pr: frozenset[str],
    active: frozenset[str],
) -> tuple[BranchInfo, ...]:
    if api._repository != REPOSITORY or api.repository_default_branch() != "main":
        raise RuntimeError("only the canonical default repository is eligible")
    refs = {item.name: item for item in branches}
    if len(refs) != len(branches):
        raise RuntimeError("duplicated live GitHub branch roster")
    main, saved = refs.get("main"), refs.get(ARCHIVE_BRANCH)
    if (
        main is None
        or saved is None
        or saved.sha != ARCHIVE_HEAD
        or main.sha != _git_main()
    ):
        raise RuntimeError("live main/archive/ref history changed")
    targets = {name: pair[0] for name, pair in PR_SOURCE.items()}
    targets.update(REVIEW_BASES)
    if len(PR_SOURCE) != 8 or len(REVIEW_BASES) != 5 or len(targets) != EXACT_COUNT:
        raise RuntimeError("retirement allowlist differs from the frozen 13")
    if set(MAIN_ANCESTOR_BASES) - set(REVIEW_BASES):
        raise RuntimeError("unexpected main-ancestry check list")
    out: list[BranchInfo] = []
    for name, expected in sorted(targets.items()):
        item = refs.get(name)
        if item is None:
            continue  # idempotent after first retirement
        if item.sha != expected or archive.get(name) != expected:
            raise RuntimeError("live target differs from immutable archive SHA")
        if item.protected or name in open_pr or name in active:
            continue
        if not _is_ancestor(expected, ARCHIVE_HEAD):
            raise RuntimeError("archive lost the source commit")
        if name in MAIN_ANCESTOR_BASES and not _is_ancestor(expected, main.sha):
            raise RuntimeError("main lost a frozen comparison base")
        if name in PR_SOURCE and not _closed_pr_proof(api, item, PR_SOURCE[name][1]):
            continue
        out.append(item)
    return tuple(out)


def _require_complete_one_shot(
    branches: Sequence[BranchInfo], selected: Sequence[BranchInfo]
) -> bool:
    """Require all 13 eligible or all 13 already absent before any mutation.

    The owner command is issued once. A partial live/eligible batch must never
    silently consume it and make recovery depend on a second owner command.
    """
    expected = frozenset(PR_SOURCE) | frozenset(REVIEW_BASES)
    present = expected & {item.name for item in branches}
    if not present:
        return False  # Verified complete previous retirement: idempotent readback.
    if present != expected or {item.name for item in selected} != expected:
        raise RuntimeError("all 13 archived SD2 refs must be live and eligible")
    return True


def retire(
    api: GitHubApi, *, apply: bool
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    frozen = _archive_manifest()
    before = api.branches()
    selected = _select(
        api,
        before,
        archive=frozen,
        open_pr=open_pull_request_branch_names(api.open_pull_request_refs()),
        active=api.active_workflow_branch_names(),
    )
    planned = tuple(item.name for item in selected)
    if not apply:
        return len(before), planned, ()
    if not _require_complete_one_shot(before, selected):
        return len(before), planned, ()
    _owner_command_event()
    require_retirement_origin()
    first = {item.name: item for item in before}
    main_sha = first["main"].sha
    if os.environ.get("GITHUB_SHA") != main_sha:
        raise RuntimeError("event checkout does not match current main")
    current = api.branches()
    if {item.name: item for item in current} != first:
        raise RuntimeError("GitHub branch roster changed before delete")
    # Re-read all live eligibility, do not trust the previous planning snapshot.
    if _archive_manifest() != frozen:
        raise RuntimeError("archive manifest changed before delete")
    repeated = _select(
        api,
        current,
        archive=frozen,
        open_pr=open_pull_request_branch_names(api.open_pull_request_refs()),
        active=api.active_workflow_branch_names(),
    )
    if not _require_complete_one_shot(current, repeated) or repeated != selected:
        raise RuntimeError("all 13 review refs must remain eligible before delete")
    api.delete_branches_with_leases(selected)
    for item in selected:
        first.pop(item.name)
    after = {item.name: item for item in api.branches()}
    if first != after:
        raise RuntimeError("post-retirement GitHub branch roster differs")
    return len(before), planned, planned


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GitHub Actions token required")
    api = GitHubApi(
        repository=REPOSITORY,
        token=token,
        api_url="https://api.github.com",
    )
    observed, planned, deleted = retire(api, apply=args.apply)
    lines = [
        "## Archived SD2 reviewer source retirement (13 pinned refs)",
        f"- mode: {'apply' if args.apply else 'preview'}",
        f"- observed: {observed}",
        f"- eligible: {len(planned)}",
        f"- deleted and verified: {len(deleted)}",
        f"- immutable archive: {ARCHIVE_BRANCH}@{ARCHIVE_HEAD}",
        *(f"- retired: {name}" for name in deleted),
    ]
    report = "\n".join(lines)
    print(report)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a", encoding="utf-8") as file:
            file.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
