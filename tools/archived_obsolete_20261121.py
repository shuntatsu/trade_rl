"""One-shot owner-only retirement of eight frozen obsolete branch names.

All original histories are retained by a pinned 56-tip archival commit.
Neither merging this code nor reviewing a PR authorizes the deletion: only
the exact owner-created Issue #1057 command can invoke --apply on main.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.branch_hygiene import (
    BranchInfo,
    GitHubApi,
    open_pull_request_branch_names,
    require_retirement_origin,
)

REPOSITORY = "shuntatsu/trade_rl"
ARCHIVE_BRANCH = "archive/branch-snapshot-20261121"
ARCHIVE_HEAD = "b95286e38df9e46a10f9fb43fbfac14dd3c51499"
OWNER_LOGIN = "shuntatsu"
OWNER_ID = 148511409
ISSUE_NUMBER = 1057
COMMAND = "/retire-eight-superseded-refs-20261121 " + ARCHIVE_HEAD
_SNAPSHOT_MAIN = "d13cc66464e6efd5708cf056e65bd4002a8696d1"
_SHA = re.compile(r"^[0-9a-f]{40}$")

# Six exact closed/unmerged PR source branches. No research, M1, audit, infra,
# provenance, main, or archive refs are listed here.
CLOSED_PR_SOURCES: Mapping[str, tuple[str, int]] = {
    "fix/core7-daily-fallback-digest-20261120": (
        "1850dd4d1bd16f74db37e175d979062a8da6a00d",
        1284,
    ),
    "fix/issue1131-json-staging-foreign-cleanup-20261119": (
        "924c81f51a8d457b9023561579f5a4f0e3c88664",
        1274,
    ),
    "fix/m2-fd-exception-close-20261120": (
        "1c9887a577506c46d4013901d60106b1afb00bff",
        1296,
    ),
    "review/issue810-trend-oos-20261120-protocol": (
        "6269f701b70ac83fb91aed1e182b7c70f44408d1",
        1300,
    ),
    "review/issue810-trend-oos-20261120-runner": (
        "cd6691940e9d6d7bc6e882f43354ebba0f3f6e56",
        1301,
    ),
    "review/issue810-trend-oos-20261120-scenarios": (
        "e855fb8ebfca0ed2b79c69a43f10b97ccecef1d9",
        1302,
    ),
}
# Two exact old review bases are already historical ancestors of main.
HISTORICAL_MAIN_ANCESTORS: Mapping[str, str] = {
    "review/issue810-trend-oos-20261120-base": (
        "f1f8f80475242fd4cc3d10e4801074df7ae231f0"
    ),
    "review/issue810-trend-oos-20261120-test-part1": (
        "f1f8f80475242fd4cc3d10e4801074df7ae231f0"
    ),
}
EXPECTED_COUNT = 8


def _git(*argv: str) -> str:
    proc = subprocess.run(["git", *argv], check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def _is_ancestor(older: str, newer: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _git_main() -> str:
    return _git("rev-parse", "refs/remotes/origin/main")


def _archive_manifest() -> dict[str, str]:
    if _git("rev-parse", f"refs/remotes/origin/{ARCHIVE_BRANCH}") != ARCHIVE_HEAD:
        raise RuntimeError("pinned recovery archive branch changed")
    data: Any = json.loads(_git("show", f"{ARCHIVE_HEAD}:branch-tip-manifest.json"))
    if (
        not isinstance(data, dict)
        or data.get("schema") != "trade_rl_branch_tip_snapshot_v2"
        or data.get("repository") != REPOSITORY
        or data.get("archive_branch") != ARCHIVE_BRANCH
        or data.get("main_head") != _SNAPSHOT_MAIN
        or not isinstance(data.get("branches"), list)
        or len(data["branches"]) != 56
    ):
        raise RuntimeError("archive manifest identity/count mismatch")
    result: dict[str, str] = {}
    for entry in data["branches"]:
        if not isinstance(entry, dict):
            raise RuntimeError("archive branch entry not an object")
        name, sha = entry.get("name"), entry.get("sha")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(sha, str)
            or not _SHA.fullmatch(sha)
            or type(entry.get("protected")) is not bool
            or name in result
        ):
            raise RuntimeError("archive branch entry is malformed/duplicated")
        result[name] = sha
    if result.get("main") != _SNAPSHOT_MAIN:
        raise RuntimeError("archive captured main identity mismatch")
    return result


def _owner_created_event() -> None:
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_ACTOR") != OWNER_LOGIN
        or os.environ.get("GITHUB_EVENT_NAME") != "issue_comment"
        or not os.environ.get("GITHUB_EVENT_PATH")
    ):
        raise RuntimeError("only owner-created Issue #1057 event may retire refs")
    event: Any = json.loads(
        Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8")
    )
    if not isinstance(event, dict):
        raise RuntimeError("owner event must be object")
    issue, comment = event.get("issue"), event.get("comment")
    if (
        event.get("action") != "created"
        or not isinstance(issue, dict)
        or type(issue.get("number")) is not int
        or issue["number"] != ISSUE_NUMBER
        or not isinstance(comment, dict)
        or not isinstance(comment.get("user"), dict)
        or comment["user"].get("login") != OWNER_LOGIN
        or comment["user"].get("id") != OWNER_ID
        or comment.get("body") != COMMAND
    ):
        raise RuntimeError("not an exact owner-created retirement command")


def _closed_pr_proof(api: GitHubApi, branch: BranchInfo, number: int) -> bool:
    value: Any = api._request_json("GET", f"pulls/{number}")
    if not isinstance(value, dict):
        raise RuntimeError("closed PR proof is not an object")
    head, base = value.get("head"), value.get("base")
    if not isinstance(head, dict) or not isinstance(base, dict):
        raise RuntimeError("closed PR head/base is malformed")
    hr, br = head.get("repo"), base.get("repo")
    if hr is None or br is None:
        return False
    if not isinstance(hr, dict) or not isinstance(br, dict):
        raise RuntimeError("closed PR repository identities are malformed")
    return (
        type(value.get("number")) is int
        and value["number"] == number
        and value.get("state") == "closed"
        and value.get("merged_at") is None
        and isinstance(value.get("closed_at"), str)
        and bool(value.get("closed_at"))
        and head.get("ref") == branch.name
        and head.get("sha") == branch.sha
        and hr.get("full_name") == REPOSITORY
        and br.get("full_name") == REPOSITORY
    )


def _select(
    api: GitHubApi,
    branches: Sequence[BranchInfo],
    *,
    frozen: Mapping[str, str],
    open_names: frozenset[str],
    active_names: frozenset[str],
) -> tuple[BranchInfo, ...]:
    if api._repository != REPOSITORY or api.repository_default_branch() != "main":
        raise RuntimeError("only shuntatsu/trade_rl/main is eligible")
    refs = {b.name: b for b in branches}
    if len(refs) != len(branches):
        raise RuntimeError("duplicate live branch roster")
    main, archive = refs.get("main"), refs.get(ARCHIVE_BRANCH)
    if (
        main is None
        or archive is None
        or archive.sha != ARCHIVE_HEAD
        or main.sha != _git_main()
    ):
        raise RuntimeError("main/archive identity or checkout drift")
    targets = dict(HISTORICAL_MAIN_ANCESTORS)
    targets.update({name: pair[0] for name, pair in CLOSED_PR_SOURCES.items()})
    if (
        len(targets) != EXPECTED_COUNT
        or len(CLOSED_PR_SOURCES) != 6
        or len(HISTORICAL_MAIN_ANCESTORS) != 2
    ):
        raise RuntimeError("frozen allowlist unexpectedly changed")
    result: list[BranchInfo] = []
    for name, sha in sorted(targets.items()):
        branch = refs.get(name)
        if branch is None:
            continue  # Repeat invocation remains idempotent after prior cleanup.
        if branch.sha != sha or frozen.get(name) != sha or not _SHA.fullmatch(sha):
            raise RuntimeError("live ref SHA does not match frozen archive")
        if branch.protected or name in open_names or name in active_names:
            continue
        if not _is_ancestor(sha, ARCHIVE_HEAD):
            raise RuntimeError("archive lacks target source commit")
        if name in HISTORICAL_MAIN_ANCESTORS:
            if not _is_ancestor(sha, main.sha):
                raise RuntimeError("old review base no longer reachable from main")
        elif name == "fix/core7-daily-fallback-digest-20261120":
            if not _is_ancestor(sha, main.sha):
                raise RuntimeError("Core7 fallback code no longer in main history")
        if name in CLOSED_PR_SOURCES and not _closed_pr_proof(
            api, branch, CLOSED_PR_SOURCES[name][1]
        ):
            continue
        result.append(branch)
    return tuple(result)


def retire(
    api: GitHubApi, *, apply: bool
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    manifest = _archive_manifest()
    before = api.branches()
    selected = _select(
        api,
        before,
        frozen=manifest,
        open_names=open_pull_request_branch_names(api.open_pull_request_refs()),
        active_names=api.active_workflow_branch_names(),
    )
    names = tuple(x.name for x in selected)
    if not apply or not selected:
        return len(before), names, ()
    _owner_created_event()
    require_retirement_origin()
    before_refs = {b.name: b for b in before}
    main_sha = before_refs["main"].sha
    if os.environ.get("GITHUB_SHA") != main_sha:
        raise RuntimeError("owner command is not bound to fresh current main")
    now = api.branches()
    if {b.name: b for b in now} != before_refs:
        raise RuntimeError("GitHub branch roster changed before apply")
    selected_now = _select(
        api,
        now,
        frozen=manifest,
        open_names=open_pull_request_branch_names(api.open_pull_request_refs()),
        active_names=api.active_workflow_branch_names(),
    )
    if not set(selected).issubset(selected_now):
        raise RuntimeError("open PR/workflow/protected/ref eligibility drift")
    # One atomic, exact-SHA-leased batch, never a wildcard deletion.
    api.delete_branches_with_leases(selected)
    for branch in selected:
        before_refs.pop(branch.name)
    after = {b.name: b for b in api.branches()}
    if after != before_refs:
        raise RuntimeError("post-delete exact GitHub roster mismatch")
    return len(before), names, names


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GitHub Actions token is required")
    api = GitHubApi(
        repository=REPOSITORY,
        token=token,
        api_url="https://api.github.com",
    )
    observed, planned, deleted = retire(api, apply=args.apply)
    report = "\n".join(
        [
            "## Frozen eight obsolete branch name retirement",
            f"- mode: {'apply' if args.apply else 'preview'}",
            f"- observed branch names: {observed}",
            f"- exact eligible candidates: {len(planned)}",
            f"- deleted and verified: {len(deleted)}",
            f"- archive: {ARCHIVE_BRANCH}@{ARCHIVE_HEAD}",
            "- main, active PR/workflow, protected, research, M1, audit, infra and provenance refs are untouched",
            *(f"- retired: {name}" for name in deleted),
        ]
    )
    print(report)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as file:
            file.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
