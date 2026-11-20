"""One-time, explicitly enumerated closure of frozen, obsolete PR source branches.

This exceptional archival retirement does not merge abandoned experimental
code into main and does not weaken the default canonical merged-PR retirement
contract. No branch is selected by naming prefix or age alone.

Each retired source must match the fixed 85-tip archive, a currently *closed*
GitHub PR at its exact final head, the explicit allowlist, and a fresh absence
of open PR and active workflow references. Every mutation is atomic and
SHA-leased, and the complete branch roster is re-read after every batch.
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
    M1_ADDITIONAL_SOURCE_REFS,
    M1_SOURCE_REF_PREFIX,
    BranchInfo,
    GitHubApi,
    OpenPullRequestRefs,
    open_pull_request_branch_names,
    require_retirement_origin,
)

COMMAND = "/retire-archived-closed-prs-20261120 " + ARCHIVE_HEAD
OWNER_LOGIN = "shuntatsu"
OWNER_ID = 148511409
ISSUE_NUMBER = 1057
BATCH_SIZE = 7
MAX_DELETES = 28

# Frozen owner-directed closure set. Every name/SHA is also pinned in the
# immutable archive manifest; every PR number must be genuinely closed at its
# exact matching final head. These sources are discarded historical work, not
# claimed to have been merged or validated as economic results.
CLOSED_PR_ALLOWLIST: Mapping[str, int] = {
    "codex/issue1025-native-paired-action-v1-20261028": 1027,
    "codex/issue1025-td2-synthetic-label-maturity-v1-20261028": 1028,
    "codex/issue1025-td3-linear-action-value-20261029": 1031,
    "codex/issue1025-td4-synthetic-native-attempt-20261029": 1032,
    "codex/issue1025-td4b-costaware-synthetic-20261029": 1033,
    "codex/issue810-artificial-q-preflight": 946,
    "codex/issue810-artificial-reference-account-luna": 960,
    "codex/issue810-artificial-witness-admission": 952,
    "codex/issue810-cash-control-admission-luna": 961,
    "codex/issue810-consolidation-20261023-gpt6": 945,
    "codex/issue810-fa36-declaration-profile": 953,
    "codex/issue810-forecast-baseline-diagnostics-luna": 958,
    "codex/issue810-longterm-plan": 954,
    "codex/issue810-native-advance-speed-luna": 955,
    "codex/issue810-native-filled-ledger-test": 944,
    "codex/issue810-native-pair-plan-memo": 949,
    "codex/issue810-objective-owner-guard-20261025": 1003,
    "codex/issue810-optin-retain-debt-20261025": 1011,
    "codex/issue810-pair-plan-source-profile": 951,
    "codex/issue810-r2-zip-byte-auditor-20261025": 1015,
    "codex/issue810-rl-clock-contract-luna": 957,
    "codex/issue810-windows-cim-ci-readiness": 950,
    "dependabot/docker_compose/deploy/market-store/postgres-16.16-bookworm": 1271,
    "docs/fix-issue1253-retired-prune-workflow-20261114": 1260,
    "review/issue1146-budget-source-20261114": 1254,
    "review/issue1146-final-code-and-tests-20261114": 1256,
    "review/issue1146-nofollow-source-20261114": 1252,
    "review/issue1146-whole-cli-final-blobs-20261114": 1258,
}
_SHA = re.compile(r"^[0-9a-f]{40}$")


def _owner_comment_event() -> None:
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_ACTOR") != OWNER_LOGIN
        or os.environ.get("GITHUB_EVENT_NAME") != "issue_comment"
    ):
        raise RuntimeError("only the owner-created Issue #1057 event may retire refs")
    filename = os.environ.get("GITHUB_EVENT_PATH")
    if not filename:
        raise RuntimeError("owner event payload unavailable")
    event: Any = json.loads(Path(filename).read_text(encoding="utf-8"))
    if not isinstance(event, dict):
        raise RuntimeError("invalid issue-comment event")
    issue, comment = event.get("issue"), event.get("comment")
    if (
        not isinstance(issue, dict)
        or type(issue.get("number")) is not int
        or issue.get("number") != ISSUE_NUMBER
        or not isinstance(comment, dict)
        or not isinstance(comment.get("user"), dict)
        or comment["user"].get("id") != OWNER_ID
        or comment["user"].get("login") != OWNER_LOGIN
        or comment.get("body") != COMMAND
    ):
        raise RuntimeError("missing exact owner Issue #1057 command")


def _closed_pr_proof(api: GitHubApi, branch: BranchInfo) -> bool:
    """Never treat a merely closed PR at a different head as closure evidence."""
    number = CLOSED_PR_ALLOWLIST.get(branch.name)
    if type(number) is not int or number <= 0:
        return False
    p: Any = api._request_json("GET", f"pulls/{number}")
    if not isinstance(p, dict):
        raise RuntimeError("GitHub PR response is not an object")
    head, base = p.get("head"), p.get("base")
    if not isinstance(head, dict) or not isinstance(base, dict):
        raise RuntimeError("GitHub PR source identity is malformed")
    head_repo, base_repo = head.get("repo"), base.get("repo")
    if head_repo is None or base_repo is None:
        # Deleted or transferred PR sources may not establish an exact ref.
        return False
    if not isinstance(head_repo, dict) or not isinstance(base_repo, dict):
        raise RuntimeError("GitHub PR repository identity is malformed")
    return (
        p.get("number") == number
        and p.get("state") == "closed"
        and isinstance(p.get("closed_at"), str)
        and bool(p["closed_at"])
        and head.get("ref") == branch.name
        and head.get("sha") == branch.sha
        and head_repo.get("full_name") == REPOSITORY
        and base_repo.get("full_name") == REPOSITORY
    )


def _select(
    api: GitHubApi,
    branches: Sequence[BranchInfo],
    *,
    frozen: Mapping[str, str],
    open_refs: Sequence[OpenPullRequestRefs],
    active: frozenset[str],
) -> tuple[BranchInfo, ...]:
    """Read-only selection; closed PRs and archive are authority for *only* allowlisted names."""
    refs = {b.name: b for b in branches}
    if len(refs) != len(branches):
        raise RuntimeError("duplicate GitHub branch identities")
    main, archive = refs.get("main"), refs.get(ARCHIVE_BRANCH)
    if main is None or archive is None or archive.sha != ARCHIVE_HEAD:
        raise RuntimeError("main or pinned archive ref missing/changed")
    if main.sha != _git_main():
        raise RuntimeError("GitHub main differs from complete checkout history")
    open_names = open_pull_request_branch_names(open_refs)
    if (
        len(CLOSED_PR_ALLOWLIST) != MAX_DELETES
        or len(set(CLOSED_PR_ALLOWLIST.values())) != MAX_DELETES
    ):
        raise RuntimeError("one-shot closed PR allowlist has changed")
    out = []
    for name, number in sorted(CLOSED_PR_ALLOWLIST.items()):
        branch = refs.get(name)
        if branch is None:
            continue  # Idempotent: previously retired ref is no longer selected.
        if (
            branch.protected
            or name in open_names
            or name in active
            or name.startswith(
                (
                    M1_SOURCE_REF_PREFIX,
                    "research/",
                    "audit/",
                    "provenance/",
                    "infra/",
                    "archive/",
                )
            )
            or name in M1_ADDITIONAL_SOURCE_REFS
            or frozen.get(name) != branch.sha
            or _SHA.fullmatch(branch.sha) is None
        ):
            continue
        if not _git_ancestor(branch.sha, ARCHIVE_HEAD):
            raise RuntimeError(f"frozen archive does not reach {name}")
        if _closed_pr_proof(api, branch):
            out.append(branch)
    return tuple(out)


def _git_main() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "refs/remotes/origin/main"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def retire(
    api: GitHubApi, *, apply: bool
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    if api._repository != REPOSITORY or api.repository_default_branch() != "main":
        raise RuntimeError("ref retirement is restricted to shuntatsu/trade_rl/main")
    frozen = _archive_manifest()
    before = api.branches()
    selected = _select(
        api,
        before,
        frozen=frozen,
        open_refs=api.open_pull_request_refs(),
        active=api.active_workflow_branch_names(),
    )
    selected_names = tuple(b.name for b in selected)
    if not apply or not selected:
        return len(before), selected_names, ()
    _owner_comment_event()
    require_retirement_origin()
    main_sha = next(x.sha for x in before if x.name == "main")
    if os.environ.get("GITHUB_SHA") != main_sha:
        raise RuntimeError("run is not bound to exact current main")
    expected = {b.name: b for b in before}
    deleted: list[str] = []
    for start in range(0, len(selected), BATCH_SIZE):
        batch = selected[start : start + BATCH_SIZE]
        current = api.branches()
        refs = {b.name: b for b in current}
        if refs != expected or refs.get("main") is None or refs["main"].sha != main_sha:
            raise RuntimeError("branch snapshot or main changed before batch")
        if refs.get(ARCHIVE_BRANCH) is None or refs[ARCHIVE_BRANCH].sha != ARCHIVE_HEAD:
            raise RuntimeError("frozen archive changed before batch")
        newly = _select(
            api,
            current,
            frozen=frozen,
            open_refs=api.open_pull_request_refs(),
            active=api.active_workflow_branch_names(),
        )
        if not set(batch).issubset(newly):
            raise RuntimeError(
                "closure/open PR/active run/ref eligibility changed before batch"
            )
        api.delete_branches_with_leases(tuple(batch))
        after = {b.name: b for b in api.branches()}
        for b in batch:
            expected.pop(b.name)
        if after != expected:
            raise RuntimeError("GitHub post-delete exact branch roster differs")
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
    before, selected, deleted = retire(api, apply=args.apply)
    summary = "\n".join(
        (
            "## One-shot frozen closed-PR source retirement",
            f"- mode: {'apply' if args.apply else 'preview'}",
            f"- observed: {before}",
            f"- exact closed-PR candidates: {len(selected)}",
            f"- deleted and verified: {len(deleted)}",
            f"- immutable archive: {ARCHIVE_BRANCH}@{ARCHIVE_HEAD}",
            "- No open PR, active workflow, research, M1, audit, infra, archive or provenance branch is eligible.",
            *(f"- deleted: {name}" for name in deleted),
        )
    )
    print(summary)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(summary + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
