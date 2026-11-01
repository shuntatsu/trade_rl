"""Only fully integrated, independently retained PR source branches may be pruned."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tools.branch_hygiene import BranchInfo, OpenPullRequestRefs
from tools.verified_merged_branch_prune import (
    prune_verified_merged_branch,
    require_origin_checkout,
)

REPO = "shuntatsu/trade_rl"
NAME = "feat/merged"
TIP = "a" * 40
MAIN = "c" * 40
RETENTION = "d" * 40
RECORD = "e" * 40
MERGE = "f" * 40
NOW = datetime(2026, 11, 2, tzinfo=timezone.utc)


class FakeGitHub:
    def __init__(self) -> None:
        self.records = [
            BranchInfo("main", MAIN, False),
            BranchInfo("provenance/branch-retention", RETENTION, False),
            BranchInfo(NAME, TIP, False),
        ]
        self.open_refs: tuple[OpenPullRequestRefs, ...] = ()
        self.active: frozenset[str] = frozenset()
        self.pr: dict[str, object] = {
            "number": 1076,
            "state": "closed",
            "merged": True,
            "merged_at": "2026-11-01T13:00:00Z",
            "merge_commit_sha": MERGE,
            "head": {
                "ref": NAME,
                "sha": TIP,
                "repo": {"full_name": REPO},
            },
            "base": {"ref": "main", "repo": {"full_name": REPO}},
        }
        self.retention_record: dict[str, object] = {
            "sha": RECORD,
            "message": (
                "Observe non-anchor branch tips before cleanup grace period\n\n"
                + NAME + "\t" + TIP
            ),
            "committer": {"date": "2026-11-01T12:00:00Z"},
            "parents": [
                {"sha": "b" * 40},
                {"sha": TIP},
            ],
        }
        self.branch_reads = 0
        self.second_tip: str | None = None
        self.second_main: str | None = None
        self.deleted: list[BranchInfo] = []
        self.requests: list[str] = []
        self.compare_invalid: str | None = None
        self.after_delete_still_present = False

    def repository_default_branch(self) -> str:
        return "main"

    def branches(self) -> tuple[BranchInfo, ...]:
        self.branch_reads += 1
        items = list(self.records)
        if self.branch_reads > 1 and self.second_tip is not None:
            items[-1] = BranchInfo(NAME, self.second_tip, False)
        if self.branch_reads > 1 and self.second_main is not None:
            items[0] = BranchInfo("main", self.second_main, False)
        return tuple(items)

    def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
        return self.open_refs

    def active_workflow_branch_names(self) -> frozenset[str]:
        return self.active

    def _request_json(
        self, method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> object:
        assert method == "GET"
        assert query is None
        self.requests.append(suffix)
        if suffix == "pulls/1076":
            return self.pr
        if suffix == "git/commits/" + RECORD:
            return self.retention_record
        if suffix.startswith("compare/"):
            base, _, _ = suffix.removeprefix("compare/").partition("...")
            return {
                "status": "ahead",
                "ahead_by": 1,
                "behind_by": 1 if suffix == self.compare_invalid else 0,
                "merge_base_commit": {"sha": base},
            }
        raise AssertionError(f"unexpected GET: {suffix}")

    def delete_branches_with_leases(self, branches: tuple[BranchInfo, ...]) -> None:
        self.deleted.extend(branches)
        if not self.after_delete_still_present:
            self.records = [
                item for item in self.records if item.name not in {b.name for b in branches}
            ]


def _prune(api: FakeGitHub, *, apply: bool = False):
    return prune_verified_merged_branch(
        api,
        repository=REPO,
        branch=NAME,
        expected_sha=TIP,
        merged_pr_number=1076,
        retention_record_sha=RECORD,
        now=NOW,
        apply=apply,
    )


def test_preview_is_read_only_and_evidence_bound() -> None:
    api = FakeGitHub()
    result = _prune(api)
    assert result.deleted is False
    assert result.branch == NAME
    assert result.expected_sha == TIP
    assert result.main_sha == MAIN
    assert result.retention_sha == RETENTION
    assert api.deleted == []
    assert api.branch_reads == 1


def test_single_merged_branch_is_deleted_with_exact_sha_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeGitHub()
    monkeypatch.setattr(
        "tools.verified_merged_branch_prune.require_origin_checkout", lambda _: None
    )
    result = _prune(api, apply=True)
    assert result.deleted is True
    assert api.deleted == [BranchInfo(NAME, TIP, False)]
    assert api.branch_reads >= 3
    assert {b.name for b in api.branches()} == {
        "main", "provenance/branch-retention"
    }


def test_candidate_tip_change_before_deletion_fails_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeGitHub()
    api.second_tip = "8" * 40
    monkeypatch.setattr(
        "tools.verified_merged_branch_prune.require_origin_checkout", lambda _: None
    )
    with pytest.raises(ValueError, match="branch tip changed"):
        _prune(api, apply=True)
    assert api.deleted == []


def test_current_main_move_blocks_pruning(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeGitHub()
    api.second_main = "8" * 40
    monkeypatch.setattr(
        "tools.verified_merged_branch_prune.require_origin_checkout", lambda _: None
    )
    with pytest.raises(ValueError, match="main"):
        _prune(api, apply=True)
    assert api.deleted == []


def test_open_pr_head_or_base_is_protected() -> None:
    api = FakeGitHub()
    api.open_refs = (OpenPullRequestRefs(NAME, TIP, "main", MAIN),)
    with pytest.raises(ValueError, match="open PR"):
        _prune(api)
    assert api.deleted == []

    api.open_refs = (OpenPullRequestRefs("other", TIP, NAME, TIP),)
    with pytest.raises(ValueError, match="open PR"):
        _prune(api)


def test_active_workflow_blocks_pruning() -> None:
    api = FakeGitHub()
    api.active = frozenset({NAME})
    with pytest.raises(ValueError, match="workflow"):
        _prune(api)


@pytest.mark.parametrize(
    "change",
    [
        {"state": "open"},
        {"merged": False},
        {"head": {"ref": NAME, "sha": "b" * 40, "repo": {"full_name": REPO}}},
        {"head": {"ref": NAME, "sha": TIP, "repo": {"full_name": "other/repo"}}},
        {"base": {"ref": "other", "repo": {"full_name": REPO}}},
    ],
)
def test_only_matching_closed_merged_local_pr_is_eligible(
    change: dict[str, object],
) -> None:
    api = FakeGitHub()
    api.pr.update(change)
    with pytest.raises(ValueError, match="merged PR"):
        _prune(api)
    assert api.deleted == []


def test_unmerged_commit_does_not_pass_main_ancestry() -> None:
    api = FakeGitHub()
    api.compare_invalid = "compare/" + TIP + "..." + MAIN
    with pytest.raises(ValueError, match="main ancestor"):
        _prune(api)


def test_retention_must_include_exact_observation_and_tip() -> None:
    api = FakeGitHub()
    api.retention_record["message"] = "Observe non-anchor branch tips before cleanup grace period"
    with pytest.raises(ValueError, match="retention record"):
        _prune(api)
    api = FakeGitHub()
    api.retention_record["parents"] = [{"sha": "b" * 40}]
    with pytest.raises(ValueError, match="retention record"):
        _prune(api)
    api = FakeGitHub()
    api.compare_invalid = "compare/" + TIP + "..." + RETENTION
    with pytest.raises(ValueError, match="retention"):
        _prune(api)


def test_retention_grace_is_mandatory() -> None:
    api = FakeGitHub()
    api.retention_record["committer"] = {"date": "2026-11-01T23:59:00Z"}
    with pytest.raises(ValueError, match="grace"):
        _prune(api)


@pytest.mark.parametrize(
    "name",
    [
        "main",
        "provenance/branch-retention",
        "research/evidence",
        "seal/study",
        "freeze/config",
        "run/execution",
        "feat/../main",
    ],
)
def test_provenance_and_unsafe_branch_names_are_never_candidates(
    name: str,
) -> None:
    api = FakeGitHub()
    with pytest.raises(ValueError, match="branch"):
        prune_verified_merged_branch(
            api,
            repository=REPO,
            branch=name,
            expected_sha=TIP,
            merged_pr_number=1076,
            retention_record_sha=RECORD,
            now=NOW,
        )
    assert api.requests == []


def test_post_delete_readback_failure_never_claims_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeGitHub()
    api.after_delete_still_present = True
    monkeypatch.setattr(
        "tools.verified_merged_branch_prune.require_origin_checkout", lambda _: None
    )
    with pytest.raises(RuntimeError, match="still present"):
        _prune(api, apply=True)
    assert api.deleted == [BranchInfo(NAME, TIP, False)]


def test_origin_checkout_must_be_exact_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        returncode = 0
        stdout = "https://github.com/another/repo.git\n"
        stderr = ""

    monkeypatch.setattr(
        "tools.verified_merged_branch_prune.subprocess.run",
        lambda *args, **kwargs: Result(),
    )
    with pytest.raises(RuntimeError, match="origin"):
        require_origin_checkout(REPO)
