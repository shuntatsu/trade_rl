"""Only fully integrated, independently retained PR source branches may be pruned."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.branch_hygiene import BranchInfo, OpenPullRequestRefs
from tools.verified_merged_branch_prune import (
    VerifiedMergedPrune,
    main,
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
PREVIOUS_RETENTION = "b" * 40
INTERMEDIATE_RETENTION = "6" * 40
TREE = "7" * 40
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
            "tree": {"sha": TREE},
            "message": (
                "Retain merged PR #1076 exact source after verified normal integration\n\n"
                + NAME
                + "\t"
                + TIP
            ),
            "committer": {"date": "2026-11-01T12:00:00Z"},
            "parents": [
                {"sha": "b" * 40},
                {"sha": TIP},
            ],
        }
        self.retention_tip: dict[str, object] = {
            "sha": RETENTION,
            "tree": {"sha": TREE},
            "parents": [{"sha": RECORD}, {"sha": TIP}],
        }
        self.intermediate_retention: dict[str, object] = {
            "sha": INTERMEDIATE_RETENTION,
            "tree": {"sha": TREE},
            "parents": [{"sha": RECORD}, {"sha": TIP}],
        }
        self.prior_retention: dict[str, object] = {
            "sha": PREVIOUS_RETENTION,
            "tree": {"sha": TREE},
            "parents": [],
        }
        self.source_commit: dict[str, object] = {
            "sha": TIP,
            "tree": {"sha": TREE},
            "parents": [],
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
        if suffix == "git/commits/" + RETENTION:
            return self.retention_tip
        if suffix == "git/commits/" + INTERMEDIATE_RETENTION:
            return self.intermediate_retention
        if suffix == "git/commits/" + PREVIOUS_RETENTION:
            return self.prior_retention
        if suffix == "git/commits/" + TIP:
            return self.source_commit
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
                item
                for item in self.records
                if item.name not in {b.name for b in branches}
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
    assert {b.name for b in api.branches()} == {"main", "provenance/branch-retention"}


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
    api.retention_record["message"] = (
        "Retain merged PR #1076 exact source after verified normal integration"
    )
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


def test_exact_merged_pr_retention_title_is_accepted() -> None:
    api = FakeGitHub()
    result = _prune(api)
    assert result.deleted is False
    assert result.merged_pr_number == 1076
    assert api.deleted == []


@pytest.mark.parametrize(
    "extra_mapping",
    [
        f"\nother/merged\t{'9' * 40}",
        f"\n{NAME}\t{TIP}",
        "\nnot-a-branch\tmalformed-sha",
    ],
)
def test_retention_record_with_extra_mapping_line_is_rejected(
    extra_mapping: str,
) -> None:
    api = FakeGitHub()
    api.retention_record["message"] += extra_mapping
    with pytest.raises(ValueError, match="retention record"):
        _prune(api)
    assert api.deleted == []


@pytest.mark.parametrize(
    "parents",
    [
        [{"sha": "b" * 40}, {"sha": TIP}, {"sha": "9" * 40}],
        [{"sha": "b" * 40}, {"sha": "9" * 40}, {"sha": TIP}],
    ],
)
def test_retention_record_must_have_exactly_one_source_parent(
    parents: list[dict[str, str]],
) -> None:
    api = FakeGitHub()
    api.retention_record["parents"] = parents
    with pytest.raises(ValueError, match="retention record"):
        _prune(api)
    assert api.deleted == []


@pytest.mark.parametrize(
    "title",
    [
        "Observe non-anchor branch tips before cleanup grace period",
        "Archive unique non-anchor branch tips before ref cleanup",
        "Observe merged PR #1076 source before cleanup grace period",
        "Retain merged PR #1077 exact source after verified normal integration",
        "Retain merged PR #1076 exact source after verified normal integration extra",
        "Retain merged PR #01076 exact source after verified normal integration",
    ],
)
def test_obsolete_or_ambiguous_retention_titles_fail_closed(title: str) -> None:
    api = FakeGitHub()
    api.retention_record["message"] = f"{title}\n\n{NAME}\t{TIP}"
    with pytest.raises(ValueError, match="retention record"):
        _prune(api)
    assert api.deleted == []


def test_record_tree_must_equal_first_retention_parent_tree() -> None:
    api = FakeGitHub()
    api.prior_retention["tree"] = {"sha": "8" * 40}
    with pytest.raises(ValueError, match="retention tree"):
        _prune(api)
    assert api.deleted == []


def test_retention_tip_must_preserve_first_parent_tree() -> None:
    api = FakeGitHub()
    api.retention_tip["tree"] = {"sha": "8" * 40}
    with pytest.raises(ValueError, match="retention tree"):
        _prune(api)
    assert api.deleted == []


def test_retention_record_must_be_in_first_parent_history() -> None:
    api = FakeGitHub()
    # Ordinary graph reachability still succeeds through the second parent.
    api.retention_tip["parents"] = [{"sha": TIP}, {"sha": RECORD}]
    with pytest.raises(ValueError, match="first-parent"):
        _prune(api)
    assert api.deleted == []


def test_missing_first_parent_ref_is_not_proof_of_retention() -> None:
    api = FakeGitHub()
    api.retention_tip["parents"] = []
    with pytest.raises(ValueError, match="first-parent"):
        _prune(api)
    assert api.deleted == []


def test_retention_first_parent_cycle_is_rejected() -> None:
    api = FakeGitHub()
    api.retention_tip["parents"] = [{"sha": RETENTION}, {"sha": RECORD}]
    with pytest.raises(ValueError, match="first-parent"):
        _prune(api)
    assert api.deleted == []


def test_retention_first_parent_walk_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeGitHub()
    monkeypatch.setattr(
        "tools.verified_merged_branch_prune._MAX_RETENTION_FIRST_PARENT_HOPS", 1
    )
    with pytest.raises(ValueError, match="first-parent"):
        _prune(api)
    assert api.deleted == []


def test_retention_record_missing_tree_fails_closed() -> None:
    api = FakeGitHub()
    api.retention_record.pop("tree")
    with pytest.raises(ValueError, match="retention tree"):
        _prune(api)
    assert api.deleted == []


def test_record_at_current_retention_tip_is_valid() -> None:
    api = FakeGitHub()
    api.records[1] = BranchInfo("provenance/branch-retention", RECORD, False)
    result = _prune(api)
    assert result.retention_sha == RECORD
    assert result.deleted is False


def test_retention_lineage_accepts_multiple_unchanged_first_parent_edges() -> None:
    api = FakeGitHub()
    api.retention_tip["parents"] = [{"sha": INTERMEDIATE_RETENTION}, {"sha": TIP}]
    assert _prune(api).deleted is False
    assert f"git/commits/{INTERMEDIATE_RETENTION}" in api.requests


def test_intermediate_retention_tree_drift_is_rejected() -> None:
    api = FakeGitHub()
    api.retention_tip["parents"] = [{"sha": INTERMEDIATE_RETENTION}, {"sha": TIP}]
    api.intermediate_retention["tree"] = {"sha": "8" * 40}
    with pytest.raises(ValueError, match="retention tree"):
        _prune(api)
    assert api.deleted == []


def test_retention_first_parent_readback_identity_must_match_sha() -> None:
    api = FakeGitHub()
    api.prior_retention["sha"] = "8" * 40
    with pytest.raises(ValueError, match="first-parent"):
        _prune(api)
    assert api.deleted == []


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


def test_apply_must_be_explicit_native_boolean() -> None:
    api = FakeGitHub()
    with pytest.raises(ValueError, match="apply must be a native boolean"):
        prune_verified_merged_branch(
            api,
            repository=REPO,
            branch=NAME,
            expected_sha=TIP,
            merged_pr_number=1076,
            retention_record_sha=RECORD,
            now=NOW,
            apply="False",
        )
    assert api.requests == []


def test_protected_merged_branch_stays_protected() -> None:
    api = FakeGitHub()
    api.records[-1] = BranchInfo(NAME, TIP, True)
    with pytest.raises(ValueError, match="protected branch"):
        _prune(api)
    assert api.deleted == []


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


def test_manual_prune_workflow_is_never_automatically_triggered() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/prune-merged-branch.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch:" in workflow
    assert "\n  push:" not in workflow
    assert "\n  schedule:" not in workflow
    assert "contents: write" in workflow
    assert "pull-requests: read" in workflow
    assert "actions: read" in workflow
    assert "persist-credentials: false" in workflow
    assert 'test "$CONFIRM" = "true"' in workflow
    assert '--expected-sha "$PRUNE_SHA"' in workflow
    assert '--retention-record-sha "$PRUNE_RETENTION"' in workflow
    assert "--apply" in workflow


def test_cli_pins_api_github_and_never_accepts_token_on_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_api(*, repository: str, token: str, api_url: str) -> object:
        captured.update(repository=repository, token=token, api_url=api_url)
        return object()

    def fake_prune(api: object, **kwargs: object) -> VerifiedMergedPrune:
        assert api is not None
        assert kwargs["apply"] is False
        return VerifiedMergedPrune(NAME, TIP, MAIN, RETENTION, 1076)

    monkeypatch.setenv("GH_TOKEN", "fixture-secret")
    monkeypatch.setattr("tools.verified_merged_branch_prune.GitHubApi", fake_api)
    monkeypatch.setattr(
        "tools.verified_merged_branch_prune.prune_verified_merged_branch",
        fake_prune,
    )
    args = [
        "--repository",
        REPO,
        "--branch",
        NAME,
        "--expected-sha",
        TIP,
        "--merged-pr-number",
        "1076",
        "--retention-record-sha",
        RECORD,
    ]
    assert main(args) == 0
    assert captured == {
        "repository": REPO,
        "token": "fixture-secret",
        "api_url": "https://api.github.com",
    }
    with pytest.raises(SystemExit):
        main([*args, "--api-url", "https://untrusted.example"])
    with pytest.raises(SystemExit):
        main([*args, "--token", "exposed-in-process-table"])
