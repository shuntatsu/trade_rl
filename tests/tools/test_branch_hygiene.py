from pathlib import Path

import pytest

from tools.branch_hygiene import (
    RETENTION_BRANCH,
    BranchDecision,
    BranchInfo,
    OpenPullRequestRefs,
    anchor_branch_names,
    anchor_shas,
    apply_cleanup,
    parse_retention_log,
    plan_cleanup,
)


def test_anchor_selection_preserves_durable_refs() -> None:
    branches = (
        BranchInfo("main", "a" * 40, False),
        BranchInfo("protected/release", "b" * 40, True),
        BranchInfo("feature/open", "c" * 40, False),
        BranchInfo("stack/base", "d" * 40, False),
        BranchInfo("research/study", "e" * 40, False),
        BranchInfo("seal/protocol", "f" * 40, False),
        BranchInfo("freeze/source", "1" * 40, False),
        BranchInfo("run/evaluation", "2" * 40, False),
        BranchInfo(RETENTION_BRANCH, "3" * 40, False),
        BranchInfo("verify/redundant", "4" * 40, False),
    )
    open_refs = (
        OpenPullRequestRefs(
            head_name="feature/open",
            head_sha="c" * 40,
            base_name="stack/base",
            base_sha="d" * 40,
        ),
    )

    names = anchor_branch_names(
        branches,
        default_branch="main",
        open_pull_request_refs=open_refs,
        active_workflow_branches=frozenset(),
    )
    shas = anchor_shas(
        branches,
        default_branch="main",
        open_pull_request_refs=open_refs,
        active_workflow_branches=frozenset(),
    )

    assert names == {
        "main",
        "protected/release",
        "feature/open",
        "stack/base",
        "research/study",
        "seal/protocol",
        "freeze/source",
        "run/evaluation",
        RETENTION_BRANCH,
    }
    assert shas == {
        "a" * 40,
        "b" * 40,
        "c" * 40,
        "d" * 40,
        "e" * 40,
        "f" * 40,
        "1" * 40,
        "2" * 40,
        "3" * 40,
    }


def test_cleanup_archives_every_unique_non_anchor_tip() -> None:
    branches = (
        BranchInfo("main", "a" * 40, False),
        BranchInfo("feature/open", "b" * 40, False),
        BranchInfo("stack/base", "c" * 40, False),
        BranchInfo("verify/absorbed", "d" * 40, False),
        BranchInfo("tmp/unique-red", "e" * 40, False),
        BranchInfo("automation/unique", "f" * 40, False),
        BranchInfo("feature/unique", "1" * 40, False),
        BranchInfo("feature/merged", "2" * 40, False),
        BranchInfo("research/evidence", "3" * 40, False),
    )
    open_refs = (
        OpenPullRequestRefs(
            head_name="feature/open",
            head_sha="b" * 40,
            base_name="stack/base",
            base_sha="c" * 40,
        ),
    )
    decisions = plan_cleanup(
        branches,
        default_branch="main",
        open_pull_request_refs=open_refs,
        active_workflow_branches=frozenset(),
        reachable_from_anchors=frozenset(
            {"a" * 40, "b" * 40, "c" * 40, "d" * 40, "2" * 40}
        ),
        retention_observed_at={
            ("verify/absorbed", "d" * 40): 0,
            ("feature/merged", "2" * 40): 0,
        },
        now_timestamp=10_000,
    )

    by_name = {decision.branch.name: decision for decision in decisions}
    assert by_name["main"].reason == "default-branch"
    assert by_name["feature/open"].reason == "open-pr-ref"
    assert by_name["stack/base"].reason == "open-pr-ref"
    assert by_name["research/evidence"].reason == "provenance"
    assert by_name["verify/absorbed"].action == "delete"
    assert by_name["feature/merged"].action == "delete"
    assert by_name["tmp/unique-red"].action == "archive-keep"
    assert by_name["automation/unique"].action == "archive-keep"
    assert by_name["tmp/unique-red"].reason == "first-seen-non-anchor-tip"
    assert by_name["automation/unique"].reason == "first-seen-non-anchor-tip"
    assert by_name["feature/unique"].action == "archive-keep"
    assert by_name["feature/unique"].reason == "first-seen-non-anchor-tip"


def test_fork_pr_and_active_workflow_refs_are_preserved() -> None:
    branches = (
        BranchInfo("main", "a" * 40, False),
        BranchInfo("release/base", "b" * 40, False),
        BranchInfo("verify/running", "c" * 40, False),
    )
    open_refs = (
        OpenPullRequestRefs(
            head_name=None,
            head_sha=None,
            base_name="release/base",
            base_sha="b" * 40,
        ),
    )

    decisions = plan_cleanup(
        branches,
        default_branch="main",
        open_pull_request_refs=open_refs,
        active_workflow_branches=frozenset({"verify/running"}),
        reachable_from_anchors=frozenset({"a" * 40, "b" * 40, "c" * 40}),
        retention_observed_at={},
        now_timestamp=10_000,
    )

    by_name = {decision.branch.name: decision for decision in decisions}
    assert by_name["release/base"].reason == "open-pr-ref"
    assert by_name["verify/running"].reason == "active-workflow"


class FakeCleanupApi:
    def __init__(
        self,
        branches: tuple[BranchInfo, ...],
        open_refs: tuple[OpenPullRequestRefs, ...] = (),
    ) -> None:
        self._branches = {branch.name: branch for branch in branches}
        self._open_refs = open_refs
        self._commit_index = 0
        self.deleted: list[str] = []
        self.events: list[str] = []
        self.messages: list[str] = []

    def branches(self) -> tuple[BranchInfo, ...]:
        return tuple(self._branches.values())

    def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
        return self._open_refs

    def active_workflow_branch_names(self) -> frozenset[str]:
        return frozenset()

    def git_commit_tree_sha(self, commit_sha: str) -> str:
        self.events.append(f"tree:{commit_sha}")
        return "f" * 40

    def create_retention_commit(
        self,
        *,
        tree_sha: str,
        parent_shas: tuple[str, ...],
        message: str,
    ) -> str:
        assert tree_sha == "f" * 40
        assert parent_shas
        self._commit_index += 1
        sha = f"{self._commit_index:040x}"
        self.events.append(f"commit:{sha}")
        self.messages.append(message)
        return sha

    def create_branch_ref(self, name: str, sha: str) -> None:
        self.events.append(f"create-ref:{name}:{sha}")
        self._branches[name] = BranchInfo(name, sha, False)

    def update_branch_ref(self, name: str, sha: str) -> None:
        self.events.append(f"update-ref:{name}:{sha}")
        self._branches[name] = BranchInfo(name, sha, False)

    def delete_branches_with_leases(self, branches: tuple[BranchInfo, ...]) -> None:
        names = ",".join(branch.name for branch in branches)
        self.events.append(f"delete:{names}")
        for branch in branches:
            self.deleted.append(branch.name)
            self._branches.pop(branch.name)


def test_apply_cleanup_archives_unique_tip_before_delete() -> None:
    main = BranchInfo("main", "a" * 40, False)
    archived = BranchInfo("verify/unique", "b" * 40, False)
    direct = BranchInfo("feature/merged", "c" * 40, False)
    api = FakeCleanupApi((main, archived, direct))
    decisions = (
        BranchDecision(main, "keep", "default-branch"),
        BranchDecision(archived, "archive-keep", "first-seen-non-anchor-tip"),
        BranchDecision(direct, "delete", "tip-reachable-from-anchor"),
    )

    deleted = apply_cleanup(api, decisions, default_branch="main")  # type: ignore[arg-type]

    assert deleted == (direct.name,)
    archive_event = next(
        index
        for index, event in enumerate(api.events)
        if event.startswith("create-ref:")
    )
    delete_event = next(
        index for index, event in enumerate(api.events) if event.startswith("delete:")
    )
    assert archive_event < delete_event
    assert archived.name in {branch.name for branch in api.branches()}
    assert RETENTION_BRANCH in {branch.name for branch in api.branches()}
    assert f"{archived.name}\t{archived.sha}" in api.messages[0]


def test_apply_cleanup_revalidates_open_refs_and_exact_sha() -> None:
    main = BranchInfo("main", "a" * 40, False)
    original = BranchInfo("verify/redundant", "b" * 40, False)
    moved = BranchInfo("automation/moved", "c" * 40, False)
    protected_by_pr = BranchInfo("tmp/open", "d" * 40, False)
    api = FakeCleanupApi(
        (
            main,
            original,
            BranchInfo(moved.name, "e" * 40, False),
            protected_by_pr,
        ),
        (
            OpenPullRequestRefs(
                head_name=protected_by_pr.name,
                head_sha=protected_by_pr.sha,
                base_name="main",
                base_sha=main.sha,
            ),
        ),
    )
    decisions = (
        BranchDecision(original, "delete", "tip-reachable-from-anchor"),
        BranchDecision(moved, "delete", "tip-reachable-from-anchor"),
        BranchDecision(protected_by_pr, "delete", "tip-reachable-from-anchor"),
    )

    deleted = apply_cleanup(api, decisions, default_branch="main")  # type: ignore[arg-type]

    assert deleted == (original.name,)
    assert api.deleted == [original.name]


def test_delete_uses_atomic_exact_sha_leases(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.branch_hygiene import GitHubApi

    observed: dict[str, object] = {}

    class Process:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command: list[str], **kwargs: object) -> Process:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr("tools.branch_hygiene.subprocess.run", fake_run)
    api = GitHubApi(
        repository="owner/repo",
        token="secret-token",
        api_url="https://api.github.com",
    )
    branches = (
        BranchInfo("verify/one", "1" * 40, False),
        BranchInfo("tmp/two", "2" * 40, False),
    )

    api.delete_branches_with_leases(branches)

    command = observed["command"]
    assert isinstance(command, list)
    assert "--atomic" in command
    assert "--porcelain" in command
    assert (
        f"--force-with-lease=refs/heads/{branches[0].name}:{branches[0].sha}" in command
    )
    assert (
        f"--force-with-lease=refs/heads/{branches[1].name}:{branches[1].sha}" in command
    )
    assert f":refs/heads/{branches[0].name}" in command
    assert f":refs/heads/{branches[1].name}" in command
    assert all("secret-token" not in part for part in command)


def test_branch_hygiene_workflow_runs_from_default_branch() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    workflow = (repository_root / ".github/workflows/branch-hygiene.yml").read_text(
        encoding="utf-8"
    )

    assert "push:" in workflow
    assert "- main" in workflow
    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "pull_request_target:" not in workflow
    assert "contents: write" in workflow
    assert "pull-requests: read" in workflow
    assert "actions: read" in workflow
    assert "ref: ${{ github.event.repository.default_branch }}" in workflow
    assert "persist-credentials: false" in workflow
    assert "python3 -m tools.branch_hygiene" in workflow


def test_repository_root_url_has_no_trailing_slash() -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo",
        token="token",
        api_url="https://api.github.test/",
    )

    assert api._url("") == "https://api.github.test/repos/owner/repo"
    assert (
        api._url("branches", {"page": "1"})
        == "https://api.github.test/repos/owner/repo/branches?page=1"
    )


def test_cleanup_requires_observation_and_grace_before_delete() -> None:
    main = BranchInfo("main", "a" * 40, False)
    first_seen = BranchInfo("feature/first", "b" * 40, False)
    recent = BranchInfo("feature/recent", "c" * 40, False)
    stale = BranchInfo("feature/stale", "d" * 40, False)
    unreachable = BranchInfo("feature/unreachable", "e" * 40, False)
    now = 10_000
    decisions = plan_cleanup(
        (main, first_seen, recent, stale, unreachable),
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_anchors=frozenset({main.sha, recent.sha, stale.sha}),
        retention_observed_at={
            (recent.name, recent.sha): now - 30,
            (stale.name, stale.sha): now - 3_601,
            (unreachable.name, unreachable.sha): now - 3_601,
        },
        now_timestamp=now,
    )

    by_name = {decision.branch.name: decision for decision in decisions}
    assert by_name[first_seen.name].action == "archive-keep"
    assert by_name[first_seen.name].reason == "first-seen-non-anchor-tip"
    assert by_name[recent.name].action == "keep"
    assert by_name[recent.name].reason == "observation-grace-period"
    assert by_name[stale.name].action == "delete"
    assert by_name[stale.name].reason == "observed-tip-reachable-from-anchor"
    assert by_name[unreachable.name].action == "keep"
    assert by_name[unreachable.name].reason == "retained-tip-not-reachable"


def test_retention_log_parser_uses_earliest_observation() -> None:
    branch = "automation/recent"
    sha = "a" * 40
    old_message = (
        f"Archive unique non-anchor branch tips before ref cleanup\n\n{branch}\t{sha}\n"
    )
    new_message = (
        "Observe non-anchor branch tips before cleanup grace period\n\n"
        f"{branch}\t{sha}\n"
    )
    payload = (
        f"200\x1f{new_message}\x1e100\x1f{old_message}\x1e50\x1funrelated commit\x1e"
    )

    assert parse_retention_log(payload) == {(branch, sha): 100}
