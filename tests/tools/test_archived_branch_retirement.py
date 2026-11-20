"""Regressions for exact-history archived branch cleanup without economic access."""

import json
from pathlib import Path

import pytest

from tools.archived_branch_retirement import (
    ARCHIVE_BRANCH,
    ARCHIVE_HEAD,
    TRIGGER,
    _require_owner_event,
    plan,
)
from tools.branch_hygiene import BranchInfo, OpenPullRequestRefs


def _ref(name: str, sha: str, *, protected: bool = False) -> BranchInfo:
    return BranchInfo(name, sha, protected)


def test_archived_merged_and_old_work_candidates_are_bounded() -> None:
    main, archive = _ref("main", "1" * 40), _ref(ARCHIVE_BRANCH, ARCHIVE_HEAD)
    merged_fix = _ref("fix/old-done", "2" * 40)
    unmerged_codex = _ref("codex/old-experiment", "3" * 40)
    old_review = _ref("review/closed-shard", "4" * 40)
    unmerged_fix = _ref("fix/new-unsolved", "5" * 40)
    research = _ref("research/frozen-signal", "6" * 40)
    m1 = _ref("fix/m1-preserve-audit", "7" * 40)
    protected = _ref("fix/protected", "8" * 40, protected=True)
    open_base = _ref("feat/used-parent", "9" * 40)
    active = _ref("codex/active", "a" * 40)
    changed = _ref("codex/changed-since-archive", "b" * 40)
    unarchived = _ref("codex/new-unknown", "c" * 40)
    refs = (
        main,
        archive,
        merged_fix,
        unmerged_codex,
        old_review,
        unmerged_fix,
        research,
        m1,
        protected,
        open_base,
        active,
        changed,
        unarchived,
    )
    snapshot = {r.name: r.sha for r in refs if r != unarchived}
    snapshot[changed.name] = "d" * 40
    opens = (OpenPullRequestRefs("other", "e" * 40, open_base.name, open_base.sha),)
    result = plan(
        refs,
        archived=snapshot,
        main_reachable=frozenset(
            {main.sha, merged_fix.sha, protected.sha, open_base.sha, m1.sha}
        ),
        open_refs=opens,
        active_branches=frozenset({active.name}),
    )
    assert [r.name for r in result] == [
        "codex/old-experiment",
        "fix/old-done",
        "review/closed-shard",
    ]


def test_open_pr_head_and_active_runs_protect_even_archived_tips() -> None:
    main = _ref("main", "1" * 40)
    archive = _ref(ARCHIVE_BRANCH, ARCHIVE_HEAD)
    a = _ref("codex/open-head", "2" * 40)
    b = _ref("review/active-run", "3" * 40)
    opens = (OpenPullRequestRefs(a.name, a.sha, "main", main.sha),)
    assert not plan(
        (main, archive, a, b),
        archived={main.name: main.sha, a.name: a.sha, b.name: b.sha},
        main_reachable=frozenset({main.sha}),
        open_refs=opens,
        active_branches=frozenset({b.name}),
    )


def test_main_and_archive_must_exist_and_snapshots_must_be_unique() -> None:
    main = _ref("main", "1" * 40)
    archive = _ref(ARCHIVE_BRANCH, ARCHIVE_HEAD)
    kwargs = dict(
        archived={main.name: main.sha},
        main_reachable=frozenset({main.sha}),
        open_refs=(),
        active_branches=frozenset(),
    )
    with pytest.raises(RuntimeError, match="missing"):
        plan((main,), **kwargs)
    with pytest.raises(RuntimeError, match="duplicate"):
        plan((main, archive, main), **kwargs)


def test_owner_issue_command_requires_all_event_authorities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    event = {
        "issue": {"number": 1057},
        "comment": {"body": TRIGGER, "user": {"login": "shuntatsu", "id": 148511409}},
    }
    p = tmp_path / "event.json"
    p.write_text(json.dumps(event), encoding="utf-8")
    for key, value in (
        ("GITHUB_ACTIONS", "true"),
        ("GITHUB_REPOSITORY", "shuntatsu/trade_rl"),
        ("GITHUB_ACTOR", "shuntatsu"),
        ("GITHUB_EVENT_NAME", "issue_comment"),
        ("GITHUB_EVENT_PATH", str(p)),
    ):
        monkeypatch.setenv(key, value)
    _require_owner_event()
    event["comment"]["body"] = "not authorised"
    p.write_text(json.dumps(event), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing exact"):
        _require_owner_event()
    event["comment"]["body"] = TRIGGER
    p.write_text(json.dumps(event), encoding="utf-8")
    monkeypatch.setenv("GITHUB_ACTOR", "other")
    with pytest.raises(RuntimeError, match="requires the owner"):
        _require_owner_event()


def test_apply_requires_fresh_pr_and_workflow_roster_before_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.archived_branch_retirement as mod

    main = _ref("main", "1" * 40)
    archive = _ref(ARCHIVE_BRANCH, ARCHIVE_HEAD)
    target = _ref("codex/old-source", "2" * 40)
    refs = (main, archive, target)
    snapshot = {r.name: r.sha for r in refs}
    monkeypatch.setattr(mod, "_archive_manifest", lambda: snapshot)
    monkeypatch.setattr(mod, "_git", lambda *args: main.sha)
    monkeypatch.setattr(mod, "_git_ancestor", lambda *args: True)
    monkeypatch.setattr(mod, "reachable_commits", lambda *args: frozenset({main.sha}))
    monkeypatch.setattr(mod, "_require_owner_event", lambda: None)
    monkeypatch.setattr(mod, "require_retirement_origin", lambda: None)
    monkeypatch.setenv("GITHUB_SHA", main.sha)

    class Api:
        _repository = "shuntatsu/trade_rl"
        removed: list[str] = []
        calls = 0

        def branches(self) -> tuple[BranchInfo, ...]:
            return refs

        def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
            self.calls += 1
            if self.calls >= 2:
                return (OpenPullRequestRefs(target.name, target.sha, "main", main.sha),)
            return ()

        def active_workflow_branch_names(self) -> frozenset[str]:
            return frozenset()

        def delete_branches_with_leases(self, items: tuple[BranchInfo, ...]) -> None:
            self.removed.extend(b.name for b in items)

    api = Api()
    with pytest.raises(RuntimeError, match="changed before deletion"):
        mod.retire(api, apply=True)
    assert api.removed == []


def test_apply_checks_post_delete_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.archived_branch_retirement as mod

    main = _ref("main", "1" * 40)
    archive = _ref(ARCHIVE_BRANCH, ARCHIVE_HEAD)
    target = _ref("codex/old-source", "2" * 40)
    refs = (main, archive, target)
    snapshot = {r.name: r.sha for r in refs}
    monkeypatch.setattr(mod, "_archive_manifest", lambda: snapshot)
    monkeypatch.setattr(mod, "_git", lambda *args: main.sha)
    monkeypatch.setattr(mod, "_git_ancestor", lambda *args: True)
    monkeypatch.setattr(mod, "reachable_commits", lambda *args: frozenset({main.sha}))
    monkeypatch.setattr(mod, "_require_owner_event", lambda: None)
    monkeypatch.setattr(mod, "require_retirement_origin", lambda: None)
    monkeypatch.setenv("GITHUB_SHA", main.sha)

    class Api:
        _repository = "shuntatsu/trade_rl"

        def branches(self) -> tuple[BranchInfo, ...]:
            return refs  # deliberately fails to reflect the deletion

        def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
            return ()

        def active_workflow_branch_names(self) -> frozenset[str]:
            return frozenset()

        def delete_branches_with_leases(self, items: tuple[BranchInfo, ...]) -> None:
            assert items == (target,)

    with pytest.raises(RuntimeError, match="post-deletion"):
        mod.retire(Api(), apply=True)


def test_one_shot_workflow_auth_token_expression_is_valid() -> None:
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/retire-merged-pr-branches.yml"
    ).read_text(encoding="utf-8")
    token_expr = "$" + "{{ github.token }}"
    assert workflow.count("GH_TOKEN: " + token_expr) == 4
    assert "GH_TOKEN: }" not in workflow
    assert (
        "github.event.comment.body == '/retire-archived-branches-20261120" in workflow
    )
