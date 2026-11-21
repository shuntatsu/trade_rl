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
from tools.branch_hygiene import (
    BranchInfo,
    MergedPullRequestSource,
    OpenPullRequestRefs,
)


def _ref(name: str, sha: str, *, protected: bool = False) -> BranchInfo:
    return BranchInfo(name, sha, protected)


def test_only_main_ancestor_work_branches_are_deletion_candidates() -> None:
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
        merged_pr_sources=(MergedPullRequestSource(merged_fix.name, merged_fix.sha),),
    )
    # A preserved archive is necessary but not sufficient to delete a branch
    # whose code is absent from main. The owner must separately review that work.
    assert [r.name for r in result] == ["fix/old-done"]


def test_open_pr_head_and_active_runs_protect_even_archived_tips() -> None:
    main = _ref("main", "1" * 40)
    archive = _ref(ARCHIVE_BRANCH, ARCHIVE_HEAD)
    a = _ref("fix/open-head", "2" * 40)
    b = _ref("fix/active-run", "3" * 40)
    opens = (OpenPullRequestRefs(a.name, a.sha, "main", main.sha),)
    assert not plan(
        (main, archive, a, b),
        archived={main.name: main.sha, a.name: a.sha, b.name: b.sha},
        main_reachable=frozenset({main.sha, a.sha, b.sha}),
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
    target = _ref("fix/old-source", "2" * 40)
    refs = (main, archive, target)
    snapshot = {r.name: r.sha for r in refs}
    monkeypatch.setattr(mod, "_archive_manifest", lambda: snapshot)
    monkeypatch.setattr(mod, "_git", lambda *args: main.sha)
    monkeypatch.setattr(mod, "_git_ancestor", lambda *args: True)
    monkeypatch.setattr(
        mod, "reachable_commits", lambda *args: frozenset({main.sha, target.sha})
    )
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

        def merged_pr_source_refs(
            self, *, include_nested: bool = False
        ) -> tuple[MergedPullRequestSource, ...]:
            assert include_nested
            return (MergedPullRequestSource(target.name, target.sha),)

        def delete_branches_with_leases(self, items: tuple[BranchInfo, ...]) -> None:
            self.removed.extend(b.name for b in items)

    api = Api()
    with pytest.raises(RuntimeError, match="PR/workflow/protection changed"):
        mod.retire(api, apply=True)
    assert api.removed == []


def test_apply_checks_post_delete_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.archived_branch_retirement as mod

    main = _ref("main", "1" * 40)
    archive = _ref(ARCHIVE_BRANCH, ARCHIVE_HEAD)
    target = _ref("fix/old-source", "2" * 40)
    refs = (main, archive, target)
    snapshot = {r.name: r.sha for r in refs}
    monkeypatch.setattr(mod, "_archive_manifest", lambda: snapshot)
    monkeypatch.setattr(mod, "_git", lambda *args: main.sha)
    monkeypatch.setattr(mod, "_git_ancestor", lambda *args: True)
    monkeypatch.setattr(
        mod, "reachable_commits", lambda *args: frozenset({main.sha, target.sha})
    )
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

        def merged_pr_source_refs(
            self, *, include_nested: bool = False
        ) -> tuple[MergedPullRequestSource, ...]:
            assert include_nested
            return (MergedPullRequestSource(target.name, target.sha),)

        def delete_branches_with_leases(self, items: tuple[BranchInfo, ...]) -> None:
            assert items == (target,)

    with pytest.raises(RuntimeError, match="post-delete readback"):
        mod.retire(Api(), apply=True)


def test_one_shot_workflow_auth_token_expression_is_valid() -> None:
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/retire-merged-pr-branches.yml"
    ).read_text(encoding="utf-8")
    token_expr = "$" + "{{ github.token }}"
    assert workflow.count("GH_TOKEN: " + token_expr) == 8
    assert "GH_TOKEN: }" not in workflow
    assert (
        "github.event.comment.body == '/retire-archived-branches-20261120" in workflow
    )


def test_main_ancestor_windows_m1_source_is_preserved() -> None:
    """The exceptional Windows marker branch is part of the M1 preservation rule."""
    main = _ref("main", "1" * 40)
    archive = _ref(ARCHIVE_BRANCH, ARCHIVE_HEAD)
    windows_m1 = _ref("fix/windows-job-child-pid-marker-20261105", "2" * 40)
    normal = _ref("fix/safely-merged-old", "3" * 40)
    result = plan(
        (main, archive, windows_m1, normal),
        archived={r.name: r.sha for r in (main, archive, windows_m1, normal)},
        main_reachable=frozenset({main.sha, windows_m1.sha, normal.sha}),
        open_refs=(),
        active_branches=frozenset(),
        merged_pr_sources=(
            MergedPullRequestSource(windows_m1.name, windows_m1.sha),
            MergedPullRequestSource(normal.name, normal.sha),
        ),
    )
    assert result == (normal,)


def _fake_snapshot_manifest() -> dict[str, object]:
    rows = [{"name": "main", "sha": "1" * 40}]
    rows.extend(
        {"name": f"fix/legacy-{i:02d}", "sha": f"{i + 2:040x}"} for i in range(84)
    )
    return {
        "schema": "trade_rl_branch_tip_snapshot_v1",
        "repository": "shuntatsu/trade_rl",
        "archive_branch": ARCHIVE_BRANCH,
        "main_head": "1" * 40,
        "branches": rows,
    }


def test_exact_archive_manifest_refuses_missing_main_and_noncanonical_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.archived_branch_retirement as mod

    current = _fake_snapshot_manifest()

    def fake_git(*args: str) -> str:
        if args[0] == "rev-parse":
            return ARCHIVE_HEAD
        if args[0] == "show":
            return json.dumps(current)
        raise AssertionError(f"unexpected Git command: {args!r}")

    monkeypatch.setattr(mod, "_git", fake_git)
    parsed = mod._archive_manifest()
    assert len(parsed) == 85 and parsed["main"] == "1" * 40

    current = _fake_snapshot_manifest()
    del current["main_head"]
    current["branches"] = [
        row for row in current["branches"] if row["name"] != "main"
    ] + [{"name": "fix/replacement-main", "sha": "f" * 40}]
    with pytest.raises(RuntimeError, match="main identity"):
        mod._archive_manifest()  # Neither absent main nor absent main_head may pass.

    current = _fake_snapshot_manifest()
    current["main_head"] = None
    with pytest.raises(RuntimeError, match="main identity"):
        mod._archive_manifest()

    current = _fake_snapshot_manifest()
    current["branches"][4]["name"] = current["branches"][3]["name"]
    with pytest.raises(RuntimeError, match="duplicate"):
        mod._archive_manifest()

    current = _fake_snapshot_manifest()
    current["branches"][4]["sha"] = "g" * 40
    with pytest.raises(RuntimeError, match="invalid"):
        mod._archive_manifest()

    current = _fake_snapshot_manifest()
    current["branches"].pop()
    with pytest.raises(RuntimeError, match="85 recorded"):
        mod._archive_manifest()

    monkeypatch.setattr(mod, "_git", lambda *args: "2" * 40)
    with pytest.raises(RuntimeError, match="archive ref moved"):
        mod._archive_manifest()


def _owner_event_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, main_sha: str
) -> None:
    event = {
        "issue": {"number": 1057},
        "comment": {
            "body": TRIGGER,
            "user": {"login": "shuntatsu", "id": 148511409},
        },
    }
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    for key, value in (
        ("GITHUB_ACTIONS", "true"),
        ("GITHUB_REPOSITORY", "shuntatsu/trade_rl"),
        ("GITHUB_ACTOR", "shuntatsu"),
        ("GITHUB_EVENT_NAME", "issue_comment"),
        ("GITHUB_EVENT_PATH", str(event_path)),
        ("GITHUB_SHA", main_sha),
    ):
        monkeypatch.setenv(key, value)


def test_actual_origin_gate_accepts_issue_comment_in_synthetic_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess

    from tools.branch_hygiene import require_retirement_origin

    _owner_event_env(monkeypatch, tmp_path, "1" * 40)
    _require_owner_event()
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "remote",
            "add",
            "origin",
            "https://github.com/shuntatsu/trade_rl.git",
        ],
        check=True,
    )
    monkeypatch.chdir(repo)
    require_retirement_origin()  # Actual imported source, not monkeypatched away.

    subprocess.run(
        ["git", "remote", "set-url", "origin", "https://github.com/other/repo.git"],
        check=True,
    )
    with pytest.raises(RuntimeError, match="exact trade_rl"):
        require_retirement_origin()


def test_more_than_max_deletes_refuses_without_truncating() -> None:
    import tools.archived_branch_retirement as mod

    main, archive = _ref("main", "1" * 40), _ref(ARCHIVE_BRANCH, ARCHIVE_HEAD)
    candidates = tuple(
        _ref(f"fix/retire-{i:02d}", f"{i + 2:040x}") for i in range(mod.MAX_DELETES + 1)
    )
    all_refs = (main, archive, *candidates)
    with pytest.raises(RuntimeError, match="safety cap"):
        plan(
            all_refs,
            archived={entry.name: entry.sha for entry in all_refs},
            main_reachable=frozenset({main.sha, *(c.sha for c in candidates)}),
            open_refs=(),
            active_branches=frozenset(),
            merged_pr_sources=tuple(
                MergedPullRequestSource(b.name, b.sha) for b in candidates
            ),
        )


def test_preview_then_owner_authorized_apply_deletes_multiple_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess

    import tools.archived_branch_retirement as mod

    main, archive = _ref("main", "1" * 40), _ref(ARCHIVE_BRANCH, ARCHIVE_HEAD)
    candidates = tuple(
        _ref(f"fix/closed-{i:02d}", f"{i + 2:040x}") for i in range(mod.BATCH_SIZE + 1)
    )
    starting = (main, archive, *candidates)
    retained = {entry.name: entry.sha for entry in starting}
    monkeypatch.setattr(mod, "_archive_manifest", lambda: retained)
    monkeypatch.setattr(mod, "_git", lambda *args: main.sha)
    monkeypatch.setattr(mod, "_git_ancestor", lambda *args: True)
    monkeypatch.setattr(
        mod,
        "reachable_commits",
        lambda *args: frozenset({main.sha, *(x.sha for x in candidates)}),
    )

    class FakeApi:
        _repository = "shuntatsu/trade_rl"

        def __init__(self) -> None:
            self.refs = list(starting)
            self.batches: list[tuple[BranchInfo, ...]] = []

        def branches(self) -> tuple[BranchInfo, ...]:
            return tuple(self.refs)

        def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
            return ()

        def active_workflow_branch_names(self) -> frozenset[str]:
            return frozenset()

        def merged_pr_source_refs(
            self, *, include_nested: bool = False
        ) -> tuple[MergedPullRequestSource, ...]:
            assert include_nested
            return tuple(MergedPullRequestSource(x.name, x.sha) for x in candidates)

        def delete_branches_with_leases(self, items: tuple[BranchInfo, ...]) -> None:
            assert all(x in self.refs for x in items)
            self.batches.append(tuple(items))
            self.refs = [x for x in self.refs if x not in items]

    fake = FakeApi()
    count, selected, deleted = mod.retire(fake, apply=False)
    assert count == len(starting)
    assert selected == tuple(x.name for x in candidates)
    assert deleted == ()
    assert fake.refs == list(starting) and fake.batches == []

    _owner_event_env(monkeypatch, tmp_path, main.sha)
    _require_owner_event()  # Actual author/id/comment verification.

    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "remote",
            "add",
            "origin",
            "https://github.com/shuntatsu/trade_rl.git",
        ],
        check=True,
    )
    monkeypatch.chdir(repo)

    # Production require_retirement_origin is *not* stubbed; only HTTP API is fake.
    count, selected, deleted = mod.retire(fake, apply=True)
    assert count == len(starting)
    assert selected == deleted == tuple(x.name for x in candidates)
    assert list(map(len, fake.batches)) == [mod.BATCH_SIZE, 1]
    assert fake.batches[0] == candidates[: mod.BATCH_SIZE]
    assert fake.batches[1] == candidates[mod.BATCH_SIZE :]
    assert fake.refs == [main, archive]


def test_snapshot_rejects_current_main_drift_before_any_remote_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.archived_branch_retirement as mod

    main, archive = _ref("main", "1" * 40), _ref(ARCHIVE_BRANCH, ARCHIVE_HEAD)

    class FakeApi:
        def branches(self) -> tuple[BranchInfo, ...]:
            return (main, archive)

    monkeypatch.setattr(mod, "_git", lambda *args: "2" * 40)
    with pytest.raises(RuntimeError, match="differs from checked-out history"):
        mod._snapshot(FakeApi(), {"main": main.sha})  # type: ignore[arg-type]


def test_cli_preview_and_one_shot_path_emit_summary_without_leaking_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import tools.archived_branch_retirement as mod

    monkeypatch.setenv("GH_TOKEN", "synthetic-not-a-real-token")
    summary = tmp_path / "summary.txt"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    class FakeApi:
        def __init__(self, *, repository: str, token: str, api_url: str) -> None:
            assert repository == "shuntatsu/trade_rl"
            assert token == "synthetic-not-a-real-token"
            assert api_url == "https://api.github.com"

    seen: list[bool] = []
    monkeypatch.setattr(mod, "GitHubApi", FakeApi)

    def fake_retire(
        api: object, *, apply: bool
    ) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
        seen.append(apply)
        return 5, ("fix/archive-ancestor",), ("fix/archive-ancestor",) if apply else ()

    monkeypatch.setattr(mod, "retire", fake_retire)

    assert mod.main([]) == 0
    assert mod.main(["--apply"]) == 0
    assert seen == [False, True]
    output = capsys.readouterr().out
    assert "mode: preview" in output and "mode: apply" in output
    assert "deleted and verified: 1" in output
    assert "synthetic-not-a-real-token" not in output
    assert summary.read_text(encoding="utf-8").count("## Archived one-shot") == 2


def test_archive_main_ancestor_without_exact_merged_pr_is_never_retired() -> None:
    main = _ref("main", "1" * 40)
    archive = _ref(ARCHIVE_BRANCH, ARCHIVE_HEAD)
    real = _ref("fix/exact-merged", "2" * 40)
    orphan = _ref("fix/old-ancestor-orphan", "3" * 40)
    stale = _ref("fix/stale-merged-sha", "4" * 40)
    entries = (main, archive, real, orphan, stale)
    selected = plan(
        entries,
        archived={r.name: r.sha for r in entries},
        main_reachable=frozenset(r.sha for r in entries),
        open_refs=(),
        active_branches=frozenset(),
        merged_pr_sources=(
            MergedPullRequestSource(real.name, real.sha),
            MergedPullRequestSource(stale.name, "5" * 40),
        ),
    )
    assert selected == (real,)
