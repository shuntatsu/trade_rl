"""Owner-scoped old main-ancestor ref retirement; safe synthetic regression tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools import archived_orphan_ancestor_retirement as module
from tools.branch_hygiene import BranchInfo, OpenPullRequestRefs

MAIN = "1" * 40
ARCHIVE = module.ARCHIVE_HEAD


def _branch(name: str, sha: str, protected: bool = False) -> BranchInfo:
    return BranchInfo(name, sha, protected)


def _fake(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, tuple[BranchInfo, ...]]:
    main = _branch("main", MAIN)
    archive = _branch(module.ARCHIVE_BRANCH, ARCHIVE)
    candidates = tuple(
        _branch(name, sha) for name, sha in module.FROZEN_SOURCES.items()
    )
    refs = (main, archive, *candidates)
    monkeypatch.setattr(
        module, "_archive_manifest", lambda: {b.name: b.sha for b in refs}
    )
    monkeypatch.setattr(module, "_git_ancestor", lambda *args: True)
    monkeypatch.setattr(module, "_git_main", lambda: MAIN)
    monkeypatch.setattr(
        module, "reachable_commits", lambda *args: frozenset(b.sha for b in refs)
    )

    class FakeAPI:
        _repository = module.REPOSITORY

        def __init__(self) -> None:
            self.refs = list(refs)
            self.open: tuple[OpenPullRequestRefs, ...] = ()
            self.running: frozenset[str] = frozenset()
            self.deleted: list[tuple[BranchInfo, ...]] = []

        def repository_default_branch(self) -> str:
            return "main"

        def branches(self) -> tuple[BranchInfo, ...]:
            return tuple(self.refs)

        def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
            return self.open

        def active_workflow_branch_names(self) -> frozenset[str]:
            return self.running

        def delete_branches_with_leases(self, batch: tuple[BranchInfo, ...]) -> None:
            self.deleted.append(batch)
            self.refs = [b for b in self.refs if b not in batch]

    return FakeAPI(), refs


def test_fixed_explicit_allowlist_is_only_six_ordinary_fix_refs() -> None:
    assert len(module.FROZEN_SOURCES) == 6
    assert all(name.startswith("fix/") for name in module.FROZEN_SOURCES)
    assert all(not name.startswith("fix/m1-") for name in module.FROZEN_SOURCES)
    assert len(set(module.FROZEN_SOURCES.values())) == 6
    assert module.ARCHIVE_HEAD == "2be720a5ef837462e4132ebecf982d8929112b6c"


def test_main_ancestor_with_archive_but_no_allowlist_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, refs = _fake(monkeypatch)
    candidate = module._select(api, refs)
    assert set(candidate) == set(refs[2:])
    additional = _branch("fix/other-main-ancestor", "e" * 40)
    api.refs.append(additional)
    assert set(module._select(api, (*refs, additional))) == set(refs[2:])


def test_archive_sha_drift_main_history_open_pr_and_active_ci_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, refs = _fake(monkeypatch)
    a = refs[2]
    api.open = (OpenPullRequestRefs(a.name, a.sha, "main", MAIN),)
    api.running = frozenset({refs[3].name})
    assert a not in module._select(api, refs)
    assert refs[3] not in module._select(api, refs)

    api.open = ()
    api.running = frozenset()
    wrong = _branch(a.name, "a" * 40)
    assert wrong not in module._select(api, (refs[0], refs[1], wrong, *refs[3:]))
    protected = _branch(a.name, a.sha, True)
    assert protected not in module._select(
        api, (refs[0], refs[1], protected, *refs[3:])
    )

    monkeypatch.setattr(module, "reachable_commits", lambda *args: frozenset({MAIN}))
    assert module._select(api, refs) == ()
    monkeypatch.setattr(
        module, "reachable_commits", lambda *args: frozenset(b.sha for b in refs)
    )
    monkeypatch.setattr(module, "_git_ancestor", lambda *args: False)
    with pytest.raises(RuntimeError, match="archive"):
        module._select(api, refs)


def test_refuses_missing_archive_or_inconsistent_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, refs = _fake(monkeypatch)
    with pytest.raises(RuntimeError, match="archive"):
        module._select(api, (refs[0], *refs[2:]))
    monkeypatch.setattr(module, "_git_main", lambda: "f" * 40)
    with pytest.raises(RuntimeError, match="main"):
        module._select(api, refs)


def _owner_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    event = {
        "action": "created",
        "issue": {"number": 1057},
        "comment": {
            "user": {"id": 148511409, "login": "shuntatsu"},
            "body": module.COMMAND,
        },
    }
    payload = tmp_path / "event.json"
    payload.write_text(json.dumps(event), encoding="utf-8")
    for key, value in (
        ("GITHUB_ACTIONS", "true"),
        ("GITHUB_REPOSITORY", module.REPOSITORY),
        ("GITHUB_ACTOR", "shuntatsu"),
        ("GITHUB_EVENT_NAME", "issue_comment"),
        ("GITHUB_EVENT_PATH", str(payload)),
        ("GITHUB_SHA", MAIN),
    ):
        monkeypatch.setenv(key, value)


def test_requires_created_exact_owner_comment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _owner_event(monkeypatch, tmp_path)
    module._require_owner_event()
    filename = Path(monkeypatch._setitem[-1][1]) if False else tmp_path / "event.json"
    event = json.loads(filename.read_text(encoding="utf-8"))
    event["action"] = "edited"
    filename.write_text(json.dumps(event), encoding="utf-8")
    with pytest.raises(RuntimeError, match="owner"):
        module._require_owner_event()
    event["action"] = "created"
    event["comment"]["body"] += "\n"
    filename.write_text(json.dumps(event), encoding="utf-8")
    with pytest.raises(RuntimeError, match="owner"):
        module._require_owner_event()


def test_preview_and_successful_exact_sha_lease_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api, refs = _fake(monkeypatch)
    before, selected, deleted = module.retire(api, apply=False)
    assert before == len(refs)
    assert len(selected) == 6
    assert deleted == () and not api.deleted
    _owner_event(monkeypatch, tmp_path)
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
    before, selected, deleted = module.retire(api, apply=True)
    assert before == len(refs)
    assert len(selected) == len(deleted) == 6
    assert list(map(len, api.deleted)) == [3, 3]
    assert api.refs == [refs[0], refs[1]]


def test_existing_ref_drift_before_batch_aborts_without_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api, refs = _fake(monkeypatch)
    _owner_event(monkeypatch, tmp_path)
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
    original = api.open_pull_request_refs
    count = 0

    def changed() -> tuple[OpenPullRequestRefs, ...]:
        nonlocal count
        count += 1
        if count >= 2:
            return (
                OpenPullRequestRefs(
                    sorted(module.FROZEN_SOURCES)[0],
                    module.FROZEN_SOURCES[sorted(module.FROZEN_SOURCES)[0]],
                    "main",
                    MAIN,
                ),
            )
        return original()

    api.open_pull_request_refs = changed
    with pytest.raises(RuntimeError, match="eligibility"):
        module.retire(api, apply=True)
    assert api.deleted == []


def test_workflow_gate_has_exact_created_comment() -> None:
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/retire-merged-pr-branches.yml"
    ).read_text(encoding="utf-8")
    assert "frozen-orphan-ancestor-retirement:" in workflow
    assert "github.event.action == 'created'" in workflow
    assert module.COMMAND in workflow
    token = "$" + "{{ github.token }}"
    assert workflow.count("GH_TOKEN: " + token) == 14
