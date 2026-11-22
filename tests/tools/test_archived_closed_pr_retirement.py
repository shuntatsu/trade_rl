"""Synthetic fail-closed regression suite for owner-requested frozen PR source cleanup."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools import archived_closed_pr_retirement as mod
from tools.branch_hygiene import BranchInfo, OpenPullRequestRefs

MAIN = "1" * 40
ARCHIVE = mod.ARCHIVE_HEAD


def _ref(name: str, sha: str, protected: bool = False) -> BranchInfo:
    return BranchInfo(name, sha, protected)


def _fixture(
    monkeypatch: pytest.MonkeyPatch, count: int = 28
) -> tuple[Any, tuple[BranchInfo, ...], dict[str, str]]:
    assert count <= len(mod.CLOSED_PR_ALLOWLIST)
    main, archive = _ref("main", MAIN), _ref(mod.ARCHIVE_BRANCH, ARCHIVE)
    candidates = tuple(
        _ref(name, f"{i + 2:040x}")
        for i, name in enumerate(sorted(mod.CLOSED_PR_ALLOWLIST)[:count])
    )
    all_refs = (main, archive, *candidates)
    frozen = {b.name: b.sha for b in all_refs}
    monkeypatch.setattr(mod, "_archive_manifest", lambda: frozen)
    monkeypatch.setattr(mod, "_git_main", lambda: MAIN)
    monkeypatch.setattr(mod, "_git_ancestor", lambda *args: True)

    class Api:
        _repository = mod.REPOSITORY

        def __init__(self) -> None:
            self.refs = list(all_refs)
            self.open: tuple[OpenPullRequestRefs, ...] = ()
            self.running: frozenset[str] = frozenset()
            self.states: dict[int, str] = {}
            self.bad_heads: set[int] = set()
            self.deleted: list[tuple[BranchInfo, ...]] = []

        def repository_default_branch(self) -> str:
            return "main"

        def branches(self) -> tuple[BranchInfo, ...]:
            return tuple(self.refs)

        def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
            return self.open

        def active_workflow_branch_names(self) -> frozenset[str]:
            return self.running

        def _request_json(self, method: str, suffix: str) -> dict[str, Any]:
            assert method == "GET"
            number = int(suffix.split("/")[-1])
            name = next(n for n, i in mod.CLOSED_PR_ALLOWLIST.items() if i == number)
            source = next(b for b in all_refs if b.name == name)
            return {
                "number": number,
                "state": self.states.get(number, "closed"),
                "closed_at": "2026-11-01T12:00:00Z",
                "head": {
                    "ref": name,
                    "sha": "f" * 40 if number in self.bad_heads else source.sha,
                    "repo": {"full_name": mod.REPOSITORY},
                },
                "base": {"repo": {"full_name": mod.REPOSITORY}},
            }

        def delete_branches_with_leases(self, batch: tuple[BranchInfo, ...]) -> None:
            self.deleted.append(batch)
            self.refs = [r for r in self.refs if r not in batch]

    return Api(), all_refs, frozen


def test_strict_whitelist_excludes_research_and_m1() -> None:
    assert len(mod.CLOSED_PR_ALLOWLIST) == 28
    assert len(set(mod.CLOSED_PR_ALLOWLIST.values())) == 28
    assert sum(n.startswith("codex/") for n in mod.CLOSED_PR_ALLOWLIST) == 22
    assert sum(n.startswith("review/") for n in mod.CLOSED_PR_ALLOWLIST) == 4
    for name in mod.CLOSED_PR_ALLOWLIST:
        assert not name.startswith(
            ("research/", "audit/", "infra/", "provenance/", "archive/", "fix/m1-")
        )
    assert mod.ARCHIVE_HEAD == "2be720a5ef837462e4132ebecf982d8929112b6c"


def test_exact_closed_source_and_reopened_or_moved_pr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, branches, frozen = _fixture(monkeypatch, 3)
    names = [b.name for b in branches[2:]]
    first = mod.CLOSED_PR_ALLOWLIST[names[0]]
    second = mod.CLOSED_PR_ALLOWLIST[names[1]]
    assert (
        len(mod._select(api, branches, frozen=frozen, open_refs=(), active=frozenset()))
        == 3
    )

    api.states[first] = "open"
    api.bad_heads.add(second)
    result = mod._select(api, branches, frozen=frozen, open_refs=(), active=frozenset())
    assert result == (branches[4],)

    api.states[first] = "closed"
    api.bad_heads.clear()
    api.open = (OpenPullRequestRefs(names[0], branches[2].sha, "main", MAIN),)
    api.running = frozenset({names[1]})
    result = mod._select(
        api, branches, frozen=frozen, open_refs=api.open, active=api.running
    )
    assert result == (branches[4],)


def test_snapshot_drift_protected_and_archive_missing_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, branches, frozen = _fixture(monkeypatch, 2)
    ref = branches[2]
    modified = (branches[0], branches[1], _ref(ref.name, "f" * 40), branches[3])
    assert mod._select(
        api, modified, frozen=frozen, open_refs=(), active=frozenset()
    ) == (branches[3],)
    protected = (branches[0], branches[1], _ref(ref.name, ref.sha, True), branches[3])
    assert mod._select(
        api, protected, frozen=frozen, open_refs=(), active=frozenset()
    ) == (branches[3],)
    with pytest.raises(RuntimeError, match="archive"):
        mod._select(
            api,
            (branches[0], *branches[2:]),
            frozen=frozen,
            open_refs=(),
            active=frozenset(),
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        mod._select(
            api,
            (*branches, branches[2]),
            frozen=frozen,
            open_refs=(),
            active=frozenset(),
        )
    monkeypatch.setattr(mod, "_git_ancestor", lambda *args: False)
    with pytest.raises(RuntimeError, match="archive does not reach"):
        mod._select(api, branches, frozen=frozen, open_refs=(), active=frozenset())


def test_one_shot_event_requires_exact_owner_and_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = {
        "issue": {"number": 1057},
        "comment": {
            "user": {"login": "shuntatsu", "id": 148511409},
            "body": mod.COMMAND,
        },
    }
    f = tmp_path / "event.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    for key, value in (
        ("GITHUB_ACTIONS", "true"),
        ("GITHUB_REPOSITORY", mod.REPOSITORY),
        ("GITHUB_ACTOR", "shuntatsu"),
        ("GITHUB_EVENT_NAME", "issue_comment"),
        ("GITHUB_EVENT_PATH", str(f)),
    ):
        monkeypatch.setenv(key, value)
    mod._owner_comment_event()
    data["comment"]["body"] += "\nother"
    f.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing exact"):
        mod._owner_comment_event()
    data["comment"]["body"] = mod.COMMAND
    f.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("GITHUB_ACTOR", "someone-else")
    with pytest.raises(RuntimeError, match="only the owner"):
        mod._owner_comment_event()


def _grant_owner_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evt = tmp_path / "event.json"
    evt.write_text(
        json.dumps(
            {
                "issue": {"number": 1057},
                "comment": {
                    "body": mod.COMMAND,
                    "user": {"login": "shuntatsu", "id": 148511409},
                },
            }
        ),
        encoding="utf-8",
    )
    for key, val in (
        ("GITHUB_ACTIONS", "true"),
        ("GITHUB_REPOSITORY", mod.REPOSITORY),
        ("GITHUB_ACTOR", "shuntatsu"),
        ("GITHUB_EVENT_NAME", "issue_comment"),
        ("GITHUB_EVENT_PATH", str(evt)),
        ("GITHUB_SHA", MAIN),
    ):
        monkeypatch.setenv(key, val)
    rep = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(rep)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(rep),
            "remote",
            "add",
            "origin",
            "https://github.com/shuntatsu/trade_rl.git",
        ],
        check=True,
    )
    monkeypatch.chdir(rep)


def test_apply_checks_actual_origin_gate_and_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api, branches, _ = _fixture(monkeypatch, 15)
    n, c, d = mod.retire(api, apply=False)
    assert n == 17 and len(c) == 15 and d == () and api.deleted == []
    _grant_owner_event(monkeypatch, tmp_path)
    n, c, d = mod.retire(api, apply=True)
    assert n == 17 and len(c) == len(d) == 15
    assert list(map(len, api.deleted)) == [7, 7, 1]
    assert api.refs == [branches[0], branches[1]]


def test_apply_stops_when_pr_reopens_before_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api, branches, _ = _fixture(monkeypatch, 2)
    _grant_owner_event(monkeypatch, tmp_path)
    first = mod.CLOSED_PR_ALLOWLIST[branches[2].name]
    old = api._request_json
    calls = 0

    def reopen(method: str, suffix: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls >= 3:
            api.states[first] = "open"
        return old(method, suffix)

    api._request_json = reopen
    with pytest.raises(RuntimeError, match="eligibility changed"):
        mod.retire(api, apply=True)
    assert api.deleted == []


def test_workflow_is_gated_to_exact_issue_comment_token() -> None:
    p = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/retire-merged-pr-branches.yml"
    )
    workflow = p.read_text(encoding="utf-8")
    token = "$" + "{{ github.token }}"
    assert workflow.count("GH_TOKEN: " + token) == 16
    assert "frozen-closed-pr-retirement:" in workflow
    assert mod.COMMAND in workflow
    assert "ref: main" in workflow
    assert "persist-credentials: false" in workflow
