"""Refusal oracles for untrusted active-workflow branch observations."""

from __future__ import annotations

import pytest

from tools.github_workflow_ref_guard import stable_workflow_branch_refs


class FakeReader:
    def __init__(self, *responses: object) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def unfinished_workflow_branch_names(self) -> frozenset[str]:
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response  # type: ignore[return-value]


def test_stable_branches_include_active_and_do_not_claim_authority() -> None:
    names = frozenset({"main", "agent/T810/e0002-owner"})
    reader = FakeReader(names, names)
    assert stable_workflow_branch_refs(
        reader, known_branch_names=frozenset({*names, "provenance/branch-retention"})
    ) == ("agent/T810/e0002-owner", "main")
    assert reader.calls == 2


def test_empty_active_roster_requires_two_reads() -> None:
    reader = FakeReader(frozenset(), frozenset())
    assert stable_workflow_branch_refs(
        reader, known_branch_names=frozenset({"main"})
    ) == ()
    assert reader.calls == 2


def test_changed_status_roster_is_not_stable() -> None:
    reader = FakeReader(frozenset({"main"}), frozenset())
    with pytest.raises(RuntimeError, match="changed"):
        stable_workflow_branch_refs(reader, known_branch_names=frozenset({"main"}))


def test_deleted_branch_referenced_by_pending_run_fails_closed() -> None:
    reader = FakeReader(frozenset({"missing"}), frozenset({"missing"}))
    with pytest.raises(RuntimeError, match="missing local branches"):
        stable_workflow_branch_refs(reader, known_branch_names=frozenset({"main"}))


@pytest.mark.parametrize(
    "value",
    (None, {}, [], ["main"], frozenset({""}), frozenset({None})),
)
def test_unknown_or_malformed_workflow_observation_is_not_empty(value: object) -> None:
    reader = FakeReader(value, value)
    with pytest.raises(ValueError, match="malformed"):
        stable_workflow_branch_refs(reader, known_branch_names=frozenset({"main"}))


@pytest.mark.parametrize("bad", ({"main"}, frozenset({None}), frozenset({" "})))
def test_invalid_known_branch_roster_fails_closed(bad: object) -> None:
    reader = FakeReader(frozenset(), frozenset())
    with pytest.raises(ValueError, match="known branch"):
        stable_workflow_branch_refs(
            reader, known_branch_names=bad  # type: ignore[arg-type]
        )


def test_failed_api_page_read_does_not_return_partial_roster() -> None:
    reader = FakeReader(frozenset({"main"}), RuntimeError("GitHub API 503"))
    with pytest.raises(RuntimeError, match="503"):
        stable_workflow_branch_refs(reader, known_branch_names=frozenset({"main"}))
