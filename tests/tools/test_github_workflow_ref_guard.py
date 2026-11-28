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
    known_branches = frozenset({*names, "provenance/branch-retention"})
    actual = stable_workflow_branch_refs(reader, known_branch_names=known_branches)
    assert actual == ("agent/T810/e0002-owner", "main")
    assert reader.calls == 2


def test_empty_active_roster_requires_two_reads() -> None:
    reader = FakeReader(frozenset(), frozenset())
    known_branches = frozenset({"main"})
    actual = stable_workflow_branch_refs(reader, known_branch_names=known_branches)
    assert actual == ()
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
    known_branches: frozenset[str] = bad  # type: ignore[assignment]
    with pytest.raises(ValueError, match="known branch"):
        stable_workflow_branch_refs(reader, known_branch_names=known_branches)


def test_failed_api_page_read_does_not_return_partial_roster() -> None:
    reader = FakeReader(frozenset({"main"}), RuntimeError("GitHub API 503"))
    with pytest.raises(RuntimeError, match="503"):
        stable_workflow_branch_refs(reader, known_branch_names=frozenset({"main"}))


_INVALID_NAMES = (
    "",
    " ",
    "topic/name ",
    "topic name",
    "topic\tname",
    "topic\nname",
    "topic\x00name",
    "topic\x7fname",
    "topic..child",
    "topic@{child",
    "topic/.hidden",
    ".hidden",
    "topic.lock",
    "topic/child.lock",
    "topic/child.",
    "topic.",
    "topic//child",
    "/topic",
    "topic/",
    "topic~child",
    "topic^child",
    "topic:child",
    "topic?child",
    "topic*child",
    "topic[child",
    "topic\\child",
    "topic\ud800",
    "topic\udfff",
)


@pytest.mark.parametrize("name", _INVALID_NAMES)
@pytest.mark.parametrize("where", ["known", "first", "second"])
def test_github_ref_grammar_is_validated_at_each_untrusted_roster_boundary(
    name: str, where: str
) -> None:
    safe = frozenset({"main"})
    with_bad = frozenset({"main", name})
    if where == "known":
        reader = FakeReader(safe, safe)
        with pytest.raises(ValueError, match="known branch roster is malformed"):
            stable_workflow_branch_refs(reader, known_branch_names=with_bad)
        assert reader.calls == 0
    elif where == "first":
        reader = FakeReader(with_bad, RuntimeError("unnecessary second page"))
        with pytest.raises(
            ValueError, match="unfinished workflow branch refs are malformed"
        ):
            stable_workflow_branch_refs(reader, known_branch_names=safe)
        assert reader.calls == 1
    else:
        reader = FakeReader(safe, with_bad)
        with pytest.raises(
            ValueError, match="unfinished workflow branch refs are malformed"
        ):
            stable_workflow_branch_refs(reader, known_branch_names=safe)
        assert reader.calls == 2


@pytest.mark.parametrize(
    "names",
    [
        frozenset({"topic", "topic/child"}),
        frozenset({"topic/child", "topic"}),
        frozenset({"a/b", "a/b/c", "a/other"}),
    ],
)
def test_impossible_git_ref_namespace_is_rejected_before_read(
    names: frozenset[str],
) -> None:
    reader = FakeReader(frozenset(), frozenset())
    with pytest.raises(ValueError, match="known branch roster is malformed"):
        stable_workflow_branch_refs(reader, known_branch_names=names)
    assert reader.calls == 0


def test_similar_git_ref_names_are_not_false_prefix_collisions() -> None:
    names = frozenset({"topic-other", "topic/child", "topic/child2", "main"})
    reader = FakeReader(names, names)
    assert stable_workflow_branch_refs(reader, known_branch_names=names) == tuple(
        sorted(names)
    )
    assert reader.calls == 2


@pytest.mark.parametrize("second", [RuntimeError("GitHub 503"), frozenset()])
def test_initial_orphan_reference_fails_before_second_read(second: object) -> None:
    reader = FakeReader(frozenset({"missing"}), second)
    with pytest.raises(RuntimeError, match="missing local branches"):
        stable_workflow_branch_refs(reader, known_branch_names=frozenset({"main"}))
    assert reader.calls == 1


def test_unicode_scalar_supplementary_branch_remains_legal() -> None:
    name = "feature/\U00020000"
    reader = FakeReader(frozenset({name}), frozenset({name}))
    assert stable_workflow_branch_refs(
        reader, known_branch_names=frozenset({"main", name})
    ) == (name,)
    assert reader.calls == 2


def test_malformed_first_observation_masks_no_following_network_failure() -> None:
    reader = FakeReader(frozenset({"bad..name"}), RuntimeError("network refused"))
    with pytest.raises(
        ValueError, match="unfinished workflow branch refs are malformed"
    ):
        stable_workflow_branch_refs(reader, known_branch_names=frozenset({"main"}))
    assert reader.calls == 1
