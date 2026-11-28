"""Fail-closed direct Task phase and malformed JSON projection boundaries."""

from __future__ import annotations

import pytest

from tools.agent_repo.coordination.github_state import (
    TaskStatusRecord,
    parse_status_comment,
    render_status_comment,
)
from tools.agent_repo.coordination.model import TaskCondition, TaskPhase
from tools.agent_repo.coordination.registry import (
    CoordinatorRegistry,
    parse_registry_bytes,
)
from tools.agent_repo.coordination.state import validate_transition


def _deep_json(*, nested: bool) -> str:
    # Bounded literal input, intentionally deeper than CPython's JSON parser.
    value = "[" * 20_000 + "0" + "]" * 20_000
    return '{"value":' + value + "}" if nested else value


@pytest.mark.parametrize("nested", [False, True])
def test_registry_deep_json_is_a_controlled_value_error(nested: bool) -> None:
    data = _deep_json(nested=nested).encode("utf-8")
    with pytest.raises(ValueError, match="registry JSON nesting"):
        parse_registry_bytes(data)


@pytest.mark.parametrize("nested", [False, True])
def test_status_comment_deep_json_is_a_controlled_value_error(nested: bool) -> None:
    fence = chr(96) * 3
    body = "<!-- agent-coordination:T500-04 -->\n" + fence + "json\n"
    body += _deep_json(nested=nested) + "\n" + fence
    with pytest.raises(ValueError, match="status comment JSON nesting"):
        parse_status_comment(body)


def test_registry_and_status_valid_controls_preserved() -> None:
    registry = CoordinatorRegistry("shuntatsu/trade_rl")
    assert parse_registry_bytes(registry.canonical_bytes()) == registry
    record = TaskStatusRecord(
        task_id="T500-04",
        task_revision=1,
        task_contract_digest="c" * 64,
        phase=TaskPhase.PLANNED,
        condition=TaskCondition.HEALTHY,
        base_sha="a" * 40,
    )
    assert parse_status_comment(render_status_comment(record)) == record


@pytest.mark.parametrize("which", ["current", "target"])
def test_foreign_phase_must_not_run_lookup_hooks(which: str) -> None:
    hooks: list[str] = []

    class ForeignPhase(str):
        def __hash__(self) -> int:
            hooks.append("hash")
            return hash("ready")

        def __eq__(self, other: object) -> bool:
            hooks.append("eq")
            return other == "ready"

    if which == "current":
        values = (ForeignPhase("complete"), TaskPhase.EXECUTING)
    else:
        values = (TaskPhase.PLANNED, ForeignPhase("complete"))
    with pytest.raises(ValueError, match="native str or TaskPhase"):
        validate_transition(*values)
    assert hooks == []


@pytest.mark.parametrize("which", ["current", "target"])
def test_foreign_phase_hash_trap_is_not_executed(which: str) -> None:
    class ForeignPhase(str):
        def __hash__(self) -> int:
            raise AssertionError("unsafe hash hook was invoked")

    values = (
        (ForeignPhase("ready"), TaskPhase.EXECUTING)
        if which == "current"
        else (TaskPhase.PLANNED, ForeignPhase("ready"))
    )
    with pytest.raises(ValueError, match="native str or TaskPhase"):
        validate_transition(*values)


_ALLOWED = {
    TaskPhase.PLANNED: {TaskPhase.READY},
    TaskPhase.READY: {TaskPhase.EXECUTING},
    TaskPhase.EXECUTING: {TaskPhase.REVIEW},
    TaskPhase.REVIEW: {TaskPhase.EXECUTING, TaskPhase.VERIFICATION},
    TaskPhase.VERIFICATION: {
        TaskPhase.EXECUTING,
        TaskPhase.INTEGRATION,
        TaskPhase.COMPLETE,
    },
    TaskPhase.INTEGRATION: {
        TaskPhase.EXECUTING,
        TaskPhase.VERIFICATION,
        TaskPhase.COMPLETE,
    },
    TaskPhase.COMPLETE: set(),
}


@pytest.mark.parametrize("current", list(TaskPhase))
@pytest.mark.parametrize("target", list(TaskPhase))
@pytest.mark.parametrize("as_text", [False, True])
def test_native_transition_matrix_unchanged(
    current: TaskPhase, target: TaskPhase, as_text: bool
) -> None:
    before = current.value if as_text else current
    after = target.value if as_text else target
    if target in _ALLOWED[current]:
        assert validate_transition(before, after) is None
    else:
        with pytest.raises(ValueError):
            validate_transition(before, after)


@pytest.mark.parametrize("which", ["current", "target"])
@pytest.mark.parametrize("spoof", [False, True])
def test_foreign_phase_metaclass_cannot_run_equality_or_spoof_type(
    which: str, spoof: bool
) -> None:
    hooks: list[str] = []

    class ForeignMeta(type):
        __hash__ = type.__hash__

        def __eq__(self, other: object) -> bool:
            hooks.append("metaclass equality")
            if not spoof:
                raise AssertionError("foreign metaclass equality invoked")
            return True

    class ForeignPhase(str, metaclass=ForeignMeta):
        def __hash__(self) -> int:
            hooks.append("value hash")
            return hash("ready")

        def __eq__(self, other: object) -> bool:
            hooks.append("value equality")
            return other == "ready"

    values = (
        (ForeignPhase("complete"), TaskPhase.EXECUTING)
        if which == "current"
        else (TaskPhase.PLANNED, ForeignPhase("complete"))
    )
    with pytest.raises(ValueError, match="native str or TaskPhase"):
        validate_transition(*values)
    assert hooks == []
