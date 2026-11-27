"""Parent completion requires affirmative, native boolean oracle results."""

from typing import Any

import pytest

from tools.agent_repo.coordination.model import TaskPhase, TaskStatus
from tools.agent_repo.coordination.state import parent_completion_allowed


@pytest.mark.parametrize("value", ["false", 1, [False], {"passed": False}])
@pytest.mark.parametrize("field", ["acceptance", "invariants"])
def test_non_boolean_oracle_cannot_allow_completion(value: Any, field: str) -> None:
    assert (
        parent_completion_allowed(
            {"child": TaskStatus(TaskPhase.COMPLETE)},
            parent_acceptance_satisfied=value if field == "acceptance" else True,
            parent_invariants_satisfied=value if field == "invariants" else True,
        )
        is False
    )


def test_parent_completion_preserves_boolean_controls() -> None:
    children = {"child": TaskStatus(TaskPhase.COMPLETE)}
    for acceptance in (False, True):
        for invariants in (False, True):
            assert parent_completion_allowed(
                children,
                parent_acceptance_satisfied=acceptance,
                parent_invariants_satisfied=invariants,
            ) is (acceptance and invariants)


def test_oracle_objects_are_not_coerced_to_bool_or_compared() -> None:
    class Impostor:
        def __bool__(self) -> bool:
            raise AssertionError("must not coerce oracle")

        def __eq__(self, other: object) -> bool:
            raise AssertionError("must not compare oracle")

    value: Any = Impostor()
    children = {"child": TaskStatus(TaskPhase.COMPLETE)}
    for acceptance, invariants in ((value, True), (True, value)):
        assert (
            parent_completion_allowed(
                children,
                parent_acceptance_satisfied=acceptance,
                parent_invariants_satisfied=invariants,
            )
            is False
        )


class _HiddenChildren(dict):
    def values(self):
        return ()


class _ForgedCompleteChildren(dict):
    def values(self):
        return (TaskStatus(TaskPhase.COMPLETE),)


class _TrapChildren(dict):
    def values(self):
        pytest.fail("must not enumerate a foreign child roster")


@pytest.mark.parametrize(
    "roster",
    [
        _HiddenChildren(child=TaskStatus(TaskPhase.VERIFICATION)),
        _ForgedCompleteChildren(child=TaskStatus(TaskPhase.VERIFICATION)),
        _TrapChildren(child=TaskStatus(TaskPhase.VERIFICATION)),
    ],
)
def test_parent_completion_rejects_foreign_child_roster(roster):
    assert not parent_completion_allowed(
        roster, parent_acceptance_satisfied=True, parent_invariants_satisfied=True
    )


def test_parent_completion_rejects_foreign_child_status():
    class ForgedStatus:
        @property
        def phase(self):
            return TaskPhase.COMPLETE

        @property
        def condition(self):
            from tools.agent_repo.coordination.model import TaskCondition

            return TaskCondition.HEALTHY

    assert not parent_completion_allowed(
        {"child": ForgedStatus()},
        parent_acceptance_satisfied=True,
        parent_invariants_satisfied=True,
    )
