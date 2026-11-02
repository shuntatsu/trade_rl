"""Fail-closed, read-only cross-check of unfinished workflow branch references.

The supplied reader must fully enumerate requested/waiting/pending/queued/in_progress
runs and reject incomplete pages itself. Two equal branch-name observations are
only a drift check, not an atomic snapshot or a complete Task/artifact roster.
"""

from __future__ import annotations

from typing import Protocol


class WorkflowBranchReader(Protocol):
    def unfinished_workflow_branch_names(self) -> frozenset[str]: ...


def stable_workflow_branch_refs(
    reader: WorkflowBranchReader, *, known_branch_names: frozenset[str]
) -> tuple[str, ...]:
    """Reject incomplete, changing, or orphaned active workflow branch refs.

    This never authorizes lease reassignment, Task enrollment or branch deletion.
    The caller must independently verify the authenticated repository, PR and
    Task roster, branch SHA, workflow attempts/jobs and artifacts.
    """
    if type(known_branch_names) is not frozenset or any(
        type(name) is not str or not name.strip() for name in known_branch_names
    ):
        raise ValueError("known branch roster is malformed")

    first = reader.unfinished_workflow_branch_names()
    second = reader.unfinished_workflow_branch_names()
    for result in (first, second):
        if type(result) is not frozenset or any(
            type(name) is not str or not name.strip() for name in result
        ):
            raise ValueError("unfinished workflow branch refs are malformed")
    if first != second:
        raise RuntimeError("unfinished workflow refs changed during observation")
    missing = first - known_branch_names
    if missing:
        raise RuntimeError("unfinished workflow refs have missing local branches")
    return tuple(sorted(first))


__all__ = ["stable_workflow_branch_refs"]
