"""Fail-closed, read-only cross-check of unfinished workflow branch references.

The supplied reader must fully enumerate requested/waiting/pending/queued/in_progress
runs and reject incomplete pages itself. Two equal branch-name observations are
only a drift check, not an atomic snapshot or a complete Task/artifact roster.
"""

from __future__ import annotations

from typing import Protocol, cast


class WorkflowBranchReader(Protocol):
    def unfinished_workflow_branch_names(self) -> frozenset[str]: ...


def _valid_branch_name(value: object) -> bool:
    """Accept only Unicode scalar Git branch names under refs/heads/."""
    if type(value) is not str or not value or value.endswith("."):
        return False
    if ".." in value or "@{" in value:
        return False
    for char in value:
        codepoint = ord(char)
        if (
            codepoint < 32
            or codepoint == 127
            or 0xD800 <= codepoint <= 0xDFFF
            or char in " ~^:?*[\\"
        ):
            return False
    for component in value.split("/"):
        if not component or component.startswith(".") or component.endswith(".lock"):
            return False
    return True


def _validated_branch_names(value: object, *, invalid_message: str) -> frozenset[str]:
    if type(value) is not frozenset:
        raise ValueError(invalid_message)
    names = cast(frozenset[str], value)
    for name in names:
        if not _valid_branch_name(name):
            raise ValueError(invalid_message)
    for name in names:
        components = name.split("/")
        for index in range(1, len(components)):
            if "/".join(components[:index]) in names:
                raise ValueError(invalid_message)
    return names


def stable_workflow_branch_refs(
    reader: WorkflowBranchReader, *, known_branch_names: frozenset[str]
) -> tuple[str, ...]:
    """Reject incomplete, changing, or orphaned active workflow branch refs.

    This never authorizes lease reassignment, Task enrollment or branch deletion.
    The caller must independently verify the authenticated repository, PR and
    Task roster, branch SHA, workflow attempts/jobs and artifacts.
    """
    known = _validated_branch_names(
        known_branch_names, invalid_message="known branch roster is malformed"
    )
    first = _validated_branch_names(
        reader.unfinished_workflow_branch_names(),
        invalid_message="unfinished workflow branch refs are malformed",
    )
    if first - known:
        raise RuntimeError("unfinished workflow refs have missing local branches")
    second = _validated_branch_names(
        reader.unfinished_workflow_branch_names(),
        invalid_message="unfinished workflow branch refs are malformed",
    )
    if first != second:
        raise RuntimeError("unfinished workflow refs changed during observation")
    return tuple(sorted(first))


__all__ = ["stable_workflow_branch_refs"]
