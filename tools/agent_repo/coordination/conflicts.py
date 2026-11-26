"""Semantic and side-effect conflict classification for coordinated tasks."""

from __future__ import annotations

from enum import Enum

from tools.agent_repo.coordination.model import (
    ExecutionMode,
    TaskPacket,
    resource_key_parts,
)

_HARD_NAMESPACES = {"identity", "schema", "side-effect"}
_SOFT_NAMESPACES = {"authority", "workflow", "artifact"}


class ConflictLevel(str, Enum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"


def _file_overlap(left: str, right: str) -> bool:
    # An empty normalized prefix denotes the repository root, not a literal
    # star-named directory. A root resource conflicts with every file resource.
    return (
        not left
        or not right
        or left == right
        or left.startswith(f"{right}/")
        or right.startswith(f"{left}/")
    )


def classify_conflict(left: TaskPacket, right: TaskPacket) -> ConflictLevel:
    """Classify whether two tasks can safely execute concurrently."""

    left_keys = [resource_key_parts(key) for key in left.resource_keys]
    right_keys = [resource_key_parts(key) for key in right.resource_keys]

    shared_side_effect = any(
        left_namespace == right_namespace == "side-effect" and left_value == right_value
        for left_namespace, left_value in left_keys
        for right_namespace, right_value in right_keys
    )
    if (
        left.execution_mode is ExecutionMode.READ_ONLY
        and right.execution_mode is ExecutionMode.READ_ONLY
        and not shared_side_effect
    ):
        return ConflictLevel.NONE

    level = ConflictLevel.NONE
    for left_namespace, left_value in left_keys:
        for right_namespace, right_value in right_keys:
            if left_namespace != right_namespace:
                continue
            if left_namespace == "file":
                if _file_overlap(left_value, right_value):
                    return ConflictLevel.HARD
                continue
            if left_value != right_value:
                continue
            if left_namespace in _HARD_NAMESPACES:
                return ConflictLevel.HARD
            if left_namespace in _SOFT_NAMESPACES:
                level = ConflictLevel.SOFT
    return level


__all__ = ["ConflictLevel", "classify_conflict"]
