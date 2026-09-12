"""Semantic and side-effect conflict classification for coordinated tasks."""

from __future__ import annotations

from enum import Enum

from tools.agent_repo.coordination.model import ExecutionMode, TaskPacket

_ALLOWED_NAMESPACES = {
    "file",
    "authority",
    "identity",
    "schema",
    "workflow",
    "artifact",
    "side-effect",
}
_HARD_NAMESPACES = {"identity", "schema", "side-effect"}
_SOFT_NAMESPACES = {"authority", "workflow", "artifact"}


class ConflictLevel(str, Enum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"


def _split_key(key: str) -> tuple[str, str]:
    namespace, separator, value = key.partition(":")
    if not separator or namespace not in _ALLOWED_NAMESPACES or not value:
        raise ValueError(f"unknown resource namespace: {key}")
    return namespace, value


def _file_prefix(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    if not normalized or ".." in normalized.split("/"):
        raise ValueError(f"invalid file resource key: {value}")
    for suffix in ("/**", "/*"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip("/")
    return normalized


def _file_overlap(left: str, right: str) -> bool:
    left_prefix = _file_prefix(left)
    right_prefix = _file_prefix(right)
    return (
        left_prefix == right_prefix
        or left_prefix.startswith(f"{right_prefix}/")
        or right_prefix.startswith(f"{left_prefix}/")
    )


def classify_conflict(left: TaskPacket, right: TaskPacket) -> ConflictLevel:
    """Classify whether two tasks can safely execute concurrently."""

    left_keys = [_split_key(key) for key in left.resource_keys]
    right_keys = [_split_key(key) for key in right.resource_keys]

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
