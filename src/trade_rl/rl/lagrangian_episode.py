"""Episode completion semantics for constraint statistics."""

from __future__ import annotations

from enum import IntEnum


class EpisodeCompletionKind(IntEnum):
    """How one transition affects completed-episode constraint statistics."""

    NONE = 0
    ECONOMIC_TERMINATION = 1
    TIME_LIMIT_COMPLETION = 2
    CENSORED_EXTERNAL_TRUNCATION = 3


def _reason_value(reason: object | None) -> str | None:
    if reason is None:
        return None
    value = getattr(reason, "value", reason)
    return str(value)


def classify_episode_completion(
    *,
    terminated: bool,
    truncated: bool,
    time_limit_truncated: bool,
    termination_reason: object | None,
) -> EpisodeCompletionKind:
    """Classify a transition without treating external censoring as safety."""

    for field_name, value in (
        ("terminated", terminated),
        ("truncated", truncated),
        ("time_limit_truncated", time_limit_truncated),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{field_name} must be a boolean")
    if terminated and truncated:
        raise ValueError("transition cannot be both terminated and truncated")
    if time_limit_truncated and not truncated:
        raise ValueError("time-limit flag requires truncation")

    reason = _reason_value(termination_reason)
    shadow_reason = reason is not None and reason.startswith("shadow_")
    if shadow_reason and not truncated:
        raise ValueError("shadow completion reason requires truncation")

    if terminated:
        return EpisodeCompletionKind.ECONOMIC_TERMINATION
    if truncated:
        if not time_limit_truncated:
            raise ValueError("truncated transition requires TimeLimit.truncated")
        if shadow_reason:
            return EpisodeCompletionKind.CENSORED_EXTERNAL_TRUNCATION
        if reason not in {None, "time_limit", "time_limit_reached"}:
            raise ValueError(f"unknown truncation reason: {reason}")
        return EpisodeCompletionKind.TIME_LIMIT_COMPLETION
    if reason is not None:
        raise ValueError("completion reason requires termination or truncation")
    return EpisodeCompletionKind.NONE


__all__ = ["EpisodeCompletionKind", "classify_episode_completion"]
