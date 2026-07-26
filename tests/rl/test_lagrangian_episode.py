from __future__ import annotations

import pytest

from trade_rl.rl.lagrangian_episode import (
    EpisodeCompletionKind,
    classify_episode_completion,
)


@pytest.mark.parametrize(
    ("terminated", "truncated", "time_limit", "reason", "expected"),
    [
        (False, False, False, None, EpisodeCompletionKind.NONE),
        (
            True,
            False,
            False,
            "margin_call",
            EpisodeCompletionKind.ECONOMIC_TERMINATION,
        ),
        (
            False,
            True,
            True,
            None,
            EpisodeCompletionKind.TIME_LIMIT_COMPLETION,
        ),
        (
            False,
            True,
            True,
            "shadow_minimum_equity",
            EpisodeCompletionKind.CENSORED_EXTERNAL_TRUNCATION,
        ),
    ],
)
def test_episode_completion_classification(
    terminated: bool,
    truncated: bool,
    time_limit: bool,
    reason: str | None,
    expected: EpisodeCompletionKind,
) -> None:
    assert (
        classify_episode_completion(
            terminated=terminated,
            truncated=truncated,
            time_limit_truncated=time_limit,
            termination_reason=reason,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("terminated", "truncated", "time_limit", "reason", "message"),
    [
        (True, True, True, "margin_call", "both terminated and truncated"),
        (False, True, False, None, "TimeLimit.truncated"),
        (False, True, True, "manual_reset", "unknown truncation reason"),
        (False, False, False, "shadow_minimum_equity", "shadow"),
        (True, False, False, "shadow_minimum_equity", "shadow"),
        (False, False, True, None, "time-limit flag"),
    ],
)
def test_episode_completion_classification_fails_closed(
    terminated: bool,
    truncated: bool,
    time_limit: bool,
    reason: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        classify_episode_completion(
            terminated=terminated,
            truncated=truncated,
            time_limit_truncated=time_limit,
            termination_reason=reason,
        )


def test_episode_completion_accepts_enum_like_reason() -> None:
    class _Reason:
        value = "margin_call"

    assert (
        classify_episode_completion(
            terminated=True,
            truncated=False,
            time_limit_truncated=False,
            termination_reason=_Reason(),
        )
        is EpisodeCompletionKind.ECONOMIC_TERMINATION
    )
