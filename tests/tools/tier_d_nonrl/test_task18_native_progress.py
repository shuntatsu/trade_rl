"""Observed native progress contracts: unknown is not zero, debt is not clipped."""

from __future__ import annotations

import pytest

from tools.tier_d_nonrl.task18_native_progress import (
    CHECKPOINT_ADVANCES,
    NativeProgressTracker,
    progress_from_advance,
    validate_observed_progress,
)
from tools.tier_d_nonrl.task18_receipts import ArmAttemptProgress
from trade_rl.evaluation.replay import NativeReplayAdvance


def progress(count, *, equity=100000.0, drawdown=0.0):
    return progress_from_advance(
        NativeReplayAdvance(count, 2903 + count, equity, drawdown)
    )


def test_tracker_keeps_every_observation_and_bounds_durable_writes():
    tracker = NativeProgressTracker()
    assert tracker.latest is None
    checkpoints = []
    for count in range(1, 6600):
        latest = progress(count)
        if tracker.observe(latest):
            checkpoints.append(count)
        assert tracker.latest == latest
    assert checkpoints == [1, *range(64, 6599, 64), 6599]
    assert CHECKPOINT_ADVANCES == 64


@pytest.mark.parametrize(
    "bad",
    (
        ArmAttemptProgress(),
        ArmAttemptProgress(
            native_advances_observed=0,
            last_native_index=2903,
            last_observed_equity_usdt=100000.0,
            max_drawdown_observed=0.0,
        ),
        ArmAttemptProgress(
            native_advances_observed=6600,
            last_native_index=9503,
            last_observed_equity_usdt=100000.0,
            max_drawdown_observed=0.0,
        ),
        ArmAttemptProgress(
            native_advances_observed=1,
            last_native_index=2903,
            last_observed_equity_usdt=100000.0,
            max_drawdown_observed=0.0,
        ),
        ArmAttemptProgress(
            native_advances_observed=1,
            last_native_index=2904,
            last_observed_equity_usdt=None,
            max_drawdown_observed=0.0,
        ),
    ),
)
def test_native_progress_requires_a_complete_bounded_observation(bad):
    with pytest.raises(ValueError, match="frontier"):
        validate_observed_progress(bad)


def test_tracker_refuses_skips_duplicates_and_drawdown_regression():
    tracker = NativeProgressTracker()
    with pytest.raises(ValueError, match="first advance"):
        tracker.observe(progress(2))
    first = progress(1, equity=99000.0, drawdown=0.01)
    assert tracker.observe(first)
    for bad in (first, progress(3), progress(2, drawdown=0.009)):
        with pytest.raises(ValueError, match="regressed"):
            tracker.observe(bad)
        assert tracker.latest == first
    debt = progress(2, equity=-501.0, drawdown=1.00501)
    assert tracker.observe(debt)  # Preserve terminal debt even off checkpoint cadence.
    assert tracker.latest == debt


def test_observer_count_and_index_must_agree():
    with pytest.raises(ValueError, match="frontier"):
        progress_from_advance(NativeReplayAdvance(1, 2905, 100000.0, 0.0))
