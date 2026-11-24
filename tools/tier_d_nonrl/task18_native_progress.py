"""Synthetic Task18 observed native frontier; never a completion declaration."""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.tier_d_nonrl.task18_receipts import ArmAttemptProgress, _finite_number
from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from trade_rl.evaluation.replay import (
    NativeReplayAdvance,
    SharedCashLedgerIntervalEvidence,
)

_PLAN = build_task18_protocol().index_plan

# Bound the cost of durable run receipt updates. Every returned advance is
# observed in memory; exceptions flush the last observed frontier. Forced
# termination can only prove the last successfully written checkpoint.
CHECKPOINT_ADVANCES = 64


def validate_observed_progress(progress: ArmAttemptProgress) -> int:
    """Validate a whole native frontier, distinct from historical partial claims."""
    if type(progress) is not ArmAttemptProgress:
        raise ValueError("expected typed native progress")
    count = progress.native_advances_observed
    if (
        type(count) is not int
        or not 1 <= count <= _PLAN.native_advances_per_arm
        or type(progress.last_native_index) is not int
        or progress.last_native_index != _PLAN.actor_start_inclusive + count
        or not _finite_number(progress.last_observed_equity_usdt)
        or not _finite_number(progress.max_drawdown_observed)
        or progress.max_drawdown_observed < 0
    ):
        raise ValueError("invalid observed native progress frontier")
    return count


def progress_from_advance(advance: NativeReplayAdvance) -> ArmAttemptProgress:
    progress = ArmAttemptProgress(
        native_advances_observed=advance.native_advances,
        last_native_index=advance.next_index,
        last_observed_equity_usdt=advance.portfolio_value,
        max_drawdown_observed=advance.max_drawdown,
    )
    validate_observed_progress(progress)
    return progress


@dataclass(slots=True)
class NativeProgressTracker:
    """Exclusive single-arm in-memory witness with periodic durable boundaries."""

    latest: ArmAttemptProgress | None = field(default=None, init=False)

    def observe(self, progress: ArmAttemptProgress) -> bool:
        count = validate_observed_progress(progress)
        drawdown = progress.max_drawdown_observed
        equity = progress.last_observed_equity_usdt
        assert drawdown is not None and equity is not None
        previous = self.latest
        if previous is None:
            if count != 1:
                raise ValueError(
                    "observed native progress must start at the first advance"
                )
        else:
            previous_count = validate_observed_progress(previous)
            previous_drawdown = previous.max_drawdown_observed
            assert previous_drawdown is not None
            if count != previous_count + 1 or drawdown < previous_drawdown:
                raise ValueError(
                    "observed native progress is not consecutive or drawdown regressed"
                )
        self.latest = progress
        return (
            count == 1
            or count % CHECKPOINT_ADVANCES == 0
            or count == _PLAN.native_advances_per_arm
            or equity <= 0
        )


def require_progress_matches_ledger(
    progress: ArmAttemptProgress,
    intervals: tuple[SharedCashLedgerIntervalEvidence, ...],
) -> None:
    """Late completion must not contradict a previously observed native prefix."""
    count = validate_observed_progress(progress)
    if count > len(intervals):
        raise ValueError("completed ledger omits the observed native progress")
    interval = intervals[count - 1]
    if (
        interval.next_index != progress.last_native_index
        or interval.portfolio_value_after != progress.last_observed_equity_usdt
        or interval.max_drawdown_after != progress.max_drawdown_observed
    ):
        raise ValueError("completed ledger contradicts observed native progress")
