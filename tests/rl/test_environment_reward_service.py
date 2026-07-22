from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_rl.rl.environment_reward import (
    EnvironmentRewardCoordinator,
    EnvironmentRewardRequest,
)
from trade_rl.rl.rewards import RewardBreakdown
from trade_rl.simulation.accounting import BookState


def _breakdown() -> RewardBreakdown:
    return RewardBreakdown(
        absolute_log_growth=0.01,
        excess_log_growth=0.005,
        incremental_drawdown=0.0,
        rolling_baseline_underperformance=0.0,
        projection_distance=0.2,
        terminal_equity_shortfall=0.0,
        margin_deficit=0.05,
        absolute_component=0.01,
        excess_component=0.0,
        drawdown_penalty=0.0,
        baseline_penalty=0.0,
        projection_penalty=0.0,
        terminal_penalty=0.0,
        margin_penalty=0.05,
        unscaled_total=-0.04,
        scaled_total=-4.0,
    )


class _RecordingTracker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = _breakdown()

    def step(self, **kwargs: Any) -> RewardBreakdown:
        self.calls.append(kwargs)
        return self.result


def _book(*, cash: float, peak: float, margin_deficit: float = 0.0) -> BookState:
    book = BookState.zero(2, 100.0, np.array([10.0, 20.0]))
    book.cash = cash
    book.peak_value = peak
    book.margin_deficit = margin_deficit
    return book


def test_reward_coordinator_maps_books_to_tracker_inputs() -> None:
    tracker = _RecordingTracker()
    coordinator = EnvironmentRewardCoordinator(tracker, initial_capital=100.0)
    hybrid = _book(cash=50.0, peak=100.0, margin_deficit=5.0)
    shadow = _book(cash=80.0, peak=100.0)

    result = coordinator.step(
        EnvironmentRewardRequest(
            hybrid_log_return=0.01,
            shadow_log_return=0.005,
            hybrid=hybrid,
            shadow=shadow,
            projection_distance=0.2,
            hybrid_terminated=True,
            shadow_terminated=False,
        )
    )

    assert result is tracker.result
    assert tracker.calls == [
        {
            "hybrid_log_return": 0.01,
            "shadow_log_return": 0.005,
            "hybrid_drawdown": 0.5,
            "shadow_drawdown": pytest.approx(0.2),
            "projection_distance": 0.2,
            "hybrid_margin_deficit_fraction": 0.05,
            "hybrid_equity_fraction": 0.5,
            "shadow_equity_fraction": 0.8,
            "hybrid_terminated": True,
            "shadow_terminated": False,
        }
    ]


def test_reward_coordinator_clamps_negative_equity_fraction_to_zero() -> None:
    tracker = _RecordingTracker()
    coordinator = EnvironmentRewardCoordinator(tracker, initial_capital=100.0)
    hybrid = _book(cash=-1.0, peak=100.0)
    shadow = _book(cash=100.0, peak=100.0)

    coordinator.step(
        EnvironmentRewardRequest(
            hybrid_log_return=-0.1,
            shadow_log_return=0.0,
            hybrid=hybrid,
            shadow=shadow,
            projection_distance=0.0,
            hybrid_terminated=True,
            shadow_terminated=False,
        )
    )

    assert tracker.calls[0]["hybrid_equity_fraction"] == 0.0
    assert tracker.calls[0]["hybrid_drawdown"] == 1.0


def test_reward_coordinator_requires_positive_initial_capital() -> None:
    with pytest.raises(ValueError, match="initial_capital must be positive"):
        EnvironmentRewardCoordinator(_RecordingTracker(), initial_capital=0.0)
