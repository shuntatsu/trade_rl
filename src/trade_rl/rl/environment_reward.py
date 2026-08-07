"""Reward transition coordination for one environment decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from trade_rl.rl.rewards import RewardBreakdown
from trade_rl.simulation.accounting import BookState


class RewardTrackerLike(Protocol):
    def step(
        self,
        *,
        hybrid_log_return: float,
        shadow_log_return: float,
        hybrid_drawdown: float,
        shadow_drawdown: float,
        projection_distance: float = 0.0,
        hybrid_margin_deficit_fraction: float = 0.0,
        hybrid_equity_fraction: float = 1.0,
        shadow_equity_fraction: float = 1.0,
        hybrid_terminated: bool = False,
        shadow_terminated: bool = False,
    ) -> RewardBreakdown: ...


@dataclass(frozen=True, slots=True)
class EnvironmentRewardRequest:
    hybrid_log_return: float
    shadow_log_return: float
    hybrid: BookState
    shadow: BookState
    projection_distance: float
    hybrid_terminated: bool
    shadow_terminated: bool


class EnvironmentRewardCoordinator:
    """Map environment account state into the stable reward-tracker contract."""

    def __init__(
        self,
        reward_tracker: RewardTrackerLike,
        *,
        initial_capital: float,
    ) -> None:
        if not np.isfinite(initial_capital) or initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        self.reward_tracker = reward_tracker
        self.initial_capital = float(initial_capital)

    @staticmethod
    def drawdown(book: BookState) -> float:
        value = max(book.portfolio_value, 0.0)
        return min(
            1.0,
            max(0.0, 1.0 - value / max(book.peak_value, value, 1e-12)),
        )

    def step(self, request: EnvironmentRewardRequest) -> RewardBreakdown:
        return self.reward_tracker.step(
            hybrid_log_return=request.hybrid_log_return,
            shadow_log_return=request.shadow_log_return,
            hybrid_drawdown=self.drawdown(request.hybrid),
            shadow_drawdown=self.drawdown(request.shadow),
            projection_distance=request.projection_distance,
            hybrid_margin_deficit_fraction=(
                request.hybrid.margin_deficit / self.initial_capital
            ),
            hybrid_equity_fraction=(
                max(request.hybrid.portfolio_value, 0.0) / self.initial_capital
            ),
            shadow_equity_fraction=(
                max(request.shadow.portfolio_value, 0.0) / self.initial_capital
            ),
            hybrid_terminated=request.hybrid_terminated,
            shadow_terminated=request.shadow_terminated,
        )


__all__ = [
    "EnvironmentRewardCoordinator",
    "EnvironmentRewardRequest",
]
