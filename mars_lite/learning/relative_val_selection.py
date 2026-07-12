from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rollout_aligned_eval_freq(
    *, total_timesteps: int, one_rollout_steps: int, n_eval_targets: int = 10
) -> int:
    if total_timesteps <= 0 or one_rollout_steps <= 0 or n_eval_targets <= 0:
        raise ValueError("timesteps, rollout size, and target count must be positive")
    rollouts = max(1, total_timesteps // one_rollout_steps)
    rollouts_per_eval = max(1, rollouts // n_eval_targets)
    return rollouts_per_eval * one_rollout_steps


@dataclass(frozen=True)
class RelativeCheckpointScore:
    step: int
    block_excess: tuple[float, ...]
    median_excess: float
    positive_block_ratio: float
    drawdown_excess: float = 0.0
    turnover_excess: float = 0.0
    baseline_fallback: bool = False

    @classmethod
    def from_blocks(
        cls,
        *,
        step: int,
        blocks,
        drawdown_excess: float = 0.0,
        turnover_excess: float = 0.0,
    ) -> "RelativeCheckpointScore":
        values = np.asarray(tuple(blocks), dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("blocks must contain finite values")
        if step < 0:
            raise ValueError("step must be non-negative")
        if not np.isfinite(drawdown_excess) or not np.isfinite(turnover_excess):
            raise ValueError("tie-break metrics must be finite")
        return cls(
            step=int(step),
            block_excess=tuple(float(x) for x in values),
            median_excess=float(np.median(values)),
            positive_block_ratio=float(np.mean(values > 0.0)),
            drawdown_excess=float(drawdown_excess),
            turnover_excess=float(turnover_excess),
        )

    @property
    def eligible(self) -> bool:
        return self.median_excess > 0.0 and self.positive_block_ratio >= 0.5

    @classmethod
    def identity(cls) -> "RelativeCheckpointScore":
        return cls(
            step=0,
            block_excess=(0.0,),
            median_excess=0.0,
            positive_block_ratio=0.0,
            baseline_fallback=True,
        )


def choose_relative_checkpoint(
    scores: list[RelativeCheckpointScore],
) -> RelativeCheckpointScore:
    eligible = [score for score in scores if score.eligible]
    if not eligible:
        return RelativeCheckpointScore.identity()
    return max(
        eligible,
        key=lambda score: (
            score.median_excess,
            score.positive_block_ratio,
            -score.drawdown_excess,
            -score.turnover_excess,
            -score.step,
        ),
    )
