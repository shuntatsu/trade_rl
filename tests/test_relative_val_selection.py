import numpy as np

from mars_lite.learning.relative_val_selection import (
    RelativeCheckpointScore,
    choose_relative_checkpoint,
    rollout_aligned_eval_freq,
)


def test_rollout_aligned_eval_frequency_produces_at_least_ten_targets() -> None:
    freq = rollout_aligned_eval_freq(
        total_timesteps=204_800, one_rollout_steps=2_048, n_eval_targets=10
    )

    assert freq % 2_048 == 0
    assert 204_800 // freq >= 10


def test_checkpoint_requires_positive_median_and_half_positive_blocks() -> None:
    rejected = RelativeCheckpointScore.from_blocks(step=10, blocks=[0.03, -0.04, -0.01])
    accepted = RelativeCheckpointScore.from_blocks(step=20, blocks=[0.03, 0.01, -0.005])

    selected = choose_relative_checkpoint([rejected, accepted])

    assert selected is accepted
    assert selected.median_excess > 0.0
    assert selected.positive_block_ratio >= 0.5


def test_no_valid_checkpoint_returns_identity_fallback() -> None:
    scores = [
        RelativeCheckpointScore.from_blocks(step=10, blocks=[-0.01, 0.0, -0.02]),
        RelativeCheckpointScore.from_blocks(step=20, blocks=[0.01, -0.02, -0.03]),
    ]

    selected = choose_relative_checkpoint(scores)

    assert selected.step == 0
    assert selected.baseline_fallback is True
    assert selected.median_excess == 0.0


def test_tie_break_prefers_lower_drawdown_then_turnover_then_earlier_step() -> None:
    first = RelativeCheckpointScore.from_blocks(
        step=20,
        blocks=[0.01, 0.02, 0.03],
        drawdown_excess=0.02,
        turnover_excess=2.0,
    )
    better = RelativeCheckpointScore.from_blocks(
        step=30,
        blocks=[0.01, 0.02, 0.03],
        drawdown_excess=0.01,
        turnover_excess=3.0,
    )

    assert choose_relative_checkpoint([first, better]) is better
    assert np.isfinite(better.median_excess)
