from __future__ import annotations

import pytest

from trade_rl.rl.rewards import (
    AbsoluteGrowthRewardConfig,
    RewardContext,
    absolute_growth_reward,
)


def context(
    *,
    baseline_penalty: float = 0.0,
    drawdown_severity: float = 0.0,
) -> RewardContext:
    return RewardContext(
        rolling_hybrid_log_growth=0.0,
        rolling_shadow_log_growth=0.0,
        baseline_shortfall=0.0,
        baseline_tolerance=0.015,
        baseline_penalty=baseline_penalty,
        hybrid_drawdown=0.0,
        drawdown_severity=drawdown_severity,
        history_bars=180,
    )


def test_absolute_reward_preserves_net_log_growth_by_default() -> None:
    reward = absolute_growth_reward(
        hybrid_log_return=0.02,
        before=context(),
        after=context(),
        config=AbsoluteGrowthRewardConfig(),
    )

    assert reward.growth_raw == pytest.approx(0.02)
    assert reward.total_scaled == pytest.approx(2.0)


def test_existing_penalty_levels_are_not_repeated_or_refunded() -> None:
    unchanged = absolute_growth_reward(
        hybrid_log_return=0.0,
        before=context(baseline_penalty=0.01, drawdown_severity=0.10),
        after=context(baseline_penalty=0.01, drawdown_severity=0.10),
        config=AbsoluteGrowthRewardConfig(),
    )
    recovered = absolute_growth_reward(
        hybrid_log_return=0.0,
        before=context(baseline_penalty=0.01, drawdown_severity=0.10),
        after=context(baseline_penalty=0.0, drawdown_severity=0.0),
        config=AbsoluteGrowthRewardConfig(),
    )

    assert unchanged.total_scaled == pytest.approx(0.0)
    assert recovered.total_scaled == pytest.approx(0.0)


def test_new_baseline_and_drawdown_worsening_are_explicit() -> None:
    reward = absolute_growth_reward(
        hybrid_log_return=-0.02,
        before=context(),
        after=context(baseline_penalty=0.01, drawdown_severity=0.05),
        config=AbsoluteGrowthRewardConfig(
            baseline_penalty_weight=2.0,
            drawdown_penalty_weight=3.0,
        ),
    )

    expected_raw = -0.02 - 2.0 * 0.01 - 3.0 * 0.05
    assert reward.baseline_penalty_delta == pytest.approx(0.01)
    assert reward.drawdown_penalty_delta == pytest.approx(0.05)
    assert reward.total_scaled == pytest.approx(100.0 * expected_raw)
