from __future__ import annotations

import math

import pytest

from trade_rl.rl.rewards import (
    AbsoluteGrowthRewardConfig,
    RewardContext,
    absolute_growth_reward,
    build_reward_context,
    drawdown_severity,
)


def context(
    *,
    baseline_penalty: float = 0.0,
    drawdown_severity_value: float = 0.0,
) -> RewardContext:
    return RewardContext(
        rolling_hybrid_log_growth=0.0,
        rolling_shadow_log_growth=0.0,
        baseline_shortfall=baseline_penalty,
        baseline_tolerance=0.0,
        baseline_penalty=baseline_penalty,
        hybrid_drawdown=0.0,
        drawdown_severity=drawdown_severity_value,
        history_bars=10,
    )


def test_absolute_growth_is_primary_reward() -> None:
    config = AbsoluteGrowthRewardConfig(scale=100.0)

    result = absolute_growth_reward(
        hybrid_log_return=0.012,
        before=context(),
        after=context(),
        config=config,
    )

    assert result.growth_raw == pytest.approx(0.012)
    assert result.baseline_penalty_weighted == pytest.approx(0.0)
    assert result.drawdown_penalty_weighted == pytest.approx(0.0)
    assert result.total_raw == pytest.approx(0.012)
    assert result.total_scaled == pytest.approx(1.2)


def test_baseline_penalty_activates_at_exact_minimum_history() -> None:
    config = AbsoluteGrowthRewardConfig(baseline_tolerance=0.0)

    result = build_reward_context(
        hybrid_returns=(0.0, 0.0, 0.0),
        shadow_returns=(0.01, 0.01, 0.01),
        hybrid_drawdown=0.0,
        window_bars=4,
        minimum_history_bars=3,
        config=config,
    )

    assert result.history_bars == 3
    assert result.baseline_shortfall == pytest.approx(3.0 * math.log1p(0.01))
    assert result.baseline_penalty == pytest.approx(result.baseline_shortfall)


def test_baseline_penalty_is_disabled_before_minimum_history() -> None:
    config = AbsoluteGrowthRewardConfig(baseline_tolerance=0.0)

    result = build_reward_context(
        hybrid_returns=(0.0, 0.0),
        shadow_returns=(0.02, 0.02),
        hybrid_drawdown=0.0,
        window_bars=4,
        minimum_history_bars=3,
        config=config,
    )

    assert result.baseline_shortfall > 0.0
    assert result.baseline_penalty == pytest.approx(0.0)


def test_partial_window_scales_baseline_tolerance() -> None:
    config = AbsoluteGrowthRewardConfig(baseline_tolerance=0.02)

    result = build_reward_context(
        hybrid_returns=(0.0, 0.0),
        shadow_returns=(0.0, 0.0),
        hybrid_drawdown=0.0,
        window_bars=8,
        minimum_history_bars=2,
        config=config,
    )

    assert result.baseline_tolerance == pytest.approx(0.005)


def test_unchanged_penalty_level_is_not_repeated() -> None:
    config = AbsoluteGrowthRewardConfig(
        scale=1.0,
        baseline_penalty_weight=0.10,
        drawdown_penalty_weight=0.05,
    )

    result = absolute_growth_reward(
        hybrid_log_return=0.0,
        before=context(baseline_penalty=0.03, drawdown_severity_value=0.20),
        after=context(baseline_penalty=0.03, drawdown_severity_value=0.20),
        config=config,
    )

    assert result.baseline_penalty_delta == pytest.approx(0.0)
    assert result.drawdown_penalty_delta == pytest.approx(0.0)
    assert result.total_raw == pytest.approx(0.0)


def test_improving_penalty_state_does_not_create_bonus() -> None:
    config = AbsoluteGrowthRewardConfig(scale=1.0)

    result = absolute_growth_reward(
        hybrid_log_return=0.0,
        before=context(baseline_penalty=0.03, drawdown_severity_value=0.20),
        after=context(baseline_penalty=0.01, drawdown_severity_value=0.05),
        config=config,
    )

    assert result.baseline_penalty_delta == pytest.approx(0.0)
    assert result.drawdown_penalty_delta == pytest.approx(0.0)
    assert result.total_raw == pytest.approx(0.0)


def test_drawdown_severity_uses_continuous_increasing_slopes() -> None:
    config = AbsoluteGrowthRewardConfig()

    assert drawdown_severity(0.05, config) == pytest.approx(0.0)
    assert drawdown_severity(0.10, config) == pytest.approx(0.05)
    assert drawdown_severity(0.15, config) == pytest.approx(0.20)
    assert drawdown_severity(0.20, config) == pytest.approx(0.60)
    assert drawdown_severity(0.100001, config) > drawdown_severity(0.10, config)
    assert drawdown_severity(0.150001, config) > drawdown_severity(0.15, config)


def test_new_drawdown_severity_is_penalized_by_configured_weight() -> None:
    config = AbsoluteGrowthRewardConfig(
        scale=100.0,
        baseline_penalty_weight=0.0,
        drawdown_penalty_weight=0.05,
    )

    result = absolute_growth_reward(
        hybrid_log_return=0.0,
        before=context(drawdown_severity_value=0.05),
        after=context(drawdown_severity_value=0.20),
        config=config,
    )

    assert result.drawdown_penalty_delta == pytest.approx(0.15)
    assert result.drawdown_penalty_weighted == pytest.approx(0.0075)
    assert result.total_scaled == pytest.approx(-0.75)


def test_reward_context_rejects_mismatched_book_histories() -> None:
    with pytest.raises(ValueError, match="equal length"):
        build_reward_context(
            hybrid_returns=(0.01,),
            shadow_returns=(0.01, 0.02),
            hybrid_drawdown=0.0,
            window_bars=4,
            minimum_history_bars=2,
            config=AbsoluteGrowthRewardConfig(),
        )


def test_reward_configuration_rejects_decreasing_drawdown_slopes() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        AbsoluteGrowthRewardConfig(drawdown_slopes=(1.0, 0.5, 8.0))
