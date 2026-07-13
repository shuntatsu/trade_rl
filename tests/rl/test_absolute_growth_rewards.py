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


def config() -> AbsoluteGrowthRewardConfig:
    return AbsoluteGrowthRewardConfig()


def test_default_reward_configuration_matches_approved_design() -> None:
    reward = config()

    assert reward.scale == pytest.approx(100.0)
    assert reward.baseline_window_hours == pytest.approx(720.0)
    assert reward.baseline_minimum_history_hours == pytest.approx(168.0)
    assert reward.baseline_tolerance == pytest.approx(0.015)
    assert reward.baseline_penalty_weight == pytest.approx(0.10)
    assert reward.drawdown_penalty_weight == pytest.approx(0.05)
    assert reward.drawdown_free == pytest.approx(0.05)
    assert reward.drawdown_middle == pytest.approx(0.10)
    assert reward.drawdown_high == pytest.approx(0.15)
    assert reward.drawdown_stop == pytest.approx(0.20)
    assert reward.drawdown_slopes == pytest.approx((1.0, 3.0, 8.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": 0.0},
        {"baseline_window_hours": 0.0},
        {"baseline_minimum_history_hours": 0.0},
        {"baseline_minimum_history_hours": 721.0},
        {"baseline_tolerance": -0.01},
        {"baseline_penalty_weight": -0.01},
        {"drawdown_penalty_weight": -0.01},
        {"drawdown_free": 0.11},
        {"drawdown_middle": 0.04},
        {"drawdown_high": 0.09},
        {"drawdown_stop": 0.14},
        {"drawdown_slopes": (1.0, 0.0, 8.0)},
        {"drawdown_slopes": (1.0, 0.5, 8.0)},
    ],
)
def test_reward_configuration_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AbsoluteGrowthRewardConfig(**kwargs)


def test_baseline_hinge_is_disabled_before_minimum_history() -> None:
    reward = config()
    hybrid = [0.0] * 40 + [-0.01]
    shadow = [0.0] * 41

    context = build_reward_context(
        hybrid_returns=hybrid,
        shadow_returns=shadow,
        hybrid_drawdown=0.01,
        window_bars=180,
        minimum_history_bars=42,
        config=reward,
    )

    assert context.history_bars == 41
    assert context.baseline_tolerance == pytest.approx(0.015 * 41 / 180)
    assert context.baseline_shortfall > context.baseline_tolerance
    assert context.baseline_penalty == pytest.approx(0.0)


def test_baseline_hinge_activates_at_exact_minimum_history() -> None:
    reward = AbsoluteGrowthRewardConfig(baseline_tolerance=0.0)
    hybrid = [0.0] * 41 + [-0.01]
    shadow = [0.0] * 42

    context = build_reward_context(
        hybrid_returns=hybrid,
        shadow_returns=shadow,
        hybrid_drawdown=0.01,
        window_bars=180,
        minimum_history_bars=42,
        config=reward,
    )

    assert context.history_bars == 42
    assert context.baseline_shortfall == pytest.approx(-math.log1p(-0.01))
    assert context.baseline_penalty == pytest.approx(context.baseline_shortfall)


def test_baseline_hinge_uses_scaled_tolerance_after_minimum_history() -> None:
    reward = config()
    hybrid = [0.0] * 89 + [-0.01]
    shadow = [0.0] * 90

    context = build_reward_context(
        hybrid_returns=hybrid,
        shadow_returns=shadow,
        hybrid_drawdown=0.01,
        window_bars=180,
        minimum_history_bars=42,
        config=reward,
    )

    expected_tolerance = 0.015 * 90 / 180
    expected_shortfall = -math.log1p(-0.01)
    assert context.baseline_tolerance == pytest.approx(expected_tolerance)
    assert context.baseline_shortfall == pytest.approx(expected_shortfall)
    assert context.baseline_penalty == pytest.approx(
        expected_shortfall - expected_tolerance
    )
    assert context.rolling_growth_gap == pytest.approx(-expected_shortfall)


def test_reward_context_rejects_mismatched_book_histories() -> None:
    with pytest.raises(ValueError, match="equal length"):
        build_reward_context(
            hybrid_returns=(0.01,),
            shadow_returns=(0.01, 0.02),
            hybrid_drawdown=0.0,
            window_bars=180,
            minimum_history_bars=42,
            config=config(),
        )


def test_drawdown_severity_is_continuous_and_staged() -> None:
    reward = config()

    assert drawdown_severity(0.05, reward) == pytest.approx(0.0)
    assert drawdown_severity(0.075, reward) == pytest.approx(0.025)
    assert drawdown_severity(0.10, reward) == pytest.approx(0.05)
    assert drawdown_severity(0.125, reward) == pytest.approx(0.125)
    assert drawdown_severity(0.15, reward) == pytest.approx(0.20)
    assert drawdown_severity(0.175, reward) == pytest.approx(0.40)
    assert drawdown_severity(0.20, reward) == pytest.approx(0.60)
    assert drawdown_severity(0.100001, reward) > drawdown_severity(0.10, reward)
    assert drawdown_severity(0.150001, reward) > drawdown_severity(0.15, reward)


def test_absolute_log_growth_is_the_primary_reward() -> None:
    reward = absolute_growth_reward(
        hybrid_log_return=0.012,
        before=RewardContext(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0),
        after=RewardContext(0.012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1),
        config=config(),
    )

    assert reward.growth_raw == pytest.approx(0.012)
    assert reward.baseline_penalty_weighted == pytest.approx(0.0)
    assert reward.drawdown_penalty_weighted == pytest.approx(0.0)
    assert reward.total_scaled == pytest.approx(1.2)


def test_reward_penalizes_only_new_hinge_and_drawdown_worsening() -> None:
    reward = config()
    before = RewardContext(
        rolling_hybrid_log_growth=0.0,
        rolling_shadow_log_growth=0.02,
        baseline_shortfall=0.02,
        baseline_tolerance=0.015,
        baseline_penalty=0.005,
        hybrid_drawdown=0.10,
        drawdown_severity=0.05,
        history_bars=180,
    )
    after = RewardContext(
        rolling_hybrid_log_growth=-0.01,
        rolling_shadow_log_growth=0.02,
        baseline_shortfall=0.03,
        baseline_tolerance=0.015,
        baseline_penalty=0.015,
        hybrid_drawdown=0.125,
        drawdown_severity=0.125,
        history_bars=180,
    )

    breakdown = absolute_growth_reward(
        hybrid_log_return=0.004,
        before=before,
        after=after,
        config=reward,
    )

    assert breakdown.growth_raw == pytest.approx(0.004)
    assert breakdown.baseline_penalty_delta == pytest.approx(0.010)
    assert breakdown.baseline_penalty_weighted == pytest.approx(0.001)
    assert breakdown.drawdown_penalty_delta == pytest.approx(0.075)
    assert breakdown.drawdown_penalty_weighted == pytest.approx(0.00375)
    assert breakdown.total_raw == pytest.approx(-0.00075)
    assert breakdown.total_scaled == pytest.approx(-0.075)


def test_reward_does_not_repeat_or_refund_existing_penalties() -> None:
    reward = config()
    before = RewardContext(
        rolling_hybrid_log_growth=-0.01,
        rolling_shadow_log_growth=0.02,
        baseline_shortfall=0.03,
        baseline_tolerance=0.015,
        baseline_penalty=0.015,
        hybrid_drawdown=0.125,
        drawdown_severity=0.125,
        history_bars=180,
    )
    unchanged = RewardContext(
        rolling_hybrid_log_growth=-0.01,
        rolling_shadow_log_growth=0.02,
        baseline_shortfall=0.03,
        baseline_tolerance=0.015,
        baseline_penalty=0.015,
        hybrid_drawdown=0.125,
        drawdown_severity=0.125,
        history_bars=180,
    )
    improved = RewardContext(
        rolling_hybrid_log_growth=0.0,
        rolling_shadow_log_growth=0.02,
        baseline_shortfall=0.02,
        baseline_tolerance=0.015,
        baseline_penalty=0.005,
        hybrid_drawdown=0.08,
        drawdown_severity=0.03,
        history_bars=180,
    )

    unchanged_reward = absolute_growth_reward(
        hybrid_log_return=0.0,
        before=before,
        after=unchanged,
        config=reward,
    )
    improved_reward = absolute_growth_reward(
        hybrid_log_return=0.0,
        before=before,
        after=improved,
        config=reward,
    )

    assert unchanged_reward.total_scaled == pytest.approx(0.0)
    assert improved_reward.total_scaled == pytest.approx(0.0)
