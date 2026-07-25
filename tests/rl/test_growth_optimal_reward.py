from __future__ import annotations

import pytest

from trade_rl.rl.rewards import RewardConfig, RewardTracker


def _growth_optimal_config() -> RewardConfig:
    return RewardConfig(
        scale=100.0,
        absolute_growth_weight=1.0,
        excess_growth_weight=0.0,
        incremental_drawdown_weight=0.0,
        baseline_underperformance_weight=0.0,
        projection_penalty_weight=0.0,
        terminal_equity_weight=0.0,
        margin_deficit_weight=0.0,
    )


def test_growth_optimal_reward_is_exact_net_log_growth() -> None:
    tracker = RewardTracker(_growth_optimal_config(), decision_hours=0.25)

    reward = tracker.step(
        hybrid_log_return=0.0125,
        shadow_log_return=-0.50,
        hybrid_drawdown=0.17,
        shadow_drawdown=0.90,
        projection_distance=10.0,
        hybrid_margin_deficit_fraction=0.20,
        hybrid_equity_fraction=0.83,
        hybrid_terminated=False,
        shadow_terminated=True,
    )

    assert reward.absolute_component == pytest.approx(0.0125)
    assert reward.excess_component == pytest.approx(0.0)
    assert reward.drawdown_penalty == pytest.approx(0.0)
    assert reward.baseline_penalty == pytest.approx(0.0)
    assert reward.projection_penalty == pytest.approx(0.0)
    assert reward.terminal_penalty == pytest.approx(0.0)
    assert reward.margin_penalty == pytest.approx(0.0)
    assert reward.scaled_total == pytest.approx(1.25)


def test_growth_optimal_reward_telescopes_to_episode_net_log_growth() -> None:
    tracker = RewardTracker(_growth_optimal_config(), decision_hours=0.25)
    interval_log_returns = (0.01, -0.02, 0.005, 0.03, -0.004)

    total = 0.0
    for interval_log_return in interval_log_returns:
        total += tracker.step(
            hybrid_log_return=interval_log_return,
            shadow_log_return=0.0,
            hybrid_drawdown=0.0,
            shadow_drawdown=0.0,
        ).scaled_total

    assert total == pytest.approx(100.0 * sum(interval_log_returns))


def test_growth_profile_does_not_double_count_terminal_equity_loss() -> None:
    tracker = RewardTracker(_growth_optimal_config(), decision_hours=0.25)

    mild = tracker.step(
        hybrid_log_return=-0.01,
        shadow_log_return=0.0,
        hybrid_drawdown=0.20,
        shadow_drawdown=0.0,
        hybrid_equity_fraction=0.80,
        hybrid_terminated=True,
    )
    tracker.reset()
    severe = tracker.step(
        hybrid_log_return=-0.01,
        shadow_log_return=0.0,
        hybrid_drawdown=0.90,
        shadow_drawdown=0.0,
        hybrid_equity_fraction=0.05,
        hybrid_terminated=True,
    )

    assert mild.terminal_penalty == pytest.approx(0.0)
    assert severe.terminal_penalty == pytest.approx(0.0)
    assert mild.scaled_total == pytest.approx(-1.0)
    assert severe.scaled_total == pytest.approx(-1.0)


def test_legacy_reward_defaults_remain_unchanged() -> None:
    config = RewardConfig()

    assert config.incremental_drawdown_weight == pytest.approx(0.05)
    assert config.baseline_underperformance_weight == pytest.approx(0.10)
    assert config.terminal_equity_weight == pytest.approx(1.0)
    assert config.margin_deficit_weight == pytest.approx(1.0)
