from __future__ import annotations

import pytest

from trade_rl.rl.rewards import RewardConfig, RewardTracker


def test_absolute_log_growth_is_primary_and_excess_is_secondary() -> None:
    tracker = RewardTracker(
        RewardConfig(scale=100.0, absolute_growth_weight=1.0, excess_growth_weight=0.25)
    )
    reward = tracker.step(
        hybrid_log_return=0.02,
        shadow_log_return=0.01,
        hybrid_drawdown=0.0,
        shadow_drawdown=0.0,
    )
    assert reward.absolute_component == pytest.approx(0.02)
    assert reward.excess_component == pytest.approx(0.0025)


def test_drawdown_penalizes_only_new_excess_beyond_dead_zone() -> None:
    tracker = RewardTracker(
        RewardConfig(drawdown_dead_zone=0.01, incremental_drawdown_weight=1.0)
    )
    first = tracker.step(
        hybrid_log_return=0.0,
        shadow_log_return=0.0,
        hybrid_drawdown=0.05,
        shadow_drawdown=0.04,
    )
    second = tracker.step(
        hybrid_log_return=0.0,
        shadow_log_return=0.0,
        hybrid_drawdown=0.05,
        shadow_drawdown=0.04,
    )
    assert first.incremental_drawdown == pytest.approx(0.0)
    assert second.incremental_drawdown == pytest.approx(0.0)


def test_rolling_baseline_hinge_uses_real_time_window() -> None:
    tracker = RewardTracker(
        RewardConfig(baseline_window_hours=8.0, baseline_tolerance=0.001),
        decision_hours=4.0,
    )
    assert tracker.baseline_window_steps == 2
    first = tracker.step(
        hybrid_log_return=0.0,
        shadow_log_return=0.0005,
        hybrid_drawdown=0.0,
        shadow_drawdown=0.0,
    )
    second = tracker.step(
        hybrid_log_return=0.0,
        shadow_log_return=0.002,
        hybrid_drawdown=0.0,
        shadow_drawdown=0.0,
    )
    assert first.baseline_penalty == 0.0
    assert second.baseline_penalty > 0.0


def test_terminal_penalty_is_continuous_not_fixed_jackpot() -> None:
    tracker = RewardTracker(RewardConfig(scale=100.0))
    mild = tracker.step(
        hybrid_log_return=-0.1,
        shadow_log_return=-0.1,
        hybrid_drawdown=0.2,
        shadow_drawdown=0.2,
        hybrid_equity_fraction=0.9,
        hybrid_terminated=True,
    )
    tracker.reset()
    severe = tracker.step(
        hybrid_log_return=-0.1,
        shadow_log_return=-0.1,
        hybrid_drawdown=0.9,
        shadow_drawdown=0.2,
        hybrid_equity_fraction=0.1,
        hybrid_terminated=True,
    )
    assert severe.terminal_penalty > mild.terminal_penalty


def test_margin_deficit_penalty_is_continuous() -> None:
    tracker = RewardTracker(RewardConfig(scale=100.0, margin_deficit_weight=2.0))
    mild = tracker.step(
        hybrid_log_return=0.0,
        shadow_log_return=0.0,
        hybrid_drawdown=0.0,
        shadow_drawdown=0.0,
        hybrid_margin_deficit_fraction=0.01,
    )
    severe = tracker.step(
        hybrid_log_return=0.0,
        shadow_log_return=0.0,
        hybrid_drawdown=0.0,
        shadow_drawdown=0.0,
        hybrid_margin_deficit_fraction=0.10,
    )
    assert mild.margin_penalty == pytest.approx(0.02)
    assert severe.margin_penalty == pytest.approx(0.20)
    assert severe.scaled_total < mild.scaled_total
