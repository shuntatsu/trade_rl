from __future__ import annotations

import pytest

from trade_rl.rl.rewards import relative_interval_reward


def test_relative_reward_preserves_excess_log_return_by_default() -> None:
    reward = relative_interval_reward(
        hybrid_log_return=0.02,
        shadow_log_return=0.01,
        scale=100.0,
        hybrid_terminated=False,
        shadow_terminated=False,
        hybrid_drawdown=0.04,
        shadow_drawdown=0.03,
    )

    assert reward == pytest.approx(1.0)


def test_hybrid_and_shadow_termination_are_not_treated_as_the_same_failure() -> None:
    hybrid_failure = relative_interval_reward(
        hybrid_log_return=-0.9,
        shadow_log_return=-0.1,
        scale=100.0,
        hybrid_terminated=True,
        shadow_terminated=False,
        hybrid_drawdown=0.99,
        shadow_drawdown=0.10,
    )
    shadow_failure = relative_interval_reward(
        hybrid_log_return=-0.1,
        shadow_log_return=-0.9,
        scale=100.0,
        hybrid_terminated=False,
        shadow_terminated=True,
        hybrid_drawdown=0.10,
        shadow_drawdown=0.99,
    )

    assert hybrid_failure == pytest.approx(-100.0)
    assert shadow_failure == pytest.approx(100.0)


def test_downside_and_excess_drawdown_penalties_are_explicit() -> None:
    reward = relative_interval_reward(
        hybrid_log_return=-0.02,
        shadow_log_return=-0.01,
        scale=100.0,
        hybrid_terminated=False,
        shadow_terminated=False,
        hybrid_drawdown=0.08,
        shadow_drawdown=0.03,
        downside_penalty=2.0,
        excess_drawdown_penalty=3.0,
    )

    expected = 100.0 * ((-0.02 + 0.01) - 2.0 * 0.02 - 3.0 * 0.05)
    assert reward == pytest.approx(expected)
