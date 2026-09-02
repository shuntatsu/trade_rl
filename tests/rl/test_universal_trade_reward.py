from __future__ import annotations

import math

import pytest

from trade_rl.rl.universal_trade_reward import (
    reconcile_universal_trade_reward,
    universal_net_log_growth_reward,
)


def test_reward_telescopes_to_final_wealth() -> None:
    values = (100.0, 101.0, 99.5, 103.25)
    rewards = tuple(
        universal_net_log_growth_reward(before_value=a, after_value=b)
        for a, b in zip(values[:-1], values[1:], strict=True)
    )

    assert sum(rewards) / 100.0 == pytest.approx(math.log(values[-1] / values[0]))
    reconcile_universal_trade_reward(
        rewards=rewards,
        initial_value=values[0],
        final_value=values[-1],
    )


@pytest.mark.parametrize("bad", (0.0, -1.0, float("inf"), float("nan")))
def test_reward_rejects_invalid_wealth(bad: float) -> None:
    with pytest.raises(ValueError):
        universal_net_log_growth_reward(before_value=bad, after_value=100.0)
    with pytest.raises(ValueError):
        universal_net_log_growth_reward(before_value=100.0, after_value=bad)


def test_reward_reconciliation_rejects_tampered_sequence() -> None:
    values = (100.0, 101.0, 99.5, 103.25)
    rewards = tuple(
        universal_net_log_growth_reward(before_value=a, after_value=b)
        for a, b in zip(values[:-1], values[1:], strict=True)
    )
    tampered = (rewards[0] + 1e-3, *rewards[1:])

    with pytest.raises(ValueError, match="reconcil|mismatch"):
        reconcile_universal_trade_reward(
            rewards=tampered,
            initial_value=values[0],
            final_value=values[-1],
        )


def test_reward_handles_finite_wealth_without_ratio_overflow() -> None:
    before = 1e-300
    after = 1e300
    expected = 100.0 * (math.log(after) - math.log(before))

    reward = universal_net_log_growth_reward(
        before_value=before,
        after_value=after,
    )

    assert math.isfinite(reward)
    assert reward == pytest.approx(expected, abs=1e-10, rel=0.0)
    reconcile_universal_trade_reward(
        rewards=(reward,),
        initial_value=before,
        final_value=after,
    )
