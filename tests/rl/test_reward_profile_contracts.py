from __future__ import annotations

import pytest

from trade_rl.rl.rewards import RewardConfig


def test_reward_config_rejects_zero_tolerance_progressive_hinge() -> None:
    with pytest.raises(
        ValueError,
        match="baseline_tolerance.*baseline_progressive_power",
    ):
        RewardConfig(
            baseline_tolerance=0.0,
            baseline_progressive_power=2.0,
        )
