from __future__ import annotations

import pytest

pytest.importorskip("stable_baselines3")

from tests.strategies.test_ppo_interleaved_training import pooled_market
from trade_rl.strategies.rl.a2c import fit_a2c_strategy


def test_real_cpu_a2c_fit_records_effective_steps_and_scope() -> None:
    dataset = pooled_market()
    strategy = fit_a2c_strategy(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=1.0,
        total_timesteps=11,
        seed=7,
    )
    assert strategy.policy.num_timesteps == 15
    assert strategy.fit_metadata is not None
    assert strategy.fit_metadata.requested_timesteps == 11
    assert strategy.fit_metadata.effective_timesteps == 15
    assert strategy.fit_metadata.step_rounding == "ceil_to_complete_rollout"
    assert strategy.fit_metadata.fit_symbols == ("BTCUSDT", "ETHUSDT")
    assert strategy.fit_metadata.required_coverage_timesteps == 6
    assert strategy.fit_metadata.nominal_full_episodes_per_symbol == 2
