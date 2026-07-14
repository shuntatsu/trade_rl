from __future__ import annotations

from trade_rl.evaluation.bootstrap import moving_block_mean_test


def test_bootstrap_p_value_has_finite_sample_floor() -> None:
    result = moving_block_mean_test((0.2,) * 20, n_bootstrap=99, seed=3)
    assert result.p_value >= 1 / 100


def test_bootstrap_circular_blocks_are_deterministic() -> None:
    values = (0.1, -0.05, 0.08, -0.02, 0.03, 0.01)
    assert moving_block_mean_test(
        values, n_bootstrap=100, seed=4
    ) == moving_block_mean_test(values, n_bootstrap=100, seed=4)
