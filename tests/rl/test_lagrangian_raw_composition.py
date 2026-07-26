from __future__ import annotations

import numpy as np

from trade_rl.rl.lagrangian_advantages import combine_lagrangian_advantages


def test_lagrangian_composition_uses_original_reward_and_cost_units() -> None:
    reward = np.asarray([1.0, -1.0, 2.0], dtype=np.float64)
    costs = np.asarray(
        [
            [2.0, 4.0],
            [0.0, 2.0],
            [6.0, 0.0],
        ],
        dtype=np.float64,
    )
    multipliers = np.asarray([0.5, 0.25], dtype=np.float64)

    combined = combine_lagrangian_advantages(
        reward_advantages=reward,
        cost_advantages=costs,
        multipliers=multipliers,
    )

    np.testing.assert_allclose(
        combined,
        reward - costs @ multipliers,
        rtol=0.0,
        atol=1e-12,
    )


def test_cost_unit_conversion_preserves_raw_lagrangian_advantage() -> None:
    reward = np.asarray([0.4, -0.2, 1.3, 0.1], dtype=np.float64)
    costs = np.asarray(
        [
            [0.1, 3.0],
            [0.4, 2.0],
            [0.2, 1.0],
            [0.8, 4.0],
        ],
        dtype=np.float64,
    )
    multipliers = np.asarray([2.0, 0.3], dtype=np.float64)

    baseline = combine_lagrangian_advantages(
        reward_advantages=reward,
        cost_advantages=costs,
        multipliers=multipliers,
    )
    converted_costs = costs.copy()
    converted_costs[:, 0] *= 10.0
    converted_multipliers = multipliers.copy()
    converted_multipliers[0] /= 10.0
    converted = combine_lagrangian_advantages(
        reward_advantages=reward,
        cost_advantages=converted_costs,
        multipliers=converted_multipliers,
    )

    np.testing.assert_allclose(converted, baseline, rtol=0.0, atol=1e-12)


def test_zero_multipliers_return_raw_reward_advantage() -> None:
    reward = np.asarray([1.0, 3.0, 5.0], dtype=np.float64)
    costs = np.asarray([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])

    combined = combine_lagrangian_advantages(
        reward_advantages=reward,
        cost_advantages=costs,
        multipliers=np.zeros(2, dtype=np.float64),
    )

    np.testing.assert_array_equal(combined, reward)
