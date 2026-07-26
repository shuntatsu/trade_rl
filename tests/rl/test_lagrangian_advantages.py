from __future__ import annotations

import numpy as np
import pytest
import torch

from trade_rl.rl.lagrangian_advantages import (
    combine_lagrangian_advantages,
    normalize_advantage_vector,
    normalize_cost_advantages,
)


def test_normalize_advantage_vector_uses_population_statistics() -> None:
    advantages = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    normalized = normalize_advantage_vector(advantages)

    np.testing.assert_allclose(
        normalized,
        np.asarray([-1.224744871391589, 0.0, 1.224744871391589]),
        rtol=0.0,
        atol=1e-12,
    )
    assert float(np.mean(normalized)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.std(normalized, ddof=0)) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(advantages, np.asarray([1.0, 2.0, 3.0]))


def test_normalize_advantage_vector_maps_constant_input_to_zero() -> None:
    normalized = normalize_advantage_vector(np.asarray([4.0, 4.0, 4.0]))

    np.testing.assert_array_equal(normalized, np.zeros(3, dtype=np.float64))


def test_normalize_cost_advantages_is_diagnostics_only_and_columnwise() -> None:
    advantages = np.asarray(
        [
            [1.0, 5.0, 10.0],
            [2.0, 5.0, 14.0],
            [3.0, 5.0, 18.0],
        ],
        dtype=np.float64,
    )

    normalized = normalize_cost_advantages(advantages)

    expected_column = np.asarray([-1.224744871391589, 0.0, 1.224744871391589])
    np.testing.assert_allclose(normalized[:, 0], expected_column, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(normalized[:, 1], np.zeros(3, dtype=np.float64))
    np.testing.assert_allclose(normalized[:, 2], expected_column, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(
        advantages,
        np.asarray(
            [
                [1.0, 5.0, 10.0],
                [2.0, 5.0, 14.0],
                [3.0, 5.0, 18.0],
            ]
        ),
    )


def test_requested_normalization_applies_after_raw_lagrangian_composition() -> None:
    reward = np.asarray([1.0, -1.0], dtype=np.float64)
    costs = np.asarray([[2.0, 4.0], [0.0, 2.0]], dtype=np.float64)
    multipliers = np.asarray([0.5, 0.25], dtype=np.float64)

    combined = combine_lagrangian_advantages(
        reward_advantages=reward,
        cost_advantages=costs,
        multipliers=multipliers,
        normalize_combined=True,
    )
    raw_tensor = torch.as_tensor(
        reward - costs @ multipliers,
        dtype=torch.float64,
    )
    expected = (raw_tensor - raw_tensor.mean()) / (raw_tensor.std() + 1e-8)

    np.testing.assert_array_equal(combined, expected.numpy())


def test_zero_multipliers_preserve_raw_or_final_normalized_reward_contract() -> None:
    reward = np.asarray([1.0, 3.0, 5.0], dtype=np.float64)
    costs = np.asarray([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])
    multipliers = np.zeros(2, dtype=np.float64)

    normalized = combine_lagrangian_advantages(
        reward_advantages=reward,
        cost_advantages=costs,
        multipliers=multipliers,
        normalize_combined=True,
    )
    raw = combine_lagrangian_advantages(
        reward_advantages=reward,
        cost_advantages=costs,
        multipliers=multipliers,
    )
    reward_tensor = torch.as_tensor(reward, dtype=torch.float64)
    expected = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-8)

    np.testing.assert_array_equal(normalized, expected.numpy())
    np.testing.assert_array_equal(raw, reward)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: normalize_advantage_vector(np.asarray([], dtype=np.float64)),
            "non-empty",
        ),
        (
            lambda: normalize_advantage_vector(np.asarray([[1.0, 2.0]])),
            "one-dimensional",
        ),
        (
            lambda: normalize_advantage_vector(np.asarray([1.0, np.nan])),
            "finite",
        ),
        (
            lambda: normalize_advantage_vector(
                np.asarray([1.0, 2.0]),
                epsilon=0.0,
            ),
            "epsilon",
        ),
        (
            lambda: normalize_cost_advantages(np.empty((0, 2))),
            "non-empty",
        ),
        (
            lambda: normalize_cost_advantages(np.asarray([1.0, 2.0])),
            "two-dimensional",
        ),
        (
            lambda: normalize_cost_advantages(np.asarray([[1.0, np.inf]])),
            "finite",
        ),
        (
            lambda: combine_lagrangian_advantages(
                reward_advantages=np.asarray([1.0, 2.0]),
                cost_advantages=np.asarray([[1.0], [2.0], [3.0]]),
                multipliers=np.asarray([1.0]),
            ),
            "batch",
        ),
        (
            lambda: combine_lagrangian_advantages(
                reward_advantages=np.asarray([1.0, 2.0]),
                cost_advantages=np.asarray([[1.0, 2.0], [2.0, 3.0]]),
                multipliers=np.asarray([1.0]),
            ),
            "multipliers",
        ),
        (
            lambda: combine_lagrangian_advantages(
                reward_advantages=np.asarray([1.0, 2.0]),
                cost_advantages=np.asarray([[1.0], [2.0]]),
                multipliers=np.asarray([-1.0]),
            ),
            "non-negative",
        ),
        (
            lambda: combine_lagrangian_advantages(
                reward_advantages=np.asarray([1.0, 2.0]),
                cost_advantages=np.asarray([[1.0], [2.0]]),
                multipliers=np.asarray([np.inf]),
            ),
            "finite",
        ),
        (
            lambda: combine_lagrangian_advantages(
                reward_advantages=np.asarray([1.0, 2.0]),
                cost_advantages=np.asarray([[1.0], [2.0]]),
                multipliers=np.asarray([1.0]),
                normalize_combined="yes",  # type: ignore[arg-type]
            ),
            "normalize_combined",
        ),
        (
            lambda: combine_lagrangian_advantages(
                reward_advantages=np.asarray([1.0, 2.0]),
                cost_advantages=np.asarray([[1.0], [2.0]]),
                multipliers=np.asarray([1.0]),
                normalize_combined=True,
                normalize_reward=False,
            ),
            "disagree",
        ),
    ],
)
def test_lagrangian_advantage_validation_fails_closed(
    call: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        call()
