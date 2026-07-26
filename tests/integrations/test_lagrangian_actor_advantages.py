from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch

from trade_rl.integrations.lagrangian_ppo import LagrangianPPO


def _actor_stub(*, multipliers: np.ndarray, normalize: bool) -> Any:
    model = cast(Any, object.__new__(LagrangianPPO))
    snapshot = np.asarray(multipliers, dtype=np.float64).copy()
    snapshot.flags.writeable = False
    model.frozen_lagrange_multipliers = snapshot
    model.normalize_advantage = normalize
    return model


def test_actor_applies_pinned_normalization_after_raw_composition() -> None:
    multipliers = np.asarray([2.0, 0.5], dtype=np.float64)
    model = _actor_stub(multipliers=multipliers, normalize=True)
    reward = torch.asarray([0.4, -0.2, 1.3, 0.1], dtype=torch.float32)
    costs = np.asarray(
        [[0.1, 3.0], [0.4, 2.0], [0.2, 1.0], [0.8, 4.0]],
        dtype=np.float64,
    )

    actual = model._actor_advantages(
        reward_advantages=reward,
        cost_advantages=costs,
    )

    combined_numpy = reward.detach().cpu().numpy() - costs @ multipliers
    raw = torch.as_tensor(combined_numpy, dtype=reward.dtype)
    expected = (raw - raw.mean()) / (raw.std() + 1e-8)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_actor_uses_raw_composition_when_normalization_is_disabled() -> None:
    multipliers = np.asarray([1.5], dtype=np.float64)
    model = _actor_stub(multipliers=multipliers, normalize=False)
    reward = torch.asarray([1.0, -2.0, 4.0], dtype=torch.float32)
    costs = np.asarray([[0.2], [0.4], [0.1]], dtype=np.float64)

    actual = model._actor_advantages(
        reward_advantages=reward,
        cost_advantages=costs,
    )

    combined_numpy = reward.detach().cpu().numpy() - costs @ multipliers
    expected = torch.as_tensor(combined_numpy, dtype=reward.dtype)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_actor_zero_multiplier_path_uses_pinned_ppo_normalization() -> None:
    model = _actor_stub(
        multipliers=np.asarray([0.0]),
        normalize=True,
    )
    reward = torch.asarray([1.0, 3.0, 5.0], dtype=torch.float32)
    costs = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64)

    actual = model._actor_advantages(
        reward_advantages=reward,
        cost_advantages=costs,
    )

    expected = (reward - reward.mean()) / (reward.std() + 1e-8)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
