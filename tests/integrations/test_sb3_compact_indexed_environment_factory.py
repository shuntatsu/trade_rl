from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class _CompactIndexEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.compact = False
        self.observation_space = spaces.Dict(
            {"value": spaces.Box(0.0, 10.0, shape=(1,), dtype=np.float32)}
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def set_compact_sequence_training_observations(self, enabled: bool) -> None:
        self.compact = enabled

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        del options
        super().reset(seed=seed)
        return {"value": np.asarray([self.index], dtype=np.float32)}, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        del action
        return (
            {"value": np.asarray([self.index], dtype=np.float32)},
            0.0,
            True,
            False,
            {},
        )


class _IndexedFactory:
    def __call__(self) -> _CompactIndexEnv:
        return _CompactIndexEnv(0)

    def for_environment_index(self, index: int) -> Callable[[], _CompactIndexEnv]:
        return lambda: _CompactIndexEnv(index)


def test_compact_sequence_factory_preserves_environment_index_contract() -> None:
    from trade_rl.integrations.sb3_environment import (
        _compact_filtered_environment_factory,
    )

    factory = _compact_filtered_environment_factory(_IndexedFactory())
    zero = factory.for_environment_index(0)()
    two = factory.for_environment_index(2)()
    try:
        assert zero.unwrapped.index == 0
        assert two.unwrapped.index == 2
        assert zero.unwrapped.compact is True
        assert two.unwrapped.compact is True
    finally:
        zero.close()
        two.close()
