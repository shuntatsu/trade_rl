from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.integrations.sb3_environment import (
    _build_training_environment,
    _filtered_environment_factory,
)


class _IndexEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.observation_space = spaces.Box(0.0, 10.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        return np.asarray([self.index], dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        return np.asarray([self.index], dtype=np.float32), 0.0, True, False, {}


class _IndexedFactory:
    def __call__(self) -> _IndexEnv:
        return _IndexEnv(0)

    def for_environment_index(self, index: int) -> Callable[[], _IndexEnv]:
        return lambda: _IndexEnv(index)


def test_filtered_factory_preserves_environment_index_contract() -> None:
    filtered = _filtered_environment_factory(_IndexedFactory())

    zero = filtered.for_environment_index(0)()
    one = filtered.for_environment_index(1)()
    try:
        assert zero.unwrapped.index == 0
        assert one.unwrapped.index == 1
    finally:
        zero.close()
        one.close()


def test_in_process_vector_environment_builds_distinct_indexed_workers() -> None:
    vector = _build_training_environment(
        _filtered_environment_factory(_IndexedFactory()),
        3,
        subprocesses=False,
    )
    try:
        observation = vector.reset()
        assert observation[:, 0].tolist() == [0.0, 1.0, 2.0]
    finally:
        vector.close()
