from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.integrations.sb3_training import _build_training_environment


class TinyEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}


def _tiny_environment_factory() -> TinyEnvironment:
    return TinyEnvironment()


def test_build_training_environment_returns_direct_environment_for_width_one() -> None:
    calls = 0

    def factory() -> TinyEnvironment:
        nonlocal calls
        calls += 1
        return TinyEnvironment()

    environment = _build_training_environment(factory, 1)
    try:
        assert isinstance(environment, TinyEnvironment)
        assert calls == 1
    finally:
        environment.close()


def test_build_training_environment_uses_two_subprocess_workers() -> None:
    factory: Callable[[], TinyEnvironment] = _tiny_environment_factory
    environment = _build_training_environment(factory, 2)
    try:
        assert environment.num_envs == 2
    finally:
        environment.close()
