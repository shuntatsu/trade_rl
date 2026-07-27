from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.integrations.sb3_training import (
    _build_parallel_sequence_training_environment,
)


def _full_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "decision_index": spaces.Box(0, 100, shape=(1,), dtype=np.int64),
            "current_snapshot": spaces.Box(
                -np.inf,
                np.inf,
                shape=(1, 1),
                dtype=np.float32,
            ),
            "sequence_15m_values": spaces.Box(
                -np.inf,
                np.inf,
                shape=(1, 1, 1),
                dtype=np.float16,
            ),
            "sequence_15m_available": spaces.Box(
                0,
                1,
                shape=(1, 1, 1),
                dtype=np.uint8,
            ),
            "sequence_15m_staleness": spaces.Box(
                0,
                np.inf,
                shape=(1, 1, 1),
                dtype=np.float16,
            ),
        }
    )


def _compact_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "decision_index": spaces.Box(0, 100, shape=(1,), dtype=np.int64),
            "current_snapshot": spaces.Box(
                -np.inf,
                np.inf,
                shape=(1, 1),
                dtype=np.float32,
            ),
        }
    )


class _SpawnCompactEnvironment(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = _full_space()
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self._compact = False
        self._decision_index = 3

    def set_compact_sequence_training_observations(self, enabled: bool) -> None:
        self._compact = enabled
        self.observation_space = _compact_space() if enabled else _full_space()

    def _observation(self) -> dict[str, np.ndarray]:
        current = {
            "decision_index": np.asarray([self._decision_index], dtype=np.int64),
            "current_snapshot": np.asarray(
                [[float(self._decision_index)]],
                dtype=np.float32,
            ),
        }
        if self._compact:
            return current
        value = np.asarray(
            [[[float(self._decision_index)]]],
            dtype=np.float16,
        )
        return current | {
            "sequence_15m_values": value,
            "sequence_15m_available": np.ones_like(value, dtype=np.uint8),
            "sequence_15m_staleness": np.zeros_like(value, dtype=np.float16),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._decision_index = 3
        return self._observation(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        del action
        self._decision_index = 5
        return self._observation(), 0.25, True, False, {}


def _spawn_environment_factory() -> _SpawnCompactEnvironment:
    return _SpawnCompactEnvironment()


class _SpawnReconstructor:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def reconstruct(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        normalized = np.asarray(indices, dtype=np.int64).copy()
        self.calls.append(normalized)
        values = normalized.astype(np.float16).reshape(-1, 1, 1, 1)
        return {
            "sequence_15m_values": values,
            "sequence_15m_available": np.ones_like(values, dtype=np.uint8),
            "sequence_15m_staleness": np.zeros_like(values, dtype=np.float16),
        }


def test_real_spawn_workers_exchange_only_compact_observations() -> None:
    reconstructor = _SpawnReconstructor()
    environment = _build_parallel_sequence_training_environment(
        _spawn_environment_factory,
        2,
        full_observation_space=_full_space(),
        reconstructor=reconstructor,
    )
    try:
        reset = environment.reset()
        observations, rewards, dones, infos = environment.step(
            np.zeros((2, 1), dtype=np.float32)
        )

        assert reset["sequence_15m_values"].shape == (2, 1, 1, 1)
        assert observations["sequence_15m_values"].shape == (2, 1, 1, 1)
        np.testing.assert_array_equal(
            reset["sequence_15m_values"].reshape(-1),
            np.asarray((3, 3), dtype=np.float16),
        )
        np.testing.assert_array_equal(
            observations["sequence_15m_values"].reshape(-1),
            np.asarray((3, 3), dtype=np.float16),
        )
        np.testing.assert_allclose(rewards, (0.25, 0.25), rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(dones, (True, True))
        for info in infos:
            terminal = info["terminal_observation"]
            assert terminal["sequence_15m_values"].shape == (1, 1, 1)
            assert float(terminal["sequence_15m_values"].reshape(-1)[0]) == 5.0
        assert len(reconstructor.calls) == 3
        np.testing.assert_array_equal(reconstructor.calls[0], (3, 3))
        np.testing.assert_array_equal(reconstructor.calls[1], (3, 3))
        np.testing.assert_array_equal(reconstructor.calls[2], (5, 5))
    finally:
        environment.close()
