from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv


def _digest(label: str) -> str:
    return content_digest(label)


def _binding(symbol: str) -> InstrumentDatasetBinding:
    return InstrumentDatasetBinding(
        concrete_symbol=symbol,
        source_dataset_id=_digest(f"dataset:{symbol}"),
        symbol_dataset_digest=_digest(f"dataset:{symbol}"),
        execution_metadata_digest=_digest(f"metadata:{symbol}"),
        instrument_descriptor_digest=_digest(f"descriptor:{symbol}"),
        split="train",
    )


class _ClosableChild(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, binding: InstrumentDatasetBinding, closed: list[str]) -> None:
        super().__init__()
        self.symbol = binding.concrete_symbol
        self._closed = closed
        self.dataset = SimpleNamespace(
            symbols=(self.symbol,), dataset_id=binding.source_dataset_id
        )
        self.action_spec = ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            target_weight_count=1,
            risk_tilt_enabled=False,
            validation_mode=ActionValidationMode.FAIL_CLOSED,
        )
        self.action_names = (f"target_weight:{self.symbol}",)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "current_snapshot": spaces.Box(
                    -1.0, 1.0, shape=(1, 1), dtype=np.float32
                ),
                "asset_state": spaces.Box(-1.0, 1.0, shape=(1, 1), dtype=np.float32),
                "global_state": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                "active": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "current_weights": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.initial_capital = 10_000.0
        self.decision_hours = 0.25
        self.observation_schema = "child_v1"
        self.observation_contract_digest = _digest(f"obs:{self.symbol}")
        self.environment_digest = _digest(f"env:{self.symbol}")
        self.normalizer = SimpleNamespace(statistics_digest=_digest("flat-stats"))
        self.sequence_normalizer = None
        self.sequence_layout_metadata = None
        self.pre_trade_risk = SimpleNamespace(config=SimpleNamespace())
        self.current_index = 0

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "current_snapshot": np.zeros((1, 1), dtype=np.float32),
            "asset_state": np.zeros((1, 1), dtype=np.float32),
            "global_state": np.zeros((1,), dtype=np.float32),
            "active": np.ones((1,), dtype=np.float32),
            "current_weights": np.zeros((1,), dtype=np.float32),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        del options
        super().reset(seed=seed)
        self.current_index = 10
        return self._observation(), {"start_index": 10, "end_index": 11}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        self.current_index += 1
        return self._observation(), float(action[0]), True, False, {}

    def close(self) -> None:
        self._closed.append(self.symbol)


def test_bounded_child_cache_evicts_completed_symbol_before_loading_next() -> None:
    symbols = ("AAAUSDT", "BBBUSDT")
    closed: list[str] = []
    created: list[str] = []

    def factory(binding: InstrumentDatasetBinding) -> _ClosableChild:
        created.append(binding.concrete_symbol)
        return _ClosableChild(binding, closed)

    environment = EpisodeRoutedSingleInstrumentEnv(
        train_symbols=symbols,
        partition_digest=_digest("partition"),
        bindings=tuple(_binding(symbol) for symbol in symbols),
        environment_factory=factory,
        run_seed=17,
        environment_index=0,
        training_contract_digest=_digest("training"),
        max_cached_environments=1,
    )
    try:
        _, first_info = environment.reset(seed=17)
        first_symbol = first_info["instrument_episode_binding"]["concrete_symbol"]
        environment.step(np.zeros(1, dtype=np.float32))
        _, second_info = environment.reset(seed=17)
        second_symbol = second_info["instrument_episode_binding"]["concrete_symbol"]

        assert first_symbol != second_symbol
        assert created[:2] == [first_symbol, second_symbol]
        assert closed == [first_symbol]
    finally:
        environment.close()
