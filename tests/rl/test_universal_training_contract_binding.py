from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.rl.universal_single_instrument_env import (
    UNIVERSAL_OBSERVATION_SCHEMA,
    EpisodeRoutedSingleInstrumentEnv,
)


def _digest(label: str) -> str:
    return content_digest(label)


def _binding(symbol: str) -> InstrumentDatasetBinding:
    return InstrumentDatasetBinding(
        concrete_symbol=symbol,
        source_dataset_id=_digest(f"source:{symbol}"),
        symbol_dataset_digest=_digest(f"dataset:{symbol}"),
        execution_metadata_digest=_digest(f"execution:{symbol}"),
        instrument_descriptor_digest=_digest(f"descriptor:{symbol}"),
        split="train",
    )


class _DatasetSpecificContractEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, binding: InstrumentDatasetBinding) -> None:
        super().__init__()
        symbol = binding.concrete_symbol
        self.dataset = SimpleNamespace(
            symbols=(symbol,),
            dataset_id=binding.source_dataset_id,
        )
        self.action_spec = ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            target_weight_count=1,
            risk_tilt_enabled=False,
            validation_mode=ActionValidationMode.FAIL_CLOSED,
        )
        self.action_names = (f"target_weight:{symbol}",)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "current_snapshot": spaces.Box(
                    -10.0, 10.0, shape=(1, 2), dtype=np.float32
                ),
                "asset_state": spaces.Box(
                    -10.0, 10.0, shape=(1, 2), dtype=np.float32
                ),
                "global_state": spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
                "active": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "current_weights": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.sequence_layout_metadata = None
        self.initial_capital = 10_000.0
        self.decision_hours = 0.25
        self.observation_schema = "single_symbol_contract_v1"
        self.observation_contract_digest = _digest(f"observation:{symbol}")
        self.environment_digest = _digest(f"environment:{symbol}")
        self.normalizer = SimpleNamespace(
            digest=_digest(f"normalizer-binding:{symbol}"),
            statistics_digest=_digest("shared-flat-statistics"),
        )
        self.sequence_normalizer = None
        self.alpha_artifact_digest = None
        self.factor_artifact_digest = None
        self.pre_trade_risk = SimpleNamespace(config=SimpleNamespace())
        self._active = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        del options
        super().reset(seed=seed)
        self._active = True
        return self._observation(), {"start_index": 10, "end_index": 11}

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "current_snapshot": np.zeros((1, 2), dtype=np.float32),
            "asset_state": np.zeros((1, 2), dtype=np.float32),
            "global_state": np.zeros((2,), dtype=np.float32),
            "active": np.ones((1,), dtype=np.float32),
            "current_weights": np.zeros((1,), dtype=np.float32),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        assert self._active
        self._active = False
        return self._observation(), float(action[0]), True, False, {}


def test_explicit_universal_contract_allows_dataset_specific_child_digests() -> None:
    symbols = ("AAAUSDT", "BBBUSDT")
    training_contract_digest = _digest("universal-training-contract")
    env = EpisodeRoutedSingleInstrumentEnv(
        train_symbols=symbols,
        partition_digest=_digest("partition"),
        bindings=tuple(_binding(symbol) for symbol in symbols),
        environment_factory=_DatasetSpecificContractEnv,
        run_seed=17,
        environment_index=0,
        training_contract_digest=training_contract_digest,
    )

    expected_observation_digest = content_digest(
        {
            "concrete_observation_contract_digest": None,
            "instrument_context_schema_digest": None,
            "schema_version": UNIVERSAL_OBSERVATION_SCHEMA,
            "training_contract_digest": training_contract_digest,
        }
    )
    assert env.observation_contract_digest == expected_observation_digest

    visited: set[str] = set()
    for _ in symbols:
        _, info = env.reset(seed=17)
        binding = info["instrument_episode_binding"]
        assert isinstance(binding, dict)
        visited.add(str(binding["concrete_symbol"]))
        env.step(np.zeros(1, dtype=np.float32))

    assert visited == set(symbols)
    assert env.normalizer_digest == _digest("shared-flat-statistics")
