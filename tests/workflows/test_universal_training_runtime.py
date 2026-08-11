from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import InstrumentExecutionRule, VolumeUnit
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding


def _digest(label: str) -> str:
    return content_digest(label)


def test_build_universal_instrument_contracts_preserves_point_in_time_rules() -> None:
    from trade_rl.workflows.universal_training_runner import (
        build_universal_instrument_contracts,
    )

    effective = datetime(2024, 1, 1, tzinfo=UTC)
    rule = InstrumentExecutionRule(
        effective_at=effective,
        tick_size=0.01,
        lot_size=0.1,
        minimum_notional=5.0,
    )
    resolution = SimpleNamespace(
        metadata={
            "AAAUSDT": {
                "listed_at": "2020-01-02T03:04:05+00:00",
                "delisted_at": None,
                "tick_size": 0.1,
                "lot_size": 0.01,
                "minimum_notional": 10.0,
            }
        },
        execution_rule_histories={"AAAUSDT": (rule,)},
    )

    contracts = build_universal_instrument_contracts(
        resolution,
        train_symbols=("AAAUSDT",),
    )

    contract = contracts["AAAUSDT"]
    assert contract.symbol == "AAAUSDT"
    assert contract.listed_at == datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert contract.volume_unit is VolumeUnit.QUOTE_NOTIONAL
    assert contract.execution_rules == (rule,)


class _ChildEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, binding: InstrumentDatasetBinding) -> None:
        super().__init__()
        symbol = binding.concrete_symbol
        self.dataset = SimpleNamespace(
            symbols=(symbol,), dataset_id=binding.source_dataset_id
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
                    -1.0, 1.0, shape=(1, 1), dtype=np.float32
                ),
                "asset_state": spaces.Box(-1.0, 1.0, shape=(1, 1), dtype=np.float32),
                "global_state": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                "active": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "current_weights": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.sequence_layout_metadata = None
        self.initial_capital = 10_000.0
        self.decision_hours = 0.25
        self.observation_schema = "child_v1"
        self.observation_contract_digest = _digest(f"obs:{symbol}")
        self.environment_digest = _digest(f"env:{symbol}")
        self.normalizer = SimpleNamespace(statistics_digest=_digest("flat-stats"))
        self.sequence_normalizer = None
        self.alpha_artifact_digest = None
        self.factor_artifact_digest = None
        self.pre_trade_risk = SimpleNamespace(config=SimpleNamespace())
        self.current_index = 0

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

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "current_snapshot": np.zeros((1, 1), dtype=np.float32),
            "asset_state": np.zeros((1, 1), dtype=np.float32),
            "global_state": np.zeros((1,), dtype=np.float32),
            "active": np.ones((1,), dtype=np.float32),
            "current_weights": np.zeros((1,), dtype=np.float32),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        self.current_index += 1
        return self._observation(), float(action[0]), True, False, {}


def _binding(symbol: str) -> InstrumentDatasetBinding:
    return InstrumentDatasetBinding(
        concrete_symbol=symbol,
        source_dataset_id=_digest(f"dataset:{symbol}"),
        symbol_dataset_digest=_digest(f"dataset:{symbol}"),
        execution_metadata_digest=_digest(f"metadata:{symbol}"),
        instrument_descriptor_digest=_digest(f"descriptor:{symbol}"),
        split="train",
    )


def test_routed_factory_assigns_distinct_worker_indices() -> None:
    from trade_rl.workflows.universal_training_runner import (
        UniversalRoutedEnvironmentFactory,
    )

    factory = UniversalRoutedEnvironmentFactory(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        partition_digest=_digest("partition"),
        bindings=(_binding("AAAUSDT"), _binding("BBBUSDT")),
        concrete_environment_factory=_ChildEnv,
        instrument_context_provider=None,
        training_contract_digest=_digest("training-contract"),
        run_seed=19,
    )

    env0 = factory.for_environment_index(0)()
    env1 = factory.for_environment_index(1)()
    try:
        _, info0 = env0.reset(seed=19)
        _, info1 = env1.reset(seed=19)
        binding0 = info0["instrument_episode_binding"]
        binding1 = info1["instrument_episode_binding"]
        assert isinstance(binding0, dict)
        assert isinstance(binding1, dict)
        assert binding0["environment_index"] == 0
        assert binding1["environment_index"] == 1
    finally:
        env0.close()
        env1.close()
