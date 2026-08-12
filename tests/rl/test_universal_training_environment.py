from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import InstrumentContract, VolumeUnit
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.training_environment_contract import (
    training_environment_identity,
    validate_training_environment,
)
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding


def _digest(label: str) -> str:
    return content_digest(label)


@dataclass(frozen=True, slots=True)
class _Dataset:
    symbols: tuple[str, ...]
    dataset_id: str
    timestamps: np.ndarray
    volume: np.ndarray
    close: np.ndarray
    mark_price: np.ndarray
    fee_rate: np.ndarray
    spread_rate: np.ndarray
    max_participation_rate: np.ndarray
    minimum_notional: np.ndarray
    lot_size: np.ndarray
    tick_size: np.ndarray
    volume_units: tuple[VolumeUnit, ...]


class _DictSingleSymbolEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, binding: InstrumentDatasetBinding) -> None:
        super().__init__()
        symbol = binding.concrete_symbol
        start = np.datetime64("2026-01-01T00:00:00", "ns")
        timestamps = start + np.arange(192, dtype=np.int64) * np.timedelta64(15, "m")
        values = np.arange(192, dtype=np.float64)[:, None] + 1.0
        self.dataset = _Dataset(
            symbols=(symbol,),
            dataset_id=binding.source_dataset_id,
            timestamps=timestamps,
            volume=1_000.0 + values,
            close=100.0 + values,
            mark_price=100.0 + values,
            fee_rate=np.full((192, 1), 0.0005),
            spread_rate=np.full((192, 1), 0.0002),
            max_participation_rate=np.full((192, 1), 0.05),
            minimum_notional=np.full((192, 1), 5.0),
            lot_size=np.full((192, 1), 0.001),
            tick_size=np.full((192, 1), 0.1),
            volume_units=(VolumeUnit.QUOTE_NOTIONAL,),
        )
        self.current_index = 128
        self.hybrid = SimpleNamespace(portfolio_value=10_000.0)
        self.shadow = SimpleNamespace(portfolio_value=10_500.0)
        self.minimum_start_index = 128
        self.config = SimpleNamespace(
            initial_capital=10_000.0,
            execution_cost=SimpleNamespace(impact_rate=0.0001),
        )
        self.action_spec = ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            target_weight_count=1,
            risk_tilt_enabled=False,
            validation_mode=ActionValidationMode.FAIL_CLOSED,
        )
        self.action_names = (f"target_weight:{symbol}",)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.sequence_layout_metadata = {
            "feature_counts": {"15m": 2, "1h": 2, "4h": 2, "1d": 2},
            "window_lengths": {"15m": 4, "1h": 4, "4h": 4, "1d": 4},
            "snapshot_width": 2,
            "asset_state_width": 2,
            "global_width": 2,
            "n_symbols": 1,
            "current_weight_shape": (1,),
        }
        component_spaces: dict[str, gym.Space[Any]] = {
            "current_snapshot": spaces.Box(-10.0, 10.0, shape=(1, 2), dtype=np.float32),
            "asset_state": spaces.Box(-10.0, 10.0, shape=(1, 2), dtype=np.float32),
            "global_state": spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
            "active": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "current_weights": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        }
        for timeframe in ("15m", "1h", "4h", "1d"):
            for suffix in ("values", "available", "staleness"):
                component_spaces[f"sequence_{timeframe}_{suffix}"] = spaces.Box(
                    0.0 if suffix != "values" else -10.0,
                    10.0,
                    shape=(1, 4, 2),
                    dtype=np.float32,
                )
        self.observation_space = spaces.Dict(component_spaces)
        self.observation_schema = "single_instrument_sequence_v1"
        self.observation_contract_digest = _digest("observation-contract")
        self.environment_digest = _digest("environment-contract")
        self.action_spec_digest = _digest("concrete-action")
        self.initial_capital = 10_000.0
        self.decision_hours = 0.25
        self.pre_trade_risk = SimpleNamespace(
            config=SimpleNamespace(entry_threshold=0.01, no_trade_band=0.005)
        )
        self.normalizer = None
        self.sequence_normalizer = None
        self.alpha_artifact_digest = None
        self.factor_artifact_digest = None
        self._active = False

    def _observation(self) -> dict[str, np.ndarray]:
        obs = {
            "current_snapshot": np.zeros((1, 2), dtype=np.float32),
            "asset_state": np.zeros((1, 2), dtype=np.float32),
            "global_state": np.zeros((2,), dtype=np.float32),
            "active": np.ones((1,), dtype=np.float32),
            "current_weights": np.zeros((1,), dtype=np.float32),
        }
        for timeframe in ("15m", "1h", "4h", "1d"):
            obs[f"sequence_{timeframe}_values"] = np.zeros((1, 4, 2), dtype=np.float32)
            obs[f"sequence_{timeframe}_available"] = np.ones(
                (1, 4, 2), dtype=np.float32
            )
            obs[f"sequence_{timeframe}_staleness"] = np.zeros(
                (1, 4, 2), dtype=np.float32
            )
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self.current_index = 128
        self._active = True
        return self._observation(), {"start_index": 128, "end_index": 132}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self._active
        self.current_index += 1
        self._active = False
        return self._observation(), float(action[0]), True, False, {}

    def close(self) -> None:
        pass


def _binding(symbol: str) -> InstrumentDatasetBinding:
    return InstrumentDatasetBinding(
        concrete_symbol=symbol,
        source_dataset_id=_digest(f"source:{symbol}"),
        symbol_dataset_digest=_digest(f"dataset:{symbol}"),
        execution_metadata_digest=_digest(f"execution:{symbol}"),
        instrument_descriptor_digest=_digest(f"descriptor:{symbol}"),
        split="train",
    )


def test_causal_instrument_context_has_nine_ordered_descriptors() -> None:
    from trade_rl.rl.universal_instrument_context import CausalInstrumentContextProvider

    binding = _binding("BTCUSDT")
    env = _DictSingleSymbolEnv(binding)
    provider = CausalInstrumentContextProvider(
        contracts={
            "BTCUSDT": InstrumentContract(
                symbol="BTCUSDT",
                listed_at=datetime(2025, 1, 1, tzinfo=UTC),
                volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            )
        }
    )

    context = provider(env, binding)

    assert context.shape == (1, 9)
    assert np.isfinite(context).all()
    assert context[0, 0] > 0.0
    assert context[0, 1] > 0.0
    assert context[0, 2] > 0.0
    assert context[0, 5] == pytest.approx(0.0005)
    assert context[0, 6] == pytest.approx(0.0002)
    assert context[0, 7] == pytest.approx(0.0001)
    assert context[0, 8] == pytest.approx(0.05)


def test_causal_instrument_context_rejects_non_quote_notional_volume() -> None:
    from trade_rl.rl.universal_instrument_context import CausalInstrumentContextProvider

    binding = _binding("BTCUSDT")
    env = _DictSingleSymbolEnv(binding)
    bad_contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=datetime.now(UTC) - timedelta(days=100),
        volume_unit=VolumeUnit.BASE_ASSET,
    )
    provider = CausalInstrumentContextProvider(contracts={"BTCUSDT": bad_contract})

    with pytest.raises(ValueError, match="quote-notional"):
        provider(env, binding)


def test_routed_environment_adds_context_and_exposes_training_identity() -> None:
    from trade_rl.rl.universal_instrument_context import CausalInstrumentContextProvider
    from trade_rl.rl.universal_single_instrument_env import (
        EpisodeRoutedSingleInstrumentEnv,
    )

    symbols = ("BTCUSDT", "ETHUSDT")
    bindings = tuple(_binding(symbol) for symbol in symbols)
    provider = CausalInstrumentContextProvider(
        contracts={
            symbol: InstrumentContract(
                symbol=symbol,
                listed_at=datetime(2025, 1, 1, tzinfo=UTC),
                volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            )
            for symbol in symbols
        }
    )
    env = EpisodeRoutedSingleInstrumentEnv(
        train_symbols=symbols,
        partition_digest=_digest("partition"),
        bindings=bindings,
        environment_factory=_DictSingleSymbolEnv,
        run_seed=11,
        environment_index=0,
        instrument_context_provider=provider,
        training_contract_digest=_digest("universal-training-contract"),
    )

    observation, _ = env.reset(seed=11)
    identity = training_environment_identity(env)
    validate_training_environment(identity, SimpleNamespace(decision_hours=0.25))

    assert isinstance(env.observation_space, spaces.Dict)
    assert env.observation_space.spaces["instrument_context"].shape == (1, 9)
    assert observation["instrument_context"].shape == (1, 9)
    assert identity["action_names"] == ("target_weight:INSTRUMENT",)
    assert identity["action_size"] == 1
    assert identity["initial_capital"] == 10_000.0
    assert env.sequence_layout_metadata["instrument_context_width"] == 9
    assert env.is_universal_single_instrument is True
    assert env.dataset.symbols == ("BTCUSDT",)
    assert env.minimum_start_index == 128
    assert env.hybrid.portfolio_value == 10_000.0
    assert env.shadow.portfolio_value == 10_500.0


def test_sequence_extractor_actually_uses_instrument_context() -> None:
    from trade_rl.rl.policies import SequenceAssetFeatureExtractor

    binding = _binding("BTCUSDT")
    child = _DictSingleSymbolEnv(binding)
    observation_space = spaces.Dict(
        {
            **child.observation_space.spaces,
            "instrument_context": spaces.Box(
                -np.inf, np.inf, shape=(1, 9), dtype=np.float32
            ),
        }
    )
    metadata = dict(child.sequence_layout_metadata)
    metadata["instrument_context_width"] = 9
    torch.manual_seed(3)
    extractor = SequenceAssetFeatureExtractor(
        observation_space,
        **metadata,
        sequence_tcn_capacity="compact",
        d_model=32,
        timeframe_attention_heads=4,
        timeframe_attention_layers=1,
        asset_attention_heads=4,
        asset_attention_layers=1,
        dropout=0.0,
    )
    raw = child._observation()
    first = {key: torch.as_tensor(value)[None] for key, value in raw.items()}
    second = {key: value.clone() for key, value in first.items()}
    first["instrument_context"] = torch.zeros((1, 1, 9), dtype=torch.float32)
    second["instrument_context"] = torch.ones((1, 1, 9), dtype=torch.float32)

    with torch.no_grad():
        left = extractor(first)
        right = extractor(second)

    assert left.shape == right.shape
    assert not torch.allclose(left, right)
