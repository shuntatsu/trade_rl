from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


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


class _ContextProvider:
    schema_digest = _digest("instrument-context-schema")

    def __call__(self, _environment: object, _binding: object) -> np.ndarray:
        return np.zeros((1, 9), dtype=np.float32)


class _ConcreteTeacherEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, binding: InstrumentDatasetBinding) -> None:
        super().__init__()
        self.symbol = binding.concrete_symbol
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
        self.normalizer = SimpleNamespace(statistics_digest=_digest("stats"))
        self.sequence_normalizer = None
        self.sequence_layout_metadata = None
        self.pre_trade_risk = SimpleNamespace(config=SimpleNamespace())
        self.current_index = 0
        self._end = 0

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
        super().reset(seed=seed)
        start = 10 if options is None else int(options.get("start_idx", 10))
        bars = 1 if options is None else int(options.get("episode_bars", 1))
        self.current_index = start
        self._end = start + bars
        return self._observation(), {"start_index": start, "end_index": self._end}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        self.current_index += 1
        return (
            self._observation(),
            float(action[0]),
            self.current_index >= self._end,
            False,
            {},
        )


def test_symbol_teacher_environment_uses_generic_identity_for_every_ticker() -> None:
    from trade_rl.workflows.universal_teacher_runtime import (
        build_universal_symbol_teacher_environment,
    )

    digests: list[str] = []
    for symbol in ("AAAUSDT", "BBBUSDT"):
        environment = build_universal_symbol_teacher_environment(
            symbol=symbol,
            binding=_binding(symbol),
            concrete_environment_factory=_ConcreteTeacherEnv,
            instrument_context_provider=_ContextProvider(),
            partition_digest=_digest("partition"),
            training_contract_digest=_digest("training-contract"),
            run_seed=23,
        )
        try:
            observation, _ = environment.reset(
                options={"start_idx": 10, "episode_bars": 1}
            )
            assert environment.action_names == ("target_weight:INSTRUMENT",)
            assert environment.symbols == ("INSTRUMENT",)
            assert observation["instrument_context"].shape == (1, 9)
            digests.append(environment.action_spec_digest)
        finally:
            environment.close()

    assert digests[0] == digests[1]


def test_oracle_batches_clip_manifest_range_to_environment_trainable_closure(
    monkeypatch,
) -> None:
    import trade_rl.workflows.universal_teacher_runtime as module

    binding = _binding("AAAUSDT")
    observed: list[tuple[int, int]] = []

    class Environment:
        minimum_start_index = 96
        dataset = SimpleNamespace(n_bars=900, dataset_id=binding.source_dataset_id)

        def close(self) -> None:
            return None

    def build(_environment, *, train_range, max_episodes, **_kwargs):
        observed.append(train_range)
        assert max_episodes == 1
        return SimpleNamespace(dataset_id=binding.source_dataset_id)

    monkeypatch.setattr(module, "build_episode_oracle_batch_for_environment", build)
    result = module.build_universal_oracle_batches(
        train_symbols=("AAAUSDT",),
        bindings=(binding,),
        concrete_environment_factory=lambda _binding: Environment(),
        fold_train_range=(0, 1_000),
        behavior_cloning_seed=17,
        n_envs=1,
    )

    assert tuple(result) == ("AAAUSDT",)
    assert observed == [(96, 900)]


def test_causal_trend_batch_uses_point_in_time_targets_without_solver() -> None:
    from trade_rl.workflows.universal_teacher_runtime import (
        build_episode_trend_batch_for_environment,
        causal_trend_teacher_digest,
    )

    n_bars = 24
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.exp(np.arange(n_bars, dtype=np.float64) * 0.01)[:, None]
    open_price = np.vstack((close[0], close[:-1]))
    dataset = MarketDataset(
        dataset_id="b" * 64,
        symbols=("AAAUSDT",),
        timestamps=timestamps,
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full((n_bars, 1), 1_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("dummy",),
        global_feature_names=("dummy_global",),
        periods_per_year=8_760,
        symbol_active=np.ones((n_bars, 1), dtype=np.bool_),
    )
    trend = TrendStrategy(
        TrendConfig(fast_lookback=1, base_lookback=2, slow_lookback=3)
    )
    environment = SimpleNamespace(
        config=SimpleNamespace(
            initial_state_modes=("cash",),
            signal_delay_decisions=1,
        ),
        dataset=dataset,
        decision_bars=1,
        episode_bars=4,
        minimum_start_index=3,
        trend_strategy=trend,
        initial_weights_for_reset=lambda _mode, _start: np.zeros(1),
    )

    batch = build_episode_trend_batch_for_environment(
        environment,
        train_range=(3, n_bars),
        seed=17,
        max_episodes=2,
    )

    assert batch.episode_count == 2
    assert batch.solver_provenance is None
    assert batch.teacher_config_digest == causal_trend_teacher_digest(environment)
    for contract, targets in zip(batch.contracts, batch.targets, strict=True):
        expected = np.stack(
            [
                trend.targets(dataset, index).base
                for index in range(contract.start, contract.stop - 1)
            ]
        ).astype(np.float32)
        assert np.array_equal(targets, expected)
