from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.universal_episode_router import (
    DeterministicBalancedInstrumentRouter,
)
from trade_rl.rl.universal_instrument_binding import (
    GENERIC_INSTRUMENT_SYMBOLS,
    GENERIC_TARGET_WEIGHT_ACTION_NAMES,
    InstrumentDatasetBinding,
)
from trade_rl.rl.universal_single_instrument_env import (
    EpisodeRoutedSingleInstrumentEnv,
)


def _digest(value: object) -> str:
    return content_digest(value)


def _binding(symbol: str, *, split: str = "train") -> InstrumentDatasetBinding:
    return InstrumentDatasetBinding(
        concrete_symbol=symbol,
        source_dataset_id=_digest((symbol, "source")),
        symbol_dataset_digest=_digest((symbol, "dataset")),
        execution_metadata_digest=_digest((symbol, "execution")),
        instrument_descriptor_digest=_digest((symbol, "descriptor")),
        split=split,
    )


class FakeSingleInstrumentEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        binding: InstrumentDatasetBinding,
        *,
        observation_size: int = 2,
        terminate_after: int = 1,
        malformed_termination: bool = False,
    ) -> None:
        super().__init__()
        self.binding = binding
        self.dataset = SimpleNamespace(
            symbols=(binding.concrete_symbol,),
            dataset_id=binding.source_dataset_id,
        )
        self.action_spec = ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            risk_tilt_enabled=False,
            target_weight_count=1,
            validation_mode=ActionValidationMode.FAIL_CLOSED,
        )
        self.action_names = (f"target_weight:{binding.concrete_symbol}",)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.terminate_after = terminate_after
        self.malformed_termination = malformed_termination
        self.steps = 0
        self.reset_seeds: list[int | None] = []
        self.closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.reset_seeds.append(seed)
        self.steps = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {
            "start_index": 5,
            "end_index": 17,
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        self.steps += 1
        terminated: bool | int = self.steps >= self.terminate_after
        if self.malformed_termination:
            terminated = 1
        return (
            np.full(self.observation_space.shape, self.steps, dtype=np.float32),
            float(self.steps),
            terminated,  # type: ignore[return-value]
            False,
            {"concrete_step": self.steps},
        )

    def close(self) -> None:
        self.closed = True


def _build_env(
    *,
    factory_override: Callable[[InstrumentDatasetBinding], FakeSingleInstrumentEnv]
    | None = None,
    bindings: tuple[InstrumentDatasetBinding, ...] | None = None,
    run_seed: int = 17,
    environment_index: int = 0,
) -> tuple[
    EpisodeRoutedSingleInstrumentEnv,
    list[str],
    dict[str, FakeSingleInstrumentEnv],
]:
    train_symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    resolved_bindings = bindings or tuple(_binding(symbol) for symbol in train_symbols)
    calls: list[str] = []
    created: dict[str, FakeSingleInstrumentEnv] = {}

    def factory(binding: InstrumentDatasetBinding) -> FakeSingleInstrumentEnv:
        calls.append(binding.concrete_symbol)
        env = (
            factory_override(binding)
            if factory_override is not None
            else FakeSingleInstrumentEnv(binding)
        )
        created[binding.concrete_symbol] = env
        return env

    env = EpisodeRoutedSingleInstrumentEnv(
        train_symbols=train_symbols,
        partition_digest=_digest("partition"),
        bindings=resolved_bindings,
        environment_factory=factory,
        run_seed=run_seed,
        environment_index=environment_index,
    )
    return env, calls, created


def test_facade_exposes_generic_single_action_and_routes_whole_episodes() -> None:
    env, calls, created = _build_env()
    router = DeterministicBalancedInstrumentRouter(
        train_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        partition_digest=_digest("partition"),
        run_seed=17,
        environment_index=0,
    )
    first_symbol = router.route(0).concrete_symbol
    second_symbol = router.route(1).concrete_symbol

    assert calls == [first_symbol]
    assert env.policy_symbols == GENERIC_INSTRUMENT_SYMBOLS
    assert env.action_names == GENERIC_TARGET_WEIGHT_ACTION_NAMES
    assert env.action_space.shape == (1,)

    _, reset_info = env.reset(seed=17)
    _, _, terminated, truncated, terminal_info = env.step(
        np.asarray([0.25], dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert reset_info["instrument_episode_binding"]["concrete_symbol"] == first_symbol
    assert (
        terminal_info["instrument_episode_binding"]
        == reset_info["instrument_episode_binding"]
    )
    assert (
        terminal_info["instrument_episode_binding_digest"]
        == reset_info["instrument_episode_binding_digest"]
    )
    assert env.completed_episode_count == 1

    _, second_reset = env.reset()

    assert (
        second_reset["instrument_episode_binding"]["concrete_symbol"] == second_symbol
    )
    assert calls == [first_symbol, second_symbol]
    assert created[first_symbol] is not created[second_symbol]


def test_reset_before_episode_completion_and_invalid_step_order_fail_closed() -> None:
    env, _, _ = _build_env(
        factory_override=lambda binding: FakeSingleInstrumentEnv(
            binding,
            terminate_after=2,
        )
    )

    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.asarray([0.0], dtype=np.float32))

    env.reset()
    env.step(np.asarray([0.0], dtype=np.float32))
    with pytest.raises(RuntimeError, match="active episode"):
        env.reset()

    env.step(np.asarray([0.0], dtype=np.float32))
    with pytest.raises(RuntimeError, match="completed"):
        env.step(np.asarray([0.0], dtype=np.float32))


def test_factory_failure_is_propagated_without_symbol_fallback() -> None:
    train_symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    router = DeterministicBalancedInstrumentRouter(
        train_symbols=train_symbols,
        partition_digest=_digest("partition"),
        run_seed=17,
        environment_index=0,
    )
    failing_symbol = router.route(1).concrete_symbol

    def factory(binding: InstrumentDatasetBinding) -> FakeSingleInstrumentEnv:
        if binding.concrete_symbol == failing_symbol:
            raise RuntimeError("injected dataset load failure")
        return FakeSingleInstrumentEnv(binding)

    env, calls, _ = _build_env(factory_override=factory)
    env.reset()
    env.step(np.asarray([0.0], dtype=np.float32))

    with pytest.raises(RuntimeError, match="injected dataset load failure"):
        env.reset()

    assert calls[-1] == failing_symbol
    assert len(calls) == 2


def test_later_environment_must_match_first_observation_contract() -> None:
    train_symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    router = DeterministicBalancedInstrumentRouter(
        train_symbols=train_symbols,
        partition_digest=_digest("partition"),
        run_seed=17,
        environment_index=0,
    )
    mismatched_symbol = router.route(1).concrete_symbol

    def factory(binding: InstrumentDatasetBinding) -> FakeSingleInstrumentEnv:
        return FakeSingleInstrumentEnv(
            binding,
            observation_size=(3 if binding.concrete_symbol == mismatched_symbol else 2),
        )

    env, _, _ = _build_env(factory_override=factory)
    env.reset()
    env.step(np.asarray([0.0], dtype=np.float32))

    with pytest.raises(ValueError, match="observation space"):
        env.reset()


def test_split_leakage_fails_before_environment_factory_call() -> None:
    calls: list[str] = []

    def factory(binding: InstrumentDatasetBinding) -> FakeSingleInstrumentEnv:
        calls.append(binding.concrete_symbol)
        return FakeSingleInstrumentEnv(binding)

    with pytest.raises(ValueError, match="train"):
        _build_env(
            factory_override=factory,
            bindings=(
                _binding("BTCUSDT"),
                _binding("ETHUSDT"),
                _binding("SOLUSDT", split="validation"),
            ),
        )

    assert calls == []


def test_reset_seed_cannot_change_immutable_routing_identity() -> None:
    env, _, created = _build_env(run_seed=17)

    with pytest.raises(ValueError, match="run_seed"):
        env.reset(seed=18)

    env.reset(seed=17)
    active_symbol = env.active_episode_binding.dataset_binding.concrete_symbol
    assert created[active_symbol].reset_seeds


def test_non_boolean_termination_flags_are_rejected() -> None:
    env, _, _ = _build_env(
        factory_override=lambda binding: FakeSingleInstrumentEnv(
            binding,
            malformed_termination=True,
        )
    )
    env.reset()

    with pytest.raises(TypeError, match="terminated"):
        env.step(np.asarray([0.0], dtype=np.float32))


def test_close_closes_every_cached_concrete_environment() -> None:
    env, _, created = _build_env()
    env.reset()
    env.step(np.asarray([0.0], dtype=np.float32))
    env.reset()

    env.close()

    assert created
    assert all(concrete.closed for concrete in created.values())
