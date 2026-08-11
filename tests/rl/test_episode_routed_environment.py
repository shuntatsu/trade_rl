from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.episode_routed_environment import EpisodeRoutedSingleInstrumentEnv
from trade_rl.rl.instrument_episode_routing import (
    GENERIC_INSTRUMENT_ACTION_NAMES,
    GENERIC_INSTRUMENT_SYMBOLS,
    InstrumentDatasetBinding,
    InstrumentDatasetSplit,
)


def _digest(label: str) -> str:
    return content_digest(label)


@dataclass(frozen=True, slots=True)
class _Dataset:
    symbols: tuple[str, ...]
    dataset_id: str


class _SingleSymbolEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        symbol: str,
        *,
        dataset_id: str | None = None,
        observation_contract_digest: str | None = None,
        action_spec: ActionSpec | None = None,
        fail_on_reset: bool = False,
        both_terminal_flags: bool = False,
    ) -> None:
        super().__init__()
        self.symbol = symbol
        self.dataset = _Dataset(
            symbols=(symbol,),
            dataset_id=dataset_id or _digest(f"dataset:{symbol}"),
        )
        self.action_spec = action_spec or ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            risk_tilt_enabled=False,
            target_weight_count=1,
            validation_mode=ActionValidationMode.FAIL_CLOSED,
        )
        self.action_names = self.action_spec.names_for_symbols(self.dataset.symbols)
        self.action_spec_digest = _digest(
            f"action:{self.action_spec.mode}:{self.action_names}"
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_schema = "single_instrument_observation_v1"
        self.observation_contract_digest = observation_contract_digest or _digest(
            "observation-contract"
        )
        self.environment_digest = _digest(f"environment:{symbol}")
        self.fail_on_reset = fail_on_reset
        self.both_terminal_flags = both_terminal_flags
        self.reset_count = 0
        self.step_count = 0
        self.last_reset_seed: int | None = None
        self.received_actions: list[np.ndarray] = []
        self.closed = False
        self._active = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if self.fail_on_reset:
            raise RuntimeError(f"reset failed for {self.symbol}")
        super().reset(seed=seed)
        self.last_reset_seed = seed
        self.reset_count += 1
        self.step_count = 0
        self._active = True
        start = 100 * self.reset_count
        return np.array([float(self.reset_count), 0.0], dtype=np.float32), {
            "start_index": start,
            "end_index": start + 12,
            "child_symbol": self.symbol,
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._active:
            raise RuntimeError("child step outside an episode")
        self.received_actions.append(np.asarray(action).copy())
        self.step_count += 1
        self._active = False
        return (
            np.array([float(self.reset_count), 1.0], dtype=np.float32),
            float(action[0]),
            True,
            self.both_terminal_flags,
            {"child_symbol": self.symbol},
        )

    def close(self) -> None:
        self.closed = True


def _binding(
    symbol: str,
    *,
    split: InstrumentDatasetSplit | str = InstrumentDatasetSplit.TRAIN,
) -> InstrumentDatasetBinding:
    return InstrumentDatasetBinding(
        concrete_symbol=symbol,
        source_dataset_id=_digest(f"source:{symbol}"),
        symbol_dataset_digest=_digest(f"dataset:{symbol}"),
        execution_metadata_digest=_digest(f"execution:{symbol}"),
        instrument_descriptor_digest=_digest(f"descriptor:{symbol}"),
        partition_digest=_digest("partition"),
        split=split,
    )


def _wrapper(
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    *,
    environment_index: int = 0,
    child_overrides: dict[str, _SingleSymbolEnv] | None = None,
) -> tuple[EpisodeRoutedSingleInstrumentEnv, dict[str, _SingleSymbolEnv]]:
    children = {symbol: _SingleSymbolEnv(symbol) for symbol in symbols}
    children.update(child_overrides or {})
    bindings = tuple(_binding(symbol) for symbol in symbols)
    wrapper = EpisodeRoutedSingleInstrumentEnv(
        children,
        bindings=bindings,
        run_seed=37,
        environment_index=environment_index,
        partition_digest=_digest("partition"),
    )
    return wrapper, children


def test_wrapper_exposes_only_generic_one_action_policy_contract() -> None:
    wrapper, _ = _wrapper()

    assert wrapper.symbols == GENERIC_INSTRUMENT_SYMBOLS
    assert wrapper.action_names == GENERIC_INSTRUMENT_ACTION_NAMES
    assert wrapper.action_space.shape == (1,)
    assert wrapper.action_spec.mode is ActionMode.TARGET_WEIGHT
    assert wrapper.action_spec.target_weight_count == 1
    assert wrapper.observation_schema == "single_instrument_observation_v1"
    assert wrapper.environment_digest == wrapper.environment_digest


def test_reset_routes_one_child_and_emits_digest_bound_episode_identity() -> None:
    wrapper, children = _wrapper()
    expected_route = wrapper.router.route(0)

    observation, info = wrapper.reset(seed=101)

    selected = expected_route.binding.concrete_symbol
    assert observation.shape == (2,)
    assert children[selected].reset_count == 1
    assert sum(child.reset_count for child in children.values()) == 1
    assert info["generic_symbols"] == GENERIC_INSTRUMENT_SYMBOLS
    assert info["generic_action_names"] == GENERIC_INSTRUMENT_ACTION_NAMES
    episode = info["instrument_episode_binding"]
    assert episode["concrete_symbol"] == selected
    assert episode["completed_episode_count"] == 0
    assert episode["episode_start"] == info["start_index"]
    assert episode["episode_stop"] == info["end_index"]
    assert episode["episode_binding_digest"] == wrapper.active_episode_binding.digest


def test_terminal_transition_preserves_identity_and_advances_only_after_end() -> None:
    wrapper, _ = _wrapper()
    _, reset_info = wrapper.reset(seed=101)
    binding = reset_info["instrument_episode_binding"]

    _, reward, terminated, truncated, terminal_info = wrapper.step(
        np.array([0.25], dtype=np.float32)
    )

    assert reward == pytest.approx(0.25)
    assert terminated is True
    assert truncated is False
    assert terminal_info["instrument_episode_binding"] == binding
    assert terminal_info["terminal_instrument_episode_binding"] == binding
    assert wrapper.completed_episode_count == 1

    _, next_info = wrapper.reset()
    assert next_info["instrument_episode_binding"]["completed_episode_count"] == 1
    assert next_info["instrument_episode_binding"]["routing_position"] == 1


def test_every_symbol_is_selected_once_before_any_repeat() -> None:
    wrapper, _ = _wrapper()
    observed: list[str] = []

    for _ in range(6):
        _, info = wrapper.reset()
        observed.append(info["instrument_episode_binding"]["concrete_symbol"])
        wrapper.step(np.array([0.0], dtype=np.float32))

    assert set(observed[:3]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert set(observed[3:]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def test_mid_episode_reset_and_step_outside_episode_fail_closed() -> None:
    wrapper, _ = _wrapper()

    with pytest.raises(RuntimeError, match="reset|episode"):
        wrapper.step(np.array([0.0], dtype=np.float32))

    wrapper.reset()
    with pytest.raises(RuntimeError, match="active"):
        wrapper.reset()

    wrapper.step(np.array([0.0], dtype=np.float32))
    with pytest.raises(RuntimeError, match="reset|episode"):
        wrapper.step(np.array([0.0], dtype=np.float32))


def test_non_scalar_action_is_rejected_before_child_execution() -> None:
    wrapper, children = _wrapper()
    _, info = wrapper.reset()
    selected = info["instrument_episode_binding"]["concrete_symbol"]

    with pytest.raises(ValueError, match=r"\(1,\)"):
        wrapper.step(np.array([[0.0]], dtype=np.float32))

    assert children[selected].received_actions == []


def test_selected_child_reset_failure_never_falls_back_or_advances_cursor() -> None:
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    provisional, _ = _wrapper(symbols)
    selected = provisional.router.route(0).binding.concrete_symbol
    provisional.close()
    failing = _SingleSymbolEnv(selected, fail_on_reset=True)
    wrapper, children = _wrapper(symbols, child_overrides={selected: failing})

    with pytest.raises(RuntimeError, match=selected):
        wrapper.reset()

    assert wrapper.completed_episode_count == 0
    assert sum(child.reset_count for child in children.values()) == 0
    with pytest.raises(RuntimeError, match=selected):
        wrapper.reset()
    assert wrapper.router.route(0).binding.concrete_symbol == selected


def test_reset_options_cannot_select_or_override_concrete_symbol() -> None:
    wrapper, _ = _wrapper()

    with pytest.raises(ValueError, match="symbol"):
        wrapper.reset(options={"concrete_symbol": "BTCUSDT"})


def test_child_terminal_flags_must_remain_exclusive() -> None:
    symbols = ("BTCUSDT",)
    child = _SingleSymbolEnv("BTCUSDT", both_terminal_flags=True)
    wrapper, _ = _wrapper(symbols, child_overrides={"BTCUSDT": child})
    wrapper.reset()

    with pytest.raises(RuntimeError, match="terminated.*truncated|exclusive"):
        wrapper.step(np.array([0.0], dtype=np.float32))

    assert wrapper.completed_episode_count == 0


def test_construction_rejects_dataset_action_and_observation_contract_mismatch() -> (
    None
):
    binding = _binding("BTCUSDT")

    wrong_dataset = _SingleSymbolEnv("BTCUSDT", dataset_id=_digest("wrong"))
    with pytest.raises(ValueError, match="dataset"):
        EpisodeRoutedSingleInstrumentEnv(
            {"BTCUSDT": wrong_dataset},
            bindings=(binding,),
            run_seed=1,
            environment_index=0,
            partition_digest=_digest("partition"),
        )

    wrong_action = _SingleSymbolEnv(
        "BTCUSDT",
        action_spec=ActionSpec(
            mode=ActionMode.RESIDUAL,
            risk_tilt_enabled=False,
            validation_mode=ActionValidationMode.FAIL_CLOSED,
        ),
    )
    with pytest.raises(ValueError, match="target-weight|action"):
        EpisodeRoutedSingleInstrumentEnv(
            {"BTCUSDT": wrong_action},
            bindings=(binding,),
            run_seed=1,
            environment_index=0,
            partition_digest=_digest("partition"),
        )

    eth_binding = _binding("ETHUSDT")
    children = {
        "BTCUSDT": _SingleSymbolEnv("BTCUSDT"),
        "ETHUSDT": _SingleSymbolEnv(
            "ETHUSDT",
            observation_contract_digest=_digest("different-observation"),
        ),
    }
    with pytest.raises(ValueError, match="observation"):
        EpisodeRoutedSingleInstrumentEnv(
            children,
            bindings=(binding, eth_binding),
            run_seed=1,
            environment_index=0,
            partition_digest=_digest("partition"),
        )


def test_construction_requires_exact_child_and_binding_symbol_closure() -> None:
    binding = _binding("BTCUSDT")

    with pytest.raises(ValueError, match="closure|symbols"):
        EpisodeRoutedSingleInstrumentEnv(
            {"ETHUSDT": _SingleSymbolEnv("ETHUSDT")},
            bindings=(binding,),
            run_seed=1,
            environment_index=0,
            partition_digest=_digest("partition"),
        )


def test_close_closes_every_child_once() -> None:
    wrapper, children = _wrapper()

    wrapper.close()
    wrapper.close()

    assert all(child.closed for child in children.values())
