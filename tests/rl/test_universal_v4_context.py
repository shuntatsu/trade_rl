from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from trade_rl.data.v4_context import (
    CROSS_MARKET_CORE_NAMES,
    GLOBAL_MARKET_CORE_NAMES,
    V4ContextBlock,
    V4TargetContext,
)
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv
from trade_rl.rl.universal_v4_context import V4ContextProvider


def _digest(char: str) -> str:
    return char * 64


def _context(
    *,
    symbol: str = "ETHUSDT",
    local_names: tuple[str, ...] = CROSS_MARKET_CORE_NAMES,
    global_names: tuple[str, ...] = GLOBAL_MARKET_CORE_NAMES,
    profile_name: str = "cross_market_core_v1",
) -> V4TargetContext:
    decisions = np.asarray([100, 101], dtype=np.int64)
    local_values = np.arange(2 * len(local_names), dtype=np.float64).reshape(
        2, len(local_names)
    )
    global_values = (
        np.arange(2 * len(global_names), dtype=np.float64).reshape(2, len(global_names))
        + 1000.0
    )
    local = V4ContextBlock(
        feature_names=local_names,
        decision_indices=decisions,
        values=local_values,
        available=np.ones(local_values.shape, dtype=np.bool_),
        staleness_hours=np.zeros(local_values.shape, dtype=np.float64),
        source_digest=_digest("1"),
    )
    global_market = V4ContextBlock(
        feature_names=global_names,
        decision_indices=decisions,
        values=global_values,
        available=np.ones(global_values.shape, dtype=np.bool_),
        staleness_hours=np.zeros(global_values.shape, dtype=np.float64),
        source_digest=_digest("2"),
    )
    return V4TargetContext(
        symbol=symbol,
        local=local,
        global_market=global_market,
        beta=np.asarray([0.75, 1.25], dtype=np.float64),
        beta_available=np.asarray([True, True], dtype=np.bool_),
        beta_source_digest=_digest("3"),
        profile_name=profile_name,
    )


def test_v4_context_provider_resolves_exact_artifact_backed_row() -> None:
    context = _context()
    provider = V4ContextProvider(contexts={context.symbol: context})

    resolved = provider.resolve(symbol="ETHUSDT", decision_index=101)

    assert provider.local_width == len(CROSS_MARKET_CORE_NAMES)
    assert provider.global_width == len(GLOBAL_MARKET_CORE_NAMES)
    assert len(provider.schema_digest) == 64
    assert len(provider.digest) == 64
    assert resolved.digest == context.policy_row_digest(1)
    assert resolved.local_values.dtype == np.float32
    assert resolved.local_values.shape == (1, len(CROSS_MARKET_CORE_NAMES))
    assert resolved.global_values.shape == (1, len(GLOBAL_MARKET_CORE_NAMES))
    assert resolved.beta.shape == (1, 1)
    assert resolved.beta_available.shape == (1, 1)
    np.testing.assert_allclose(resolved.local_values, context.local.values[1:2])
    np.testing.assert_allclose(
        resolved.global_values, context.global_market.values[1:2]
    )
    np.testing.assert_allclose(resolved.beta, [[1.25]])
    np.testing.assert_array_equal(resolved.beta_available, [[1.0]])


def test_v4_context_provider_rejects_unknown_symbol() -> None:
    provider = V4ContextProvider(contexts={"ETHUSDT": _context()})
    with pytest.raises(ValueError, match="symbol"):
        provider.resolve(symbol="SOLUSDT", decision_index=100)


def test_v4_context_provider_rejects_missing_decision_index() -> None:
    provider = V4ContextProvider(contexts={"ETHUSDT": _context()})
    with pytest.raises(ValueError, match="decision"):
        provider.resolve(symbol="ETHUSDT", decision_index=99)


def test_v4_context_provider_does_not_accept_external_beta() -> None:
    provider = V4ContextProvider(contexts={"ETHUSDT": _context()})
    with pytest.raises(TypeError):
        provider.resolve(  # type: ignore[call-arg]
            symbol="ETHUSDT",
            decision_index=100,
            beta=9.0,
            beta_available=True,
        )


def test_v4_context_provider_rejects_feature_order_drift() -> None:
    left = _context(symbol="ETHUSDT")
    right = _context(symbol="SOLUSDT")
    names = tuple(reversed(right.local.feature_names))
    drifted_local = V4ContextBlock(
        feature_names=names,
        decision_indices=right.local.decision_indices,
        values=right.local.values[:, ::-1],
        available=right.local.available[:, ::-1],
        staleness_hours=right.local.staleness_hours[:, ::-1],
        source_digest=_digest("4"),
    )
    right = replace(right, local=drifted_local, digest="")

    with pytest.raises(ValueError, match="schema|feature"):
        V4ContextProvider(contexts={left.symbol: left, right.symbol: right})


def test_v4_context_provider_rejects_profile_width_drift() -> None:
    context = _context(local_names=CROSS_MARKET_CORE_NAMES[:-1])
    with pytest.raises(ValueError, match="width|feature"):
        V4ContextProvider(contexts={context.symbol: context})


class _FakeConcreteEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata: dict[str, Any] = {}

    def __init__(self, binding: InstrumentDatasetBinding) -> None:
        self.dataset = type(
            "Dataset",
            (),
            {
                "symbols": (binding.concrete_symbol,),
                "dataset_id": binding.source_dataset_id,
            },
        )()
        self.observation_space = gym.spaces.Dict(
            {
                "base": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.action_spec = ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            alpha_enabled=False,
            risk_tilt_enabled=False,
            n_factors=0,
            target_weight_count=1,
            validation_mode=ActionValidationMode.FAIL_CLOSED,
        )
        self.action_names = (f"target_weight:{binding.concrete_symbol}",)
        self.observation_contract_digest = _digest("a")
        self.environment_digest = _digest("b")
        self.sequence_layout_metadata = {"base_width": 1}
        self.initial_capital = 100_000.0
        self.decision_hours = 0.25
        self.current_index = 100

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        del seed, options
        self.current_index = 100
        return {"base": np.asarray([100.0], dtype=np.float32)}, {
            "start_index": 100,
            "end_index": 101,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        del action
        self.current_index = 101
        return (
            {"base": np.asarray([101.0], dtype=np.float32)},
            0.0,
            True,
            False,
            {},
        )

    def close(self) -> None:
        return None


def _binding() -> InstrumentDatasetBinding:
    return InstrumentDatasetBinding(
        concrete_symbol="ETHUSDT",
        source_dataset_id=_digest("c"),
        symbol_dataset_digest=_digest("c"),
        execution_metadata_digest=_digest("d"),
        instrument_descriptor_digest=_digest("e"),
        split="train",
    )


def _routed_environment(
    *,
    provider: V4ContextProvider | None,
) -> EpisodeRoutedSingleInstrumentEnv:
    binding = _binding()
    return EpisodeRoutedSingleInstrumentEnv(
        train_symbols=(binding.concrete_symbol,),
        partition_digest=_digest("f"),
        bindings=(binding,),
        environment_factory=_FakeConcreteEnv,
        run_seed=7,
        environment_index=0,
        v4_context_provider=provider,
    )


def test_non_v4_routed_environment_preserves_existing_observation_contract() -> None:
    environment = _routed_environment(provider=None)
    try:
        assert isinstance(environment.observation_space, gym.spaces.Dict)
        assert set(environment.observation_space.spaces) == {"base"}
        observation, _ = environment.reset()
        assert set(observation) == {"base"}
    finally:
        environment.close()


def test_v4_routed_environment_adds_only_authored_context_keys() -> None:
    provider = V4ContextProvider(contexts={"ETHUSDT": _context()})
    environment = _routed_environment(provider=provider)
    expected_v4_keys = {
        "local_cross_market_context",
        "local_cross_market_available",
        "local_cross_market_staleness_hours",
        "global_market_context",
        "global_market_available",
        "global_market_staleness_hours",
        "causal_beta",
        "causal_beta_available",
    }
    try:
        assert isinstance(environment.observation_space, gym.spaces.Dict)
        assert set(environment.observation_space.spaces) == {"base", *expected_v4_keys}
        observation, _ = environment.reset()
        assert set(observation) == {"base", *expected_v4_keys}
        np.testing.assert_allclose(observation["causal_beta"], [[0.75]])
        assert observation["causal_beta"].dtype == np.float32
        assert observation["local_cross_market_context"].shape == (
            1,
            len(CROSS_MARKET_CORE_NAMES),
        )
        assert observation["global_market_context"].shape == (
            1,
            len(GLOBAL_MARKET_CORE_NAMES),
        )
        next_observation, _, terminated, truncated, _ = environment.step(
            np.asarray([0.0], dtype=np.float32)
        )
        assert terminated and not truncated
        np.testing.assert_allclose(next_observation["causal_beta"], [[1.25]])
        np.testing.assert_allclose(
            next_observation["local_cross_market_context"],
            _context().local.values[1:2],
        )
    finally:
        environment.close()


def test_v4_routed_environment_binds_schema_into_identity_and_layout() -> None:
    provider = V4ContextProvider(contexts={"ETHUSDT": _context()})
    plain = _routed_environment(provider=None)
    v4 = _routed_environment(provider=provider)
    try:
        assert plain.observation_contract_digest != v4.observation_contract_digest
        layout = v4.sequence_layout_metadata
        assert layout is not None
        assert layout["v4_local_context_width"] == len(CROSS_MARKET_CORE_NAMES)
        assert layout["v4_global_context_width"] == len(GLOBAL_MARKET_CORE_NAMES)
        assert layout["v4_context_schema_digest"] == provider.schema_digest
    finally:
        plain.close()
        v4.close()
