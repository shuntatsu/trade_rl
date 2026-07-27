from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

import trade_rl.rl.environment_observation as observation_module
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_observation import EnvironmentObservationAssembler


def test_compact_environment_transport_rejects_invalid_runtime_boundaries() -> None:
    environment = object.__new__(ResidualMarketEnv)
    environment.sequence_observation_builder = None
    environment._full_observation_space = spaces.Box(
        -1.0,
        1.0,
        shape=(1,),
        dtype=np.float32,
    )
    environment.observation_space = environment._full_observation_space
    environment._compact_sequence_training_observations = False

    with pytest.raises(TypeError, match="must be boolean"):
        environment.set_compact_sequence_training_observations("yes")  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="require a sequence contract"):
        environment.set_compact_sequence_training_observations(True)

    environment.sequence_observation_builder = object()
    with pytest.raises(RuntimeError, match="require a Dict space"):
        environment.set_compact_sequence_training_observations(True)


def test_compact_assembler_rejects_missing_sequence_contract() -> None:
    assembler = object.__new__(EnvironmentObservationAssembler)
    assembler.sequence_observation_builder = None

    with pytest.raises(RuntimeError, match="requires a sequence contract"):
        assembler.compact_observation(
            SimpleNamespace(current_index=0),
            trends=object(),  # type: ignore[arg-type]
            alpha=np.zeros(1, dtype=np.float64),
            factor_basis=np.empty((0, 1), dtype=np.float64),
            pre_trade_risk=object(),  # type: ignore[arg-type]
        )


def test_observation_fallback_builds_sequence_without_policy_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembler = object.__new__(EnvironmentObservationAssembler)
    assembler.dataset = SimpleNamespace(n_features=1)
    assembler.layout = object()
    assembler.sequence_policy_plane = None
    assembler.sequence_normalizer = None
    sequence = object()

    class SequenceBuilder:
        def build(self, dataset: object, *, index: int) -> object:
            assert dataset is assembler.dataset
            assert index == 7
            return sequence

    assembler.sequence_observation_builder = SequenceBuilder()
    current = np.asarray([1.0, 2.0], dtype=np.float32)

    def flat_pair(
        self: object,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        del self, args, kwargs
        return current.copy(), current.copy()

    assembler.flat_pair = MethodType(flat_pair, assembler)  # type: ignore[method-assign]

    def build_structured(**kwargs: object) -> dict[str, np.ndarray]:
        assert kwargs["sequence"] is sequence
        np.testing.assert_array_equal(kwargs["current_flat"], current)
        return {
            "current_snapshot": np.asarray([[1.0]], dtype=np.float32),
            "sequence_15m_values": np.asarray([[[2.0]]], dtype=np.float16),
        }

    monkeypatch.setattr(
        observation_module,
        "build_structured_policy_observation",
        build_structured,
    )

    result = assembler.observation(
        SimpleNamespace(current_index=7),
        trends=object(),  # type: ignore[arg-type]
        alpha=np.zeros(1, dtype=np.float64),
        factor_basis=np.empty((0, 1), dtype=np.float64),
        pre_trade_risk=object(),  # type: ignore[arg-type]
    )

    assert isinstance(result, dict)
    assert result["decision_index"].tolist() == [7]
    assert result["sequence_15m_values"].shape == (1, 1, 1)
