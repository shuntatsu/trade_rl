from __future__ import annotations

from typing import Any

import numpy as np
import torch

from trade_rl.integrations.compact_rollout_buffer import IndexBackedDictRolloutBuffer
from trade_rl.rl.training_performance import (
    TrainingPerformanceRecorder,
    activate_training_performance,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _Reconstructor:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.calls = 0

    def reconstruct(self, decision_indices: np.ndarray) -> dict[str, np.ndarray]:
        self.calls += 1
        self.clock.advance(2.0)
        np.testing.assert_array_equal(decision_indices, np.asarray((5, 6)))
        return {
            "sequence_15m_values": np.asarray(
                [[[1.0]], [[2.0]]],
                dtype=np.float16,
            )
        }


def test_compact_rollout_records_reconstruction_and_tensor_conversion() -> None:
    clock = _Clock()
    recorder = TrainingPerformanceRecorder(clock=clock)
    recorder.start(torch_module=torch, device="cpu")
    reconstructor = _Reconstructor(clock)
    buffer = object.__new__(IndexBackedDictRolloutBuffer)
    buffer._materialized_sequence_observations = None
    buffer.observations = {
        "decision_index": np.asarray([[[5]], [[6]]], dtype=np.int64),
    }
    conversions = 0

    def to_torch(value: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal conversions
        conversions += 1
        clock.advance(3.0)
        return value

    buffer.to_torch = to_torch  # type: ignore[method-assign]

    with activate_training_performance(recorder):
        first = buffer._materialize_sequence_observations(reconstructor)  # type: ignore[arg-type]
        second = buffer._materialize_sequence_observations(reconstructor)  # type: ignore[arg-type]

    evidence = recorder.finish(
        torch_module=torch,
        device="cpu",
        requested_environment_steps=1,
        observed_environment_steps=1,
    )

    assert first is second
    assert reconstructor.calls == 1
    assert conversions == 1
    assert evidence.sequence_reconstruction_calls == 1
    assert evidence.sequence_reconstruction_seconds == 2.0
    assert evidence.sequence_tensor_conversion_calls == 1
    assert evidence.sequence_tensor_conversion_seconds == 3.0
