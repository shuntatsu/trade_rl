from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.training_performance import (
    TrainingPerformanceRecorder,
    activate_training_performance,
    measure_sequence_reconstruction,
    measure_sequence_tensor_conversion,
    write_training_performance_evidence,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _FakePolicy:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock

    def extract_features(self, observations: object) -> object:
        self.clock.advance(4.0)
        return observations


class _FakeEnv:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock

    def step(self, action: object) -> object:
        self.clock.advance(5.0)
        return action


class _FakeModel:
    device = "cpu"

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.policy = _FakePolicy(clock)
        self.env = _FakeEnv(clock)

    def collect_rollouts(self) -> str:
        self.clock.advance(2.0)
        return "rollout"

    def train(self) -> str:
        self.clock.advance(3.0)
        return "trained"


def test_recorder_accumulates_exact_nested_phase_evidence() -> None:
    clock = _Clock()
    model = _FakeModel(clock)
    recorder = TrainingPerformanceRecorder(clock=clock)
    recorder.start(torch_module=torch, device="cpu")

    with activate_training_performance(recorder), recorder.instrument_model(model):
        assert model.collect_rollouts() == "rollout"
        assert model.train() == "trained"
        assert model.policy.extract_features("observations") == "observations"
        assert model.env.step("action") == "action"
        with measure_sequence_reconstruction():
            clock.advance(6.0)
        with measure_sequence_tensor_conversion():
            clock.advance(7.0)

    evidence = recorder.finish(
        torch_module=torch,
        device="cpu",
        requested_environment_steps=8,
        observed_environment_steps=8,
    )

    assert evidence.device_type == "cpu"
    assert evidence.requested_environment_steps == 8
    assert evidence.observed_environment_steps == 8
    assert evidence.wall_clock_seconds == 27.0
    assert evidence.environment_steps_per_second == pytest.approx(8.0 / 27.0)
    assert evidence.collect_rollouts_seconds == 2.0
    assert evidence.optimization_seconds == 3.0
    assert evidence.feature_extraction_host_seconds == 4.0
    assert evidence.environment_step_seconds == 5.0
    assert evidence.sequence_reconstruction_seconds == 6.0
    assert evidence.sequence_tensor_conversion_seconds == 7.0
    assert evidence.collect_rollouts_calls == 1
    assert evidence.optimization_calls == 1
    assert evidence.feature_extraction_calls == 1
    assert evidence.environment_step_calls == 1
    assert evidence.sequence_reconstruction_calls == 1
    assert evidence.sequence_tensor_conversion_calls == 1
    assert evidence.peak_cuda_allocated_bytes is None
    assert evidence.peak_cuda_reserved_bytes is None
    assert "collect_rollouts" not in model.__dict__
    assert "train" not in model.__dict__
    assert "extract_features" not in model.policy.__dict__
    assert "step" not in model.env.__dict__


def test_instrumentation_restores_original_callables_after_failure() -> None:
    clock = _Clock()
    model = _FakeModel(clock)
    recorder = TrainingPerformanceRecorder(clock=clock)
    recorder.start(torch_module=torch, device="cpu")

    def fail() -> None:
        clock.advance(1.0)
        raise RuntimeError("boom")

    model.train = fail  # type: ignore[method-assign]
    original = model.train
    with pytest.raises(RuntimeError, match="boom"):
        with recorder.instrument_model(model):
            model.train()

    assert model.train is original
    assert model.collect_rollouts() == "rollout"


def test_performance_evidence_persists_with_canonical_digest(tmp_path: Path) -> None:
    clock = _Clock()
    recorder = TrainingPerformanceRecorder(clock=clock)
    recorder.start(torch_module=torch, device="cpu")
    clock.advance(2.0)
    evidence = recorder.finish(
        torch_module=torch,
        device="cpu",
        requested_environment_steps=4,
        observed_environment_steps=4,
    )

    path = tmp_path / "training-performance.json"
    write_training_performance_evidence(path, evidence)
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("digest")

    assert payload["schema_version"] == "training_performance_evidence_v1"
    assert digest == content_digest(payload)
    assert len(digest) == 64


def test_recorder_fails_closed_for_invalid_lifecycle_and_values() -> None:
    clock = _Clock()
    recorder = TrainingPerformanceRecorder(clock=clock)

    with pytest.raises(RuntimeError, match="not started"):
        recorder.finish(
            torch_module=torch,
            device="cpu",
            requested_environment_steps=1,
            observed_environment_steps=1,
        )

    recorder.start(torch_module=torch, device="cpu")
    with pytest.raises(RuntimeError, match="already started"):
        recorder.start(torch_module=torch, device="cpu")
    clock.advance(1.0)
    with pytest.raises(ValueError, match="observed_environment_steps"):
        recorder.finish(
            torch_module=torch,
            device="cpu",
            requested_environment_steps=1,
            observed_environment_steps=-1,
        )

    other = TrainingPerformanceRecorder(clock=clock)
    other.start(torch_module=torch, device="cpu")
    clock.value = math.inf
    with pytest.raises(ValueError, match="wall_clock_seconds"):
        other.finish(
            torch_module=torch,
            device="cpu",
            requested_environment_steps=1,
            observed_environment_steps=1,
        )
