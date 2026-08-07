"""Deterministic phase, throughput, and CUDA-memory evidence for training."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "training_performance_evidence_v1"
_METRIC_NAMES = (
    "collect_rollouts",
    "optimization",
    "environment_step",
    "feature_extraction",
    "sequence_reconstruction",
    "sequence_tensor_conversion",
)
TRAINING_RUNTIME_PATCHES_ATTRIBUTE = "_trade_rl_training_runtime_patches"


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_seconds(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be finite")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0 or (positive and resolved <= 0.0):
        qualifier = "finite and positive" if positive else "finite and non-negative"
        raise ValueError(f"{field} must be {qualifier}")
    return resolved


def _optional_memory(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _non_negative_integer(value, field=field)


@dataclass(frozen=True, slots=True)
class TrainingPerformanceEvidence:
    """Immutable host timing and CUDA allocator evidence for one member learn call."""

    device_type: str
    requested_environment_steps: int
    observed_environment_steps: int
    wall_clock_seconds: float
    environment_steps_per_second: float
    collect_rollouts_seconds: float
    optimization_seconds: float
    environment_step_seconds: float
    feature_extraction_host_seconds: float
    sequence_reconstruction_seconds: float
    sequence_tensor_conversion_seconds: float
    collect_rollouts_calls: int
    optimization_calls: int
    environment_step_calls: int
    feature_extraction_calls: int
    sequence_reconstruction_calls: int
    sequence_tensor_conversion_calls: int
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None
    component_timers_overlap: bool
    digest: str
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported training performance schema")
        if not isinstance(self.device_type, str) or not self.device_type:
            raise ValueError("device_type must be non-empty")
        requested = _positive_integer(
            self.requested_environment_steps,
            field="requested_environment_steps",
        )
        observed = _positive_integer(
            self.observed_environment_steps,
            field="observed_environment_steps",
        )
        wall = _finite_seconds(
            self.wall_clock_seconds,
            field="wall_clock_seconds",
            positive=True,
        )
        throughput = _finite_seconds(
            self.environment_steps_per_second,
            field="environment_steps_per_second",
            positive=True,
        )
        expected_throughput = observed / wall
        if not math.isclose(
            throughput, expected_throughput, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("environment_steps_per_second is inconsistent")
        for field_name in (
            "collect_rollouts_seconds",
            "optimization_seconds",
            "environment_step_seconds",
            "feature_extraction_host_seconds",
            "sequence_reconstruction_seconds",
            "sequence_tensor_conversion_seconds",
        ):
            _finite_seconds(getattr(self, field_name), field=field_name)
        for field_name in (
            "collect_rollouts_calls",
            "optimization_calls",
            "environment_step_calls",
            "feature_extraction_calls",
            "sequence_reconstruction_calls",
            "sequence_tensor_conversion_calls",
        ):
            _non_negative_integer(getattr(self, field_name), field=field_name)
        _optional_memory(
            self.peak_cuda_allocated_bytes,
            field="peak_cuda_allocated_bytes",
        )
        _optional_memory(
            self.peak_cuda_reserved_bytes,
            field="peak_cuda_reserved_bytes",
        )
        if self.device_type == "cuda":
            if (
                self.peak_cuda_allocated_bytes is None
                or self.peak_cuda_reserved_bytes is None
            ):
                raise ValueError("CUDA evidence requires allocator peak values")
        elif (
            self.peak_cuda_allocated_bytes is not None
            or self.peak_cuda_reserved_bytes is not None
        ):
            raise ValueError("non-CUDA evidence cannot declare CUDA allocator peaks")
        if self.component_timers_overlap is not True:
            raise ValueError("component_timers_overlap must remain true")
        if not isinstance(self.digest, str) or len(self.digest) != 64:
            raise ValueError("training performance digest must be SHA-256")
        if self.digest != content_digest(self.payload(include_digest=False)):
            raise ValueError("training performance digest mismatch")
        object.__setattr__(self, "requested_environment_steps", requested)
        object.__setattr__(self, "observed_environment_steps", observed)
        object.__setattr__(self, "wall_clock_seconds", wall)
        object.__setattr__(self, "environment_steps_per_second", throughput)

    def payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "device_type": self.device_type,
            "requested_environment_steps": self.requested_environment_steps,
            "observed_environment_steps": self.observed_environment_steps,
            "wall_clock_seconds": self.wall_clock_seconds,
            "environment_steps_per_second": self.environment_steps_per_second,
            "collect_rollouts_seconds": self.collect_rollouts_seconds,
            "optimization_seconds": self.optimization_seconds,
            "environment_step_seconds": self.environment_step_seconds,
            "feature_extraction_host_seconds": self.feature_extraction_host_seconds,
            "sequence_reconstruction_seconds": self.sequence_reconstruction_seconds,
            "sequence_tensor_conversion_seconds": self.sequence_tensor_conversion_seconds,
            "collect_rollouts_calls": self.collect_rollouts_calls,
            "optimization_calls": self.optimization_calls,
            "environment_step_calls": self.environment_step_calls,
            "feature_extraction_calls": self.feature_extraction_calls,
            "sequence_reconstruction_calls": self.sequence_reconstruction_calls,
            "sequence_tensor_conversion_calls": self.sequence_tensor_conversion_calls,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "component_timers_overlap": self.component_timers_overlap,
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


@dataclass(slots=True)
class _Metric:
    seconds: float = 0.0
    calls: int = 0


class TrainingPerformanceRecorder:
    """Temporarily observe maintained training boundaries without checkpoint state."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._metrics = {name: _Metric() for name in _METRIC_NAMES}
        self._started_at: float | None = None
        self._finished = False
        self._device_type: str | None = None
        self._resolved_device: object | None = None

    @staticmethod
    def _resolve_device(torch_module: object, device: object) -> tuple[object, str]:
        resolver = getattr(torch_module, "device", None)
        if not callable(resolver):
            raise TypeError("torch module must expose device()")
        resolved = resolver(device)
        device_type = getattr(resolved, "type", None)
        if not isinstance(device_type, str) or not device_type:
            raise TypeError("resolved device must expose a non-empty type")
        return resolved, device_type

    def start(self, *, torch_module: object, device: object) -> None:
        if self._started_at is not None or self._finished:
            raise RuntimeError("training performance recorder already started")
        resolved_device, device_type = self._resolve_device(torch_module, device)
        if device_type == "cuda":
            cuda = getattr(torch_module, "cuda", None)
            if cuda is None or not bool(cuda.is_available()):
                raise RuntimeError("CUDA performance evidence requires available CUDA")
            cuda.synchronize(resolved_device)
            cuda.reset_peak_memory_stats(resolved_device)
        started = float(self._clock())
        if not math.isfinite(started):
            raise ValueError("training performance clock must be finite")
        self._resolved_device = resolved_device
        self._device_type = device_type
        self._started_at = started

    @contextmanager
    def _measure(self, name: str) -> Iterator[None]:
        if self._started_at is None or self._finished:
            raise RuntimeError("training performance recorder is not active")
        started = float(self._clock())
        try:
            yield
        finally:
            elapsed = float(self._clock()) - started
            resolved = _finite_seconds(elapsed, field=f"{name}_seconds")
            metric = self._metrics[name]
            metric.seconds += resolved
            metric.calls += 1

    @contextmanager
    def instrument_model(self, model: object) -> Iterator[None]:
        """Wrap callable hot-path boundaries and restore the exact object layout."""

        patches: list[tuple[object, str, bool, object | None]] = []
        missing = object()
        previous_registry = getattr(
            model,
            TRAINING_RUNTIME_PATCHES_ATTRIBUTE,
            missing,
        )
        if previous_registry is not missing:
            raise RuntimeError("training performance instrumentation is already active")

        def patch(owner: object | None, name: str, metric_name: str) -> None:
            if owner is None:
                return
            original = getattr(owner, name, None)
            if not callable(original):
                return
            namespace = getattr(owner, "__dict__", None)
            if isinstance(namespace, dict) and name in namespace:
                had_local = True
                local_value = namespace[name]
            else:
                had_local = False
                local_value = None

            def timed(*args: Any, **kwargs: Any) -> Any:
                with self._measure(metric_name):
                    return original(*args, **kwargs)

            setattr(owner, name, timed)
            patches.append((owner, name, had_local, local_value))

        patch(model, "collect_rollouts", "collect_rollouts")
        patch(model, "train", "optimization")
        policy = getattr(model, "policy", None)
        patch(policy, "extract_features", "feature_extraction")
        environment = getattr(model, "env", None)
        patch(environment, "step", "environment_step")
        setattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE, tuple(patches))
        try:
            yield
        finally:
            for owner, name, had_local, local_value in reversed(patches):
                if had_local:
                    setattr(owner, name, local_value)
                else:
                    delattr(owner, name)
            if previous_registry is missing:
                delattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE)
            else:
                setattr(
                    model,
                    TRAINING_RUNTIME_PATCHES_ATTRIBUTE,
                    previous_registry,
                )

    @contextmanager
    def measure_sequence_reconstruction(self) -> Iterator[None]:
        with self._measure("sequence_reconstruction"):
            yield

    @contextmanager
    def measure_sequence_tensor_conversion(self) -> Iterator[None]:
        with self._measure("sequence_tensor_conversion"):
            yield

    def finish(
        self,
        *,
        torch_module: object,
        device: object,
        requested_environment_steps: int,
        observed_environment_steps: int,
    ) -> TrainingPerformanceEvidence:
        if self._started_at is None:
            raise RuntimeError("training performance recorder is not started")
        if self._finished:
            raise RuntimeError("training performance recorder is already finished")
        requested = _positive_integer(
            requested_environment_steps,
            field="requested_environment_steps",
        )
        observed = _positive_integer(
            observed_environment_steps,
            field="observed_environment_steps",
        )
        resolved_device, device_type = self._resolve_device(torch_module, device)
        if device_type != self._device_type:
            raise ValueError("training performance device changed during learning")
        peak_allocated: int | None = None
        peak_reserved: int | None = None
        if device_type == "cuda":
            cuda = getattr(torch_module, "cuda", None)
            if cuda is None:
                raise TypeError("torch module must expose cuda")
            cuda.synchronize(resolved_device)
            peak_allocated = _non_negative_integer(
                int(cuda.max_memory_allocated(resolved_device)),
                field="peak_cuda_allocated_bytes",
            )
            peak_reserved = _non_negative_integer(
                int(cuda.max_memory_reserved(resolved_device)),
                field="peak_cuda_reserved_bytes",
            )
        wall = _finite_seconds(
            float(self._clock()) - self._started_at,
            field="wall_clock_seconds",
            positive=True,
        )
        payload: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "device_type": device_type,
            "requested_environment_steps": requested,
            "observed_environment_steps": observed,
            "wall_clock_seconds": wall,
            "environment_steps_per_second": observed / wall,
            "collect_rollouts_seconds": self._metrics["collect_rollouts"].seconds,
            "optimization_seconds": self._metrics["optimization"].seconds,
            "environment_step_seconds": self._metrics["environment_step"].seconds,
            "feature_extraction_host_seconds": self._metrics[
                "feature_extraction"
            ].seconds,
            "sequence_reconstruction_seconds": self._metrics[
                "sequence_reconstruction"
            ].seconds,
            "sequence_tensor_conversion_seconds": self._metrics[
                "sequence_tensor_conversion"
            ].seconds,
            "collect_rollouts_calls": self._metrics["collect_rollouts"].calls,
            "optimization_calls": self._metrics["optimization"].calls,
            "environment_step_calls": self._metrics["environment_step"].calls,
            "feature_extraction_calls": self._metrics["feature_extraction"].calls,
            "sequence_reconstruction_calls": self._metrics[
                "sequence_reconstruction"
            ].calls,
            "sequence_tensor_conversion_calls": self._metrics[
                "sequence_tensor_conversion"
            ].calls,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "component_timers_overlap": True,
        }
        self._finished = True
        return TrainingPerformanceEvidence(
            device_type=device_type,
            requested_environment_steps=requested,
            observed_environment_steps=observed,
            wall_clock_seconds=wall,
            environment_steps_per_second=observed / wall,
            collect_rollouts_seconds=self._metrics["collect_rollouts"].seconds,
            optimization_seconds=self._metrics["optimization"].seconds,
            environment_step_seconds=self._metrics["environment_step"].seconds,
            feature_extraction_host_seconds=self._metrics["feature_extraction"].seconds,
            sequence_reconstruction_seconds=self._metrics[
                "sequence_reconstruction"
            ].seconds,
            sequence_tensor_conversion_seconds=self._metrics[
                "sequence_tensor_conversion"
            ].seconds,
            collect_rollouts_calls=self._metrics["collect_rollouts"].calls,
            optimization_calls=self._metrics["optimization"].calls,
            environment_step_calls=self._metrics["environment_step"].calls,
            feature_extraction_calls=self._metrics["feature_extraction"].calls,
            sequence_reconstruction_calls=self._metrics[
                "sequence_reconstruction"
            ].calls,
            sequence_tensor_conversion_calls=self._metrics[
                "sequence_tensor_conversion"
            ].calls,
            peak_cuda_allocated_bytes=peak_allocated,
            peak_cuda_reserved_bytes=peak_reserved,
            component_timers_overlap=True,
            digest=content_digest(payload),
        )


_ACTIVE_RECORDER: ContextVar[TrainingPerformanceRecorder | None] = ContextVar(
    "trade_rl_training_performance_recorder",
    default=None,
)


@contextmanager
def activate_training_performance(
    recorder: TrainingPerformanceRecorder,
) -> Iterator[None]:
    if not isinstance(recorder, TrainingPerformanceRecorder):
        raise TypeError("recorder must be TrainingPerformanceRecorder")
    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield
    finally:
        _ACTIVE_RECORDER.reset(token)


@contextmanager
def measure_sequence_reconstruction() -> Iterator[None]:
    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        yield
        return
    with recorder.measure_sequence_reconstruction():
        yield


@contextmanager
def measure_sequence_tensor_conversion() -> Iterator[None]:
    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        yield
        return
    with recorder.measure_sequence_tensor_conversion():
        yield


def write_training_performance_evidence(
    path: Path,
    evidence: TrainingPerformanceEvidence,
) -> None:
    if not isinstance(evidence, TrainingPerformanceEvidence):
        raise TypeError("evidence must be TrainingPerformanceEvidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(evidence.payload()))


__all__ = [
    "TRAINING_RUNTIME_PATCHES_ATTRIBUTE",
    "TrainingPerformanceEvidence",
    "TrainingPerformanceRecorder",
    "activate_training_performance",
    "measure_sequence_reconstruction",
    "measure_sequence_tensor_conversion",
    "write_training_performance_evidence",
]
