"""Deterministic compute and support evidence for Cost Critic PPO runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "cost_critic_compute_evidence_v1"
_ROLLOUT_SCHEMA_VERSION = "cost_rollout_storage_v2"
_STORAGE_ARRAY_NAMES = (
    "costs",
    "values",
    "returns",
    "advantages",
    "terminal_values",
    "terminated",
    "truncated",
    "elapsed_hours",
    "completion_kinds",
)


def _non_negative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _optional_positive_float(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


def _tensor_tree_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_tensor_tree_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_tree_bytes(item) for item in value)
    return 0


def _optimizer_state_bytes(optimizer: object) -> int:
    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping):
        raise TypeError("cost_critic_optimizer must expose mapping state")
    return sum(_tensor_tree_bytes(item) for item in state.values())


def _rollout_storage_bytes(storage: object) -> int:
    total = 0
    for name in _STORAGE_ARRAY_NAMES:
        array = getattr(storage, name, None)
        nbytes = getattr(array, "nbytes", None)
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise TypeError(f"cost rollout storage {name} must expose integer nbytes")
        total += nbytes
    return total


def _event_support(model: object, event_names: tuple[str, ...]) -> dict[str, int]:
    metrics = getattr(model, "last_cost_training_metrics", None)
    if not isinstance(metrics, Mapping):
        raise TypeError("model must expose last_cost_training_metrics")
    support: dict[str, int] = {}
    for name in event_names:
        key = f"support/{name}"
        raw = metrics.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"missing finite event support metric: {key}")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0 or not value.is_integer():
            raise ValueError(
                f"event support metric must be a non-negative integer: {key}"
            )
        support[name] = int(value)
    return support


@dataclass(frozen=True, slots=True)
class CostCriticComputeEvidence:
    """Machine-readable compute, memory, and rare-event support evidence."""

    algorithm_identifier: str
    cost_names: tuple[str, ...]
    cost_schema_digest: str
    architecture_digest: str
    rollout_schema_digest: str
    cost_parameter_count: int
    cost_parameter_bytes: int
    cost_optimizer_state_bytes: int
    rollout_storage_bytes: int
    rollout_transition_count: int
    cost_optimizer_steps: int
    event_positive_support: dict[str, int]
    training_seconds: float | None
    environment_steps: int | None
    environment_steps_per_second: float | None
    cost_optimizer_steps_per_second: float | None
    peak_device_memory_bytes: int | None
    digest: str

    def payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "algorithm_identifier": self.algorithm_identifier,
            "cost_names": list(self.cost_names),
            "cost_schema_digest": self.cost_schema_digest,
            "architecture_digest": self.architecture_digest,
            "rollout_schema_digest": self.rollout_schema_digest,
            "cost_parameter_count": self.cost_parameter_count,
            "cost_parameter_bytes": self.cost_parameter_bytes,
            "cost_optimizer_state_bytes": self.cost_optimizer_state_bytes,
            "rollout_storage_bytes": self.rollout_storage_bytes,
            "rollout_transition_count": self.rollout_transition_count,
            "cost_optimizer_steps": self.cost_optimizer_steps,
            "event_positive_support": dict(self.event_positive_support),
            "training_seconds": self.training_seconds,
            "environment_steps": self.environment_steps,
            "environment_steps_per_second": self.environment_steps_per_second,
            "cost_optimizer_steps_per_second": (self.cost_optimizer_steps_per_second),
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def build_cost_critic_compute_evidence(
    model: object,
    *,
    training_seconds: float | None = None,
    environment_steps: int | None = None,
    peak_device_memory_bytes: int | None = None,
) -> CostCriticComputeEvidence:
    """Build deterministic evidence from a trained Cost Critic PPO model."""

    algorithm_identifier = getattr(model, "algorithm_identifier", None)
    if algorithm_identifier != "cost_critic_ppo":
        raise ValueError("model must use algorithm_identifier='cost_critic_ppo'")

    schema = getattr(model, "cost_schema", None)
    cost_names = tuple(getattr(schema, "names", ()))
    cost_schema_digest = getattr(schema, "digest", None)
    event_names = tuple(getattr(schema, "event_names", ()))
    if not cost_names or not isinstance(cost_schema_digest, str):
        raise TypeError("model must expose a typed cost schema")
    if not event_names or any(name not in cost_names for name in event_names):
        raise ValueError("cost schema event names are invalid")

    critic = getattr(model, "cost_critic", None)
    parameters = tuple(getattr(critic, "parameters", lambda: ())())
    architecture_digest = getattr(critic, "architecture_digest", None)
    if not parameters or not isinstance(architecture_digest, str):
        raise TypeError("model must expose a Cost Critic with architecture identity")
    cost_parameter_count = sum(int(parameter.numel()) for parameter in parameters)
    cost_parameter_bytes = sum(
        int(parameter.numel() * parameter.element_size()) for parameter in parameters
    )

    storage = getattr(model, "cost_rollout_storage", None)
    buffer_size = getattr(storage, "buffer_size", None)
    n_envs = getattr(storage, "n_envs", None)
    storage_names = tuple(getattr(storage, "cost_names", ()))
    if (
        isinstance(buffer_size, bool)
        or not isinstance(buffer_size, int)
        or buffer_size <= 0
        or isinstance(n_envs, bool)
        or not isinstance(n_envs, int)
        or n_envs <= 0
    ):
        raise TypeError("model must expose positive cost rollout dimensions")
    if storage_names != cost_names:
        raise ValueError("cost rollout order does not match the Cost Critic schema")
    rollout_transition_count = buffer_size * n_envs
    rollout_storage_bytes = _rollout_storage_bytes(storage)
    rollout_schema_digest = content_digest(
        {
            "schema_version": _ROLLOUT_SCHEMA_VERSION,
            "buffer_size": buffer_size,
            "n_envs": n_envs,
            "cost_names": list(cost_names),
            "cost_schema_digest": cost_schema_digest,
            "storage_arrays": list(_STORAGE_ARRAY_NAMES),
        }
    )

    cost_optimizer_steps = _non_negative_integer(
        getattr(model, "cost_update_count", None),
        field_name="cost_update_count",
    )
    optimizer_state_bytes = _optimizer_state_bytes(
        getattr(model, "cost_critic_optimizer", None)
    )
    support = _event_support(model, event_names)

    resolved_training_seconds = _optional_positive_float(
        training_seconds,
        field_name="training_seconds",
    )
    resolved_environment_steps: int | None = None
    if environment_steps is not None:
        resolved_environment_steps = _non_negative_integer(
            environment_steps,
            field_name="environment_steps",
        )
        if resolved_environment_steps == 0:
            raise ValueError("environment_steps must be positive")
    resolved_peak_memory: int | None = None
    if peak_device_memory_bytes is not None:
        resolved_peak_memory = _non_negative_integer(
            peak_device_memory_bytes,
            field_name="peak_device_memory_bytes",
        )

    environment_steps_per_second = None
    cost_optimizer_steps_per_second = None
    if resolved_training_seconds is not None:
        if resolved_environment_steps is not None:
            environment_steps_per_second = (
                resolved_environment_steps / resolved_training_seconds
            )
        cost_optimizer_steps_per_second = (
            cost_optimizer_steps / resolved_training_seconds
        )

    payload_without_digest: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "algorithm_identifier": algorithm_identifier,
        "cost_names": list(cost_names),
        "cost_schema_digest": cost_schema_digest,
        "architecture_digest": architecture_digest,
        "rollout_schema_digest": rollout_schema_digest,
        "cost_parameter_count": cost_parameter_count,
        "cost_parameter_bytes": cost_parameter_bytes,
        "cost_optimizer_state_bytes": optimizer_state_bytes,
        "rollout_storage_bytes": rollout_storage_bytes,
        "rollout_transition_count": rollout_transition_count,
        "cost_optimizer_steps": cost_optimizer_steps,
        "event_positive_support": support,
        "training_seconds": resolved_training_seconds,
        "environment_steps": resolved_environment_steps,
        "environment_steps_per_second": environment_steps_per_second,
        "cost_optimizer_steps_per_second": cost_optimizer_steps_per_second,
        "peak_device_memory_bytes": resolved_peak_memory,
    }
    return CostCriticComputeEvidence(
        algorithm_identifier=algorithm_identifier,
        cost_names=cost_names,
        cost_schema_digest=cost_schema_digest,
        architecture_digest=architecture_digest,
        rollout_schema_digest=rollout_schema_digest,
        cost_parameter_count=cost_parameter_count,
        cost_parameter_bytes=cost_parameter_bytes,
        cost_optimizer_state_bytes=optimizer_state_bytes,
        rollout_storage_bytes=rollout_storage_bytes,
        rollout_transition_count=rollout_transition_count,
        cost_optimizer_steps=cost_optimizer_steps,
        event_positive_support=support,
        training_seconds=resolved_training_seconds,
        environment_steps=resolved_environment_steps,
        environment_steps_per_second=environment_steps_per_second,
        cost_optimizer_steps_per_second=cost_optimizer_steps_per_second,
        peak_device_memory_bytes=resolved_peak_memory,
        digest=content_digest(payload_without_digest),
    )


def write_cost_critic_compute_evidence(
    path: Path,
    evidence: CostCriticComputeEvidence,
) -> None:
    """Persist evidence as canonical JSON without mutating the model."""

    if not isinstance(evidence, CostCriticComputeEvidence):
        raise TypeError("evidence must be CostCriticComputeEvidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(evidence.payload()))


__all__ = [
    "CostCriticComputeEvidence",
    "build_cost_critic_compute_evidence",
    "write_cost_critic_compute_evidence",
]
