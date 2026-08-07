"""Canonical TorchScript export for structured hierarchical policy observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from trade_rl.artifacts.atomic_pointer import atomic_replace_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.structured_policy_contract import (
    STRUCTURED_EXPORT_MANIFEST_NAME,
    STRUCTURED_EXPORT_MODEL_NAME,
    STRUCTURED_EXPORT_SCHEMA,
    STRUCTURED_TIMEFRAMES,
    StructuredExportManifest,
    StructuredInputSpec,
    canonical_structured_observation_keys,
    load_structured_export_manifest,
    load_structured_export_manifest_bytes,
)
from trade_rl.rl.policy_identity import (
    model_sb3_policy_identity,
)

_TRAINING_ONLY_KEYS: Final = frozenset({"decision_index"})


def _atomic_write(path: Path, payload: bytes) -> None:
    atomic_replace_bytes(path, payload)


class _StructuredDeterministicActor(nn.Module):
    def __init__(
        self,
        policy: Any,
        keys: tuple[str, ...],
        *,
        synthesize_decision_index: bool,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.keys = keys
        self.synthesize_decision_index = synthesize_decision_index
        self.use_direct_actions = callable(
            getattr(policy, "deterministic_actions", None)
        )

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        observation = {self.keys[index]: value for index, value in enumerate(inputs)}
        if self.synthesize_decision_index:
            observation["decision_index"] = torch.zeros(
                (inputs[0].shape[0], 1),
                dtype=torch.int64,
                device=inputs[0].device,
            )
        if self.use_direct_actions:
            return self.policy.deterministic_actions(observation)
        return self.policy._predict(observation, deterministic=True)


def _observation_specs(model: object) -> tuple[StructuredInputSpec, ...]:
    policy = getattr(model, "policy", None)
    observation_space = getattr(policy, "observation_space", None)
    raw_spaces = getattr(observation_space, "spaces", None)
    if not isinstance(raw_spaces, Mapping):
        raise ValueError("structured export requires Dict observation space")
    expected = canonical_structured_observation_keys()
    extra = set(raw_spaces) - set(expected)
    if not set(expected).issubset(raw_spaces) or not extra.issubset(
        _TRAINING_ONLY_KEYS
    ):
        raise ValueError("structured observation keys do not match canonical contract")
    specs: list[StructuredInputSpec] = []
    for key in expected:
        space = raw_spaces[key]
        raw_shape = getattr(space, "shape", None)
        raw_dtype = getattr(space, "dtype", None)
        if raw_shape is None or raw_dtype is None:
            raise ValueError("structured observation space lacks shape or dtype")
        specs.append(
            StructuredInputSpec(
                name=key,
                shape=tuple(int(value) for value in raw_shape),
                dtype=np.dtype(raw_dtype).name,
            )
        )
    return tuple(specs)


def _numpy_dtype(name: str) -> np.dtype[Any]:
    return np.dtype(name)


def _torch_dtype(name: str) -> torch.dtype:
    resolved = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }.get(name)
    if resolved is None:
        raise ValueError("unsupported structured tensor dtype")
    return resolved


def _validated_example(
    example_observation: Mapping[str, np.ndarray],
    specs: tuple[StructuredInputSpec, ...],
) -> tuple[torch.Tensor, ...]:
    expected = {item.name for item in specs}
    extra = set(example_observation) - expected
    if not expected.issubset(example_observation) or not extra.issubset(
        _TRAINING_ONLY_KEYS
    ):
        raise ValueError("structured export example keys are invalid")
    tensors: list[torch.Tensor] = []
    batch_size: int | None = None
    for item in specs:
        value = np.asarray(
            example_observation[item.name], dtype=_numpy_dtype(item.dtype)
        )
        if value.shape == item.shape:
            value = value.reshape((1, *item.shape))
        if value.ndim != len(item.shape) + 1 or value.shape[1:] != item.shape:
            raise ValueError(f"structured export example shape mismatch: {item.name}")
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif value.shape[0] != batch_size:
            raise ValueError("structured export example batch dimensions disagree")
        if not np.isfinite(value).all():
            raise ValueError("structured export example must be finite")
        tensors.append(torch.as_tensor(value, dtype=_torch_dtype(item.dtype)))
    return tuple(tensors)


def _parity_corpus(
    example: tuple[torch.Tensor, ...],
    specs: tuple[StructuredInputSpec, ...],
) -> tuple[tuple[torch.Tensor, ...], ...]:
    original = tuple(value.detach().clone() for value in example)
    zeros = tuple(torch.zeros_like(value) for value in example)
    stressed = [value.detach().clone() for value in example]
    index = {item.name: position for position, item in enumerate(specs)}
    for timeframe in STRUCTURED_TIMEFRAMES:
        available_position = index[f"sequence_{timeframe}_available"]
        staleness_position = index[f"sequence_{timeframe}_staleness"]
        stressed[available_position][:, -1].zero_()
        stressed[staleness_position][:, -1].fill_(100.0)
    stressed[index["active"]][:, -1].zero_()
    alternating: list[torch.Tensor] = []
    for position, value in enumerate(example):
        if value.dtype == torch.bool:
            alternating.append(torch.ones_like(value))
        elif (
            specs[position].name.endswith("_available")
            or specs[position].name == "active"
        ):
            alternating.append(torch.ones_like(value))
        else:
            pattern = torch.arange(value.numel(), device=value.device).reshape(
                value.shape
            )
            alternating.append((pattern.remainder(2).to(value.dtype) * 2.0) - 1.0)
    return original, zeros, tuple(stressed), tuple(alternating)


def _actions(
    actor: nn.Module, inputs: Sequence[torch.Tensor], action_size: int
) -> np.ndarray:
    with torch.no_grad():
        output = actor(*inputs)
    resolved = output.detach().cpu().numpy().astype(np.float32, copy=False)
    resolved = resolved.reshape(inputs[0].shape[0], -1)
    if resolved.shape[1] != action_size or not np.isfinite(resolved).all():
        raise ValueError("structured actor output violates action contract")
    return resolved


def export_structured_policy_actor(
    *,
    model: object,
    output_dir: Path,
    example_observation: Mapping[str, np.ndarray],
    action_size: int,
    tolerance: float = 1e-5,
) -> StructuredExportManifest:
    """Export one hierarchical policy while preserving the live model state."""

    if action_size <= 0:
        raise ValueError("structured export action_size must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("structured export tolerance must be finite and positive")
    identity = model_sb3_policy_identity(model)
    if identity is None:
        raise ValueError("structured export requires bound policy identity")
    if identity.get("observation_encoder") != "hierarchical_sequence_v2":
        raise ValueError("structured export requires hierarchical sequence policy")
    policy = getattr(model, "policy", None)
    if not isinstance(policy, nn.Module):
        raise TypeError("structured export policy must be a torch module")
    has_direct_actions = callable(getattr(policy, "deterministic_actions", None))
    has_predict = callable(getattr(policy, "_predict", None))
    if not has_direct_actions and not has_predict:
        raise TypeError(
            "structured export policy must expose deterministic_actions or _predict"
        )

    original_training = bool(policy.training)
    original_device = getattr(model, "device", None)
    if original_device is None:
        first_parameter = next(policy.parameters(), None)
        original_device = "cpu" if first_parameter is None else first_parameter.device
    set_training_mode = getattr(policy, "set_training_mode", None)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / STRUCTURED_EXPORT_MODEL_NAME
    manifest_path = output_dir / STRUCTURED_EXPORT_MANIFEST_NAME
    try:
        policy.to("cpu")
        if callable(set_training_mode):
            set_training_mode(False)
        else:
            policy.train(False)
        policy.eval()
        specs = _observation_specs(model)
        example = _validated_example(example_observation, specs)
        actor = _StructuredDeterministicActor(
            policy,
            tuple(item.name for item in specs),
            synthesize_decision_index=(
                "decision_index" in getattr(policy.observation_space, "spaces", {})
            ),
        ).eval()
        corpus = _parity_corpus(example, specs)
        with torch.no_grad():
            traced = torch.jit.trace(actor, example, strict=False, check_trace=False)
            traced.save(str(model_path))
            restored = torch.jit.load(str(model_path), map_location="cpu").eval()
        max_error = 0.0
        for inputs in corpus:
            expected = _actions(actor, inputs, action_size)
            actual = _actions(restored, inputs, action_size)
            max_error = max(
                max_error,
                float(np.max(np.abs(expected - actual), initial=0.0)),
            )
        if max_error > tolerance:
            raise ValueError(
                f"structured TorchScript parity error {max_error} exceeds {tolerance}"
            )
        manifest = StructuredExportManifest.build(
            model_path=model_path,
            policy_identity=identity,
            inputs=specs,
            action_size=action_size,
            tolerance=tolerance,
            max_abs_error=max_error,
        )
        _atomic_write(manifest_path, canonical_json_bytes(manifest))
        return manifest
    except Exception:
        model_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    finally:
        try:
            policy.to(original_device)
            if callable(set_training_mode):
                set_training_mode(original_training)
            else:
                policy.train(original_training)
        except Exception:
            model_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            raise


__all__ = [
    "STRUCTURED_EXPORT_MANIFEST_NAME",
    "STRUCTURED_EXPORT_MODEL_NAME",
    "STRUCTURED_EXPORT_SCHEMA",
    "StructuredExportManifest",
    "StructuredInputSpec",
    "canonical_structured_observation_keys",
    "export_structured_policy_actor",
    "load_structured_export_manifest",
    "load_structured_export_manifest_bytes",
]
