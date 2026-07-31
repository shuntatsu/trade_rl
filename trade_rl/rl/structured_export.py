"""Canonical TorchScript export for structured hierarchical policy observations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from trade_rl.artifacts.atomic_pointer import atomic_replace_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import (
    file_digest_and_size,
    open_regular_binary,
)
from trade_rl.domain.common import require_sha256
from trade_rl.rl.policy_identity import (
    model_sb3_policy_identity,
    validated_sb3_policy_identity,
)

STRUCTURED_EXPORT_SCHEMA: Final = "structured_policy_export_v2"
STRUCTURED_EXPORT_MANIFEST_NAME: Final = "structured-export.json"
STRUCTURED_EXPORT_MODEL_NAME: Final = "policy.structured.torchscript.pt"
_TIMEFRAMES: Final = ("15m", "1h", "4h", "1d")
_BASE_KEYS: Final = (
    "current_snapshot",
    "asset_state",
    "global_state",
    "active",
    "current_weights",
)
_SEQUENCE_PLANES: Final = ("values", "available", "staleness")
_TRAINING_ONLY_KEYS: Final = frozenset({"decision_index"})
_SUPPORTED_DTYPES: Final = frozenset(
    {"float16", "float32", "float64", "int32", "int64", "uint8", "bool"}
)


def _atomic_write(path: Path, payload: bytes) -> None:
    atomic_replace_bytes(path, payload)


def canonical_structured_observation_keys() -> tuple[str, ...]:
    keys = list(_BASE_KEYS)
    for timeframe in _TIMEFRAMES:
        for plane in _SEQUENCE_PLANES:
            keys.append(f"sequence_{timeframe}_{plane}")
    return tuple(keys)


@dataclass(frozen=True, slots=True)
class StructuredInputSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("structured input name must be non-empty")
        if not self.shape or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.shape
        ):
            raise ValueError("structured input shape must contain positive integers")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError("structured input dtype is unsupported")


@dataclass(frozen=True, slots=True)
class StructuredExportManifest:
    digest: str
    model_path: str
    model_digest: str
    model_size_bytes: int
    policy_identity: Mapping[str, object]
    policy_identity_digest: str
    architecture_digest: str
    inputs: tuple[StructuredInputSpec, ...]
    action_size: int
    tolerance: float
    max_abs_error: float
    schema_version: str = STRUCTURED_EXPORT_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.digest, field="structured_export.digest")
        require_sha256(self.model_digest, field="structured_export.model_digest")
        require_sha256(
            self.policy_identity_digest,
            field="structured_export.policy_identity_digest",
        )
        require_sha256(
            self.architecture_digest,
            field="structured_export.architecture_digest",
        )
        if self.schema_version != STRUCTURED_EXPORT_SCHEMA:
            raise ValueError("unsupported structured export schema")
        if self.model_path != STRUCTURED_EXPORT_MODEL_NAME:
            raise ValueError("structured export model path is invalid")
        if self.model_size_bytes <= 0:
            raise ValueError("structured export model must be non-empty")
        if not isinstance(self.policy_identity, Mapping) or not self.policy_identity:
            raise ValueError("structured export requires policy identity")
        policy_payload = validated_sb3_policy_identity(self.policy_identity)
        object.__setattr__(self, "policy_identity", policy_payload)
        if content_digest(policy_payload) != self.policy_identity_digest:
            raise ValueError("structured export policy identity digest mismatch")
        if policy_payload.get("observation_encoder") != "hierarchical_sequence_v2":
            raise ValueError("structured export requires hierarchical sequence policy")
        if policy_payload.get("policy_architecture_digest") != self.architecture_digest:
            raise ValueError("structured export architecture digest mismatch")
        if (
            tuple(item.name for item in self.inputs)
            != canonical_structured_observation_keys()
        ):
            raise ValueError("structured export input order is not canonical")
        if self.action_size <= 0:
            raise ValueError("structured export action_size must be positive")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("structured export tolerance must be finite and positive")
        if (
            not np.isfinite(self.max_abs_error)
            or self.max_abs_error < 0.0
            or self.max_abs_error > self.tolerance
        ):
            raise ValueError("structured export parity error exceeds tolerance")
        if self.digest != content_digest(self.digest_payload()):
            raise ValueError("structured export manifest digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_size": self.action_size,
            "architecture_digest": self.architecture_digest,
            "inputs": self.inputs,
            "max_abs_error": self.max_abs_error,
            "model_digest": self.model_digest,
            "model_path": self.model_path,
            "model_size_bytes": self.model_size_bytes,
            "policy_identity": dict(self.policy_identity),
            "policy_identity_digest": self.policy_identity_digest,
            "schema_version": self.schema_version,
            "tolerance": self.tolerance,
        }

    @classmethod
    def build(
        cls,
        *,
        model_path: Path,
        policy_identity: Mapping[str, object],
        inputs: tuple[StructuredInputSpec, ...],
        action_size: int,
        tolerance: float,
        max_abs_error: float,
    ) -> StructuredExportManifest:
        policy_payload = validated_sb3_policy_identity(policy_identity)
        architecture_digest = policy_payload.get("policy_architecture_digest")
        if not isinstance(architecture_digest, str):
            raise ValueError("structured export policy lacks architecture digest")
        model_digest, model_size_bytes = file_digest_and_size(
            model_path,
            field="structured export model",
        )
        policy_identity_digest = content_digest(policy_payload)
        payload = {
            "action_size": action_size,
            "architecture_digest": architecture_digest,
            "inputs": inputs,
            "max_abs_error": max_abs_error,
            "model_digest": model_digest,
            "model_path": STRUCTURED_EXPORT_MODEL_NAME,
            "model_size_bytes": model_size_bytes,
            "policy_identity": policy_payload,
            "policy_identity_digest": policy_identity_digest,
            "schema_version": STRUCTURED_EXPORT_SCHEMA,
            "tolerance": tolerance,
        }
        return cls(
            digest=content_digest(payload),
            model_path=STRUCTURED_EXPORT_MODEL_NAME,
            model_digest=model_digest,
            model_size_bytes=model_size_bytes,
            policy_identity=policy_payload,
            policy_identity_digest=policy_identity_digest,
            architecture_digest=architecture_digest,
            inputs=inputs,
            action_size=action_size,
            tolerance=tolerance,
            max_abs_error=max_abs_error,
            schema_version=STRUCTURED_EXPORT_SCHEMA,
        )


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
    for timeframe in _TIMEFRAMES:
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


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def load_structured_export_manifest_bytes(raw: bytes) -> StructuredExportManifest:
    """Parse one exact canonical structured-export manifest byte sequence."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("structured export manifest must be valid JSON") from error
    payload = _mapping(value, field="structured export manifest")
    expected = {
        "action_size",
        "architecture_digest",
        "digest",
        "inputs",
        "max_abs_error",
        "model_digest",
        "model_path",
        "model_size_bytes",
        "policy_identity",
        "policy_identity_digest",
        "schema_version",
        "tolerance",
    }
    if set(payload) != expected:
        raise ValueError("structured export manifest fields are invalid")
    raw_inputs = payload["inputs"]
    if not isinstance(raw_inputs, list):
        raise ValueError("structured export inputs must be a list")
    inputs: list[StructuredInputSpec] = []
    for index, raw_input in enumerate(raw_inputs):
        item = _mapping(raw_input, field=f"inputs[{index}]")
        if set(item) != {"dtype", "name", "shape"}:
            raise ValueError("structured input manifest fields are invalid")
        raw_shape = item["shape"]
        if not isinstance(raw_shape, list):
            raise ValueError("structured input shape must be a list")
        inputs.append(
            StructuredInputSpec(
                name=_string(item["name"], field=f"inputs[{index}].name"),
                shape=tuple(
                    _integer(value, field=f"inputs[{index}].shape")
                    for value in raw_shape
                ),
                dtype=_string(item["dtype"], field=f"inputs[{index}].dtype"),
            )
        )
    policy_identity = _mapping(payload["policy_identity"], field="policy_identity")
    manifest = StructuredExportManifest(
        digest=_string(payload["digest"], field="digest"),
        model_path=_string(payload["model_path"], field="model_path"),
        model_digest=_string(payload["model_digest"], field="model_digest"),
        model_size_bytes=_integer(
            payload["model_size_bytes"], field="model_size_bytes"
        ),
        policy_identity=dict(policy_identity),
        policy_identity_digest=_string(
            payload["policy_identity_digest"], field="policy_identity_digest"
        ),
        architecture_digest=_string(
            payload["architecture_digest"], field="architecture_digest"
        ),
        inputs=tuple(inputs),
        action_size=_integer(payload["action_size"], field="action_size"),
        tolerance=_number(payload["tolerance"], field="tolerance"),
        max_abs_error=_number(payload["max_abs_error"], field="max_abs_error"),
        schema_version=_string(payload["schema_version"], field="schema_version"),
    )
    if raw != canonical_json_bytes(manifest):
        raise ValueError("structured export manifest must use canonical encoding")
    return manifest


def load_structured_export_manifest(path: Path) -> StructuredExportManifest:
    with open_regular_binary(path, field="structured export manifest") as handle:
        return load_structured_export_manifest_bytes(handle.read())


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
