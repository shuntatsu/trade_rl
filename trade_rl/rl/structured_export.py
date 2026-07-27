"""Canonical TorchScript export for structured hierarchical policy observations."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.rl.policy_identity import model_sb3_policy_identity

STRUCTURED_EXPORT_SCHEMA: Final = "structured_policy_export_v1"
STRUCTURED_EXPORT_MANIFEST_NAME: Final = "structured-export.json"
STRUCTURED_EXPORT_MODEL_NAME: Final = "policy.structured.torchscript.pt"
_TIMEFRAMES: Final = ("15m", "1h", "4h", "1d")
_BASE_KEYS: Final = ("current_snapshot", "asset_state", "global_state", "active")
_SEQUENCE_PLANES: Final = ("values", "available", "staleness")
_SUPPORTED_DTYPES: Final = frozenset({"float32", "float64", "int32", "int64", "bool"})


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
        policy_payload = dict(self.policy_identity)
        if content_digest(policy_payload) != self.policy_identity_digest:
            raise ValueError("structured export policy identity digest mismatch")
        if (
            policy_payload.get("observation_encoder")
            != "hierarchical_sequence_v2"
        ):
            raise ValueError("structured export requires hierarchical sequence policy")
        if (
            policy_payload.get("sequence_architecture_digest")
            != self.architecture_digest
        ):
            raise ValueError("structured export architecture digest mismatch")
        if tuple(item.name for item in self.inputs) != canonical_structured_observation_keys():
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
        policy_payload = dict(policy_identity)
        architecture_digest = policy_payload.get("sequence_architecture_digest")
        if not isinstance(architecture_digest, str):
            raise ValueError("structured export policy lacks architecture digest")
        payload = {
            "action_size": action_size,
            "architecture_digest": architecture_digest,
            "inputs": inputs,
            "max_abs_error": max_abs_error,
            "model_digest": _file_digest(model_path),
            "model_path": STRUCTURED_EXPORT_MODEL_NAME,
            "model_size_bytes": model_path.stat().st_size,
            "policy_identity": policy_payload,
            "policy_identity_digest": content_digest(policy_payload),
            "schema_version": STRUCTURED_EXPORT_SCHEMA,
            "tolerance": tolerance,
        }
        return cls(digest=content_digest(payload), **payload)


class _StructuredDeterministicActor(nn.Module):
    def __init__(self, policy: nn.Module, keys: tuple[str, ...]) -> None:
        super().__init__()
        self.policy = policy
        self.keys = keys

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        observation = {self.keys[index]: value for index, value in enumerate(inputs)}
        prediction = self.policy._predict(observation, deterministic=True)  # type: ignore[attr-defined]
        return prediction


def _observation_specs(model: object) -> tuple[StructuredInputSpec, ...]:
    policy = getattr(model, "policy", None)
    observation_space = getattr(policy, "observation_space", None)
    raw_spaces = getattr(observation_space, "spaces", None)
    if not isinstance(raw_spaces, Mapping):
        raise ValueError("structured export requires Dict observation space")
    expected = canonical_structured_observation_keys()
    if set(raw_spaces) != set(expected):
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
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }.get(name)
    if resolved is None:
        raise ValueError("unsupported structured tensor dtype")
    return resolved


def _validated_example(
    example_observation: Mapping[str, np.ndarray],
    specs: tuple[StructuredInputSpec, ...],
) -> tuple[torch.Tensor, ...]:
    if set(example_observation) != {item.name for item in specs}:
        raise ValueError("structured export example keys are invalid")
    tensors: list[torch.Tensor] = []
    batch_size: int | None = None
    for item in specs:
        value = np.asarray(example_observation[item.name], dtype=_numpy_dtype(item.dtype))
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
        elif specs[position].name.endswith("_available") or specs[position].name == "active":
            alternating.append(torch.ones_like(value))
        else:
            pattern = torch.arange(value.numel(), device=value.device).reshape(value.shape)
            alternating.append((pattern.remainder(2).to(value.dtype) * 2.0) - 1.0)
    return original, zeros, tuple(stressed), tuple(alternating)


def _actions(actor: nn.Module, inputs: Sequence[torch.Tensor], action_size: int) -> np.ndarray:
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
    """Export one hierarchical policy with canonical structured-input parity evidence."""

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
        raise TypeError("structured export model policy must be a torch module")
    policy = policy.to("cpu")
    set_training_mode = getattr(policy, "set_training_mode", None)
    if callable(set_training_mode):
        set_training_mode(False)
    policy.eval()
    specs = _observation_specs(model)
    example = _validated_example(example_observation, specs)
    actor = _StructuredDeterministicActor(policy, tuple(item.name for item in specs)).eval()
    corpus = _parity_corpus(example, specs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / STRUCTURED_EXPORT_MODEL_NAME
    manifest_path = output_dir / STRUCTURED_EXPORT_MANIFEST_NAME
    try:
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


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def load_structured_export_manifest(path: Path) -> StructuredExportManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = _mapping(raw, field="structured export manifest")
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
                name=str(item["name"]),
                shape=tuple(int(value) for value in raw_shape),
                dtype=str(item["dtype"]),
            )
        )
    policy_identity = _mapping(payload["policy_identity"], field="policy_identity")
    return StructuredExportManifest(
        digest=str(payload["digest"]),
        model_path=str(payload["model_path"]),
        model_digest=str(payload["model_digest"]),
        model_size_bytes=int(payload["model_size_bytes"]),
        policy_identity=dict(policy_identity),
        policy_identity_digest=str(payload["policy_identity_digest"]),
        architecture_digest=str(payload["architecture_digest"]),
        inputs=tuple(inputs),
        action_size=int(payload["action_size"]),
        tolerance=float(payload["tolerance"]),
        max_abs_error=float(payload["max_abs_error"]),
        schema_version=str(payload["schema_version"]),
    )


__all__ = [
    "STRUCTURED_EXPORT_MANIFEST_NAME",
    "STRUCTURED_EXPORT_MODEL_NAME",
    "STRUCTURED_EXPORT_SCHEMA",
    "StructuredExportManifest",
    "StructuredInputSpec",
    "canonical_structured_observation_keys",
    "export_structured_policy_actor",
    "load_structured_export_manifest",
]
