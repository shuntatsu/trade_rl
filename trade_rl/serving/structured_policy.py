"""Canonical fail-closed loader for structured hierarchical TorchScript exports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trade_rl.artifacts.verified_file import file_digest, verified_private_copy
from trade_rl.domain.common import require_sha256
from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA
from trade_rl.rl.structured_export import (
    STRUCTURED_EXPORT_MANIFEST_NAME,
    StructuredExportManifest,
    StructuredInputSpec,
    load_structured_export_manifest,
)
from trade_rl.serving.bundle import ServingBundle


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
        raise ValueError("unsupported structured serving dtype")
    return resolved


def _numpy_dtype(name: str) -> np.dtype[Any]:
    return np.dtype(name)


class StructuredTorchScriptPolicy:
    """Single-observation policy wrapper with exact key/shape/dtype validation."""

    def __init__(self, *, root: Path, manifest: StructuredExportManifest) -> None:
        self.root = Path(root)
        self.manifest = manifest
        model_path = self.root / manifest.model_path
        if not model_path.is_file():
            raise FileNotFoundError("structured policy model is missing")
        if model_path.stat().st_size != manifest.model_size_bytes:
            raise ValueError("structured policy model size mismatch")
        if file_digest(model_path, field="structured policy model") != manifest.model_digest:
            raise ValueError("structured policy model digest mismatch")
        with verified_private_copy(
            model_path,
            expected_digest=manifest.model_digest,
            expected_size_bytes=manifest.model_size_bytes,
            field="structured policy model",
            filename=model_path.name,
        ) as verified_model_path:
            self.model = torch.jit.load(
                str(verified_model_path),
                map_location="cpu",
            ).eval()
        self._by_name = {item.name: item for item in manifest.inputs}

    def _tensor(self, value: np.ndarray, spec: StructuredInputSpec) -> torch.Tensor:
        array = np.asarray(value, dtype=_numpy_dtype(spec.dtype))
        if array.shape != spec.shape:
            raise ValueError(f"structured observation shape mismatch: {spec.name}")
        if not np.isfinite(array).all():
            raise ValueError(f"structured observation must be finite: {spec.name}")
        return torch.as_tensor(
            array.reshape((1, *spec.shape)), dtype=_torch_dtype(spec.dtype)
        )

    def predict(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        if set(observation) != set(self._by_name):
            raise ValueError("structured observation keys do not match export contract")
        inputs = tuple(
            self._tensor(np.asarray(observation[item.name]), item)
            for item in self.manifest.inputs
        )
        with torch.no_grad():
            raw = self.model(*inputs)
        action = raw.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        if (
            action.shape != (self.manifest.action_size,)
            or not np.isfinite(action).all()
        ):
            raise ValueError("structured policy output violates action contract")
        return action.copy()

    def smoke_observation(self) -> dict[str, np.ndarray]:
        observation: dict[str, np.ndarray] = {}
        for item in self.manifest.inputs:
            dtype = _numpy_dtype(item.dtype)
            if item.name == "active" or item.name.endswith("_available"):
                value = np.ones(item.shape, dtype=dtype)
            else:
                value = np.zeros(item.shape, dtype=dtype)
            observation[item.name] = value
        return observation


class CanonicalStructuredPolicyLoader:
    """Serving PolicyLoader that enforces bundle and architecture identity."""

    def __init__(self, *, expected_architecture_digest: str) -> None:
        require_sha256(
            expected_architecture_digest,
            field="expected_architecture_digest",
        )
        self.expected_architecture_digest = expected_architecture_digest

    def load(self, bundle: ServingBundle) -> StructuredTorchScriptPolicy:
        if bundle.manifest.observation_schema != SEQUENCE_OBSERVATION_SCHEMA:
            raise ValueError(
                "structured policy loader requires sequence observation schema"
            )
        manifest_path = bundle.root / STRUCTURED_EXPORT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError("structured export manifest is missing from bundle")
        bundle_files = {item.path: item for item in bundle.manifest.files}
        manifest_file = bundle_files.get(STRUCTURED_EXPORT_MANIFEST_NAME)
        if manifest_file is None:
            raise ValueError(
                "structured export manifest is not bound to serving bundle"
            )
        if manifest_path.stat().st_size != manifest_file.size_bytes:
            raise ValueError("structured export manifest size mismatch")
        if file_digest(
            manifest_path, field="structured export manifest"
        ) != manifest_file.digest:
            raise ValueError("structured export manifest digest mismatch")
        manifest = load_structured_export_manifest(manifest_path)
        if manifest.architecture_digest != self.expected_architecture_digest:
            raise ValueError(
                "structured policy architecture does not match serving runtime"
            )
        model_file = bundle_files.get(manifest.model_path)
        if model_file is None:
            raise ValueError("structured policy model is not bound to serving bundle")
        model_path = bundle.root / manifest.model_path
        if not model_path.is_file():
            raise FileNotFoundError("structured policy model is missing from bundle")
        if model_path.stat().st_size != model_file.size_bytes:
            raise ValueError("structured bundle model size mismatch")
        if file_digest(
            model_path, field="structured bundle model"
        ) != model_file.digest:
            raise ValueError("structured bundle model digest mismatch")
        return StructuredTorchScriptPolicy(root=bundle.root, manifest=manifest)


__all__ = ["CanonicalStructuredPolicyLoader", "StructuredTorchScriptPolicy"]
