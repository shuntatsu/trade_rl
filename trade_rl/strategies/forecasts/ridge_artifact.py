"""Deterministic, caller-bound persistence for frozen Ridge forecasts."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl._validation import require_git_sha, require_sha256
from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.strategies.forecasts.ridge import RidgeForecastModel

_SCHEMA = "ridge_forecast_model_artifact_v1"
_DIGEST_DOMAIN = b"trade_rl.ridge_forecast_model_artifact_v1\0"
_PAYLOAD_NAME = "artifact.json"
_NAT_NS = int(np.iinfo(np.int64).min)
_MAX_NS = int(np.iinfo(np.int64).max)
_MAX_HORIZON_HOURS = _MAX_NS // 3_600_000_000_000


def _strict_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty canonical string")
    return value


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_int_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field} must be a non-empty tuple")
    result = tuple(_strict_int(item, field=field) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique values")
    return result


def _strict_string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field} must be a non-empty tuple")
    result = tuple(_strict_string(item, field=field) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique values")
    return result


def _parse_float_hex(value: object, *, field: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a canonical hexadecimal float")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a canonical hexadecimal float") from error
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise ValueError(f"{field} must be a canonical finite hexadecimal float")
    return parsed


@dataclass(frozen=True, slots=True)
class RidgeForecastArtifactBinding:
    """Declared fit and model context required to accept a Ridge artifact."""

    dataset_id: str
    dataset_schema_version: str
    dataset_artifact_sha256: str
    fit_symbol_indices: tuple[int, ...]
    fit_symbol_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...]
    fit_cutoff_ns: int
    horizon_hours: int
    alpha_hex: str
    n_samples: int
    vintage_id: str
    source_commit: str
    implementation_sha256: str
    runtime_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        _strict_string(self.dataset_schema_version, field="dataset_schema_version")
        require_sha256(self.dataset_artifact_sha256, field="dataset_artifact_sha256")
        symbol_indices = _strict_int_tuple(
            self.fit_symbol_indices,
            field="fit_symbol_indices",
        )
        symbol_names = _strict_string_tuple(
            self.fit_symbol_names,
            field="fit_symbol_names",
        )
        feature_names = _strict_string_tuple(self.feature_names, field="feature_names")
        feature_indices = _strict_int_tuple(
            self.feature_indices,
            field="feature_indices",
        )
        if len(symbol_indices) != len(symbol_names):
            raise ValueError("fit symbol indices and names must have the same length")
        if max(feature_indices) >= len(feature_names):
            raise ValueError("feature_indices must be inside feature_names")
        cutoff_ns = _strict_int(
            self.fit_cutoff_ns,
            field="fit_cutoff_ns",
            minimum=_NAT_NS + 1,
        )
        if cutoff_ns > _MAX_NS:
            raise ValueError("fit_cutoff_ns must fit signed datetime64[ns]")
        horizon = _strict_int(self.horizon_hours, field="horizon_hours", minimum=1)
        if horizon > _MAX_HORIZON_HOURS:
            raise ValueError("horizon_hours must fit signed datetime64[ns]")
        alpha = _parse_float_hex(self.alpha_hex, field="alpha_hex")
        if alpha <= 0.0:
            raise ValueError("alpha_hex must be positive")
        _strict_int(self.n_samples, field="n_samples", minimum=1)
        _strict_string(self.vintage_id, field="vintage_id")
        require_git_sha(self.source_commit, field="source_commit")
        require_sha256(self.implementation_sha256, field="implementation_sha256")
        require_sha256(self.runtime_sha256, field="runtime_sha256")

        object.__setattr__(self, "fit_symbol_indices", symbol_indices)
        object.__setattr__(self, "fit_symbol_names", symbol_names)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "feature_indices", feature_indices)


def _binding_payload(binding: RidgeForecastArtifactBinding) -> dict[str, object]:
    return {
        "dataset_id": binding.dataset_id,
        "dataset_schema_version": binding.dataset_schema_version,
        "dataset_artifact_sha256": binding.dataset_artifact_sha256,
        "fit_symbol_indices": list(binding.fit_symbol_indices),
        "fit_symbol_names": list(binding.fit_symbol_names),
        "feature_names": list(binding.feature_names),
        "feature_indices": list(binding.feature_indices),
        "fit_cutoff_ns": binding.fit_cutoff_ns,
        "horizon_hours": binding.horizon_hours,
        "alpha_hex": binding.alpha_hex,
        "n_samples": binding.n_samples,
        "vintage_id": binding.vintage_id,
        "source_commit": binding.source_commit,
        "implementation_sha256": binding.implementation_sha256,
        "runtime_sha256": binding.runtime_sha256,
    }


def _model_payload(
    model: RidgeForecastModel,
    binding: RidgeForecastArtifactBinding,
) -> dict[str, object]:
    if type(model) is not RidgeForecastModel:
        raise TypeError("model must be a RidgeForecastModel")
    if model.feature_indices != binding.feature_indices:
        raise ValueError("model feature indices differ from artifact binding")
    if model.horizon_hours != binding.horizon_hours:
        raise ValueError("model horizon differs from artifact binding")
    if model.n_samples != binding.n_samples:
        raise ValueError("model sample count differs from artifact binding")
    if isinstance(model.alpha, bool) or not isinstance(model.alpha, (int, float)):
        raise ValueError("model alpha must be a built-in numeric value")
    alpha_hex = float(model.alpha).hex()
    if alpha_hex != binding.alpha_hex:
        raise ValueError("model alpha differs from artifact binding")
    cutoff_ns = int(
        np.asarray(model.fit_cutoff).astype("datetime64[ns]").astype(np.int64)
    )
    if cutoff_ns == _NAT_NS or cutoff_ns != binding.fit_cutoff_ns:
        raise ValueError("model fit cutoff differs from artifact binding")

    arrays: dict[str, list[str]] = {}
    for name in ("feature_mean", "feature_scale", "coefficients"):
        values = np.asarray(getattr(model, name))
        if values.dtype != np.dtype(np.float64) or values.shape != (
            len(binding.feature_indices),
        ):
            raise ValueError(f"model {name} shape or dtype is invalid")
        if not np.isfinite(values).all():
            raise ValueError(f"model {name} must be finite")
        if name == "feature_scale" and np.any(values <= 0.0):
            raise ValueError("model feature scales must be positive")
        arrays[name] = [float(value).hex() for value in values]
    intercept = float(model.intercept)
    if isinstance(model.intercept, bool) or not math.isfinite(intercept):
        raise ValueError("model intercept must be a finite non-boolean number")

    return {
        **{f"{name}_hex": value for name, value in arrays.items()},
        "intercept_hex": intercept.hex(),
    }


def _payload_bytes(
    model: RidgeForecastModel,
    binding: RidgeForecastArtifactBinding,
) -> bytes:
    payload = {
        "schema": _SCHEMA,
        "binding": _binding_payload(binding),
        "model": _model_payload(model, binding),
    }
    return canonical_json_bytes(payload)


def _artifact_digest(payload: bytes) -> str:
    return hashlib.sha256(_DIGEST_DOMAIN + payload).hexdigest()


def _read_artifact_payload(root: Path) -> bytes:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Ridge artifact must be a regular directory")
    entries = tuple(root.iterdir())
    if len(entries) != 1 or entries[0].name != _PAYLOAD_NAME:
        raise ValueError("Ridge artifact directory has unexpected files")
    payload_path = entries[0]
    if payload_path.is_symlink() or not payload_path.is_file():
        raise ValueError("Ridge artifact payload must be a regular file")
    with open_regular_binary(payload_path, field="Ridge artifact payload") as stream:
        return stream.read()


def _publish_existing_identical(
    root: Path,
    expected_payload: bytes,
    expected_digest: str,
) -> str:
    actual_payload = _read_artifact_payload(root)
    if (
        actual_payload != expected_payload
        or _artifact_digest(actual_payload) != expected_digest
    ):
        raise FileExistsError(f"Ridge artifact destination already differs: {root}")
    return expected_digest


def publish_ridge_forecast_artifact(
    root: str | Path,
    model: RidgeForecastModel,
    binding: RidgeForecastArtifactBinding,
) -> str:
    """Atomically publish one immutable Ridge model artifact without overwrite."""

    if type(binding) is not RidgeForecastArtifactBinding:
        raise TypeError("binding must be a RidgeForecastArtifactBinding")
    destination = Path(root)
    payload = _payload_bytes(model, binding)
    digest = _artifact_digest(payload)
    if destination.exists() or destination.is_symlink():
        return _publish_existing_identical(destination, payload, digest)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=str(destination.parent),
        )
    )
    try:
        atomic_write_bytes(staging / _PAYLOAD_NAME, payload)
        try:
            atomic_rename_directory(staging, destination)
        except FileExistsError:
            return _publish_existing_identical(destination, payload, digest)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return digest


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_keys(value: object, expected: set[str], *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{field} keys differ from the supported schema")
    return value


def _list_of_ints(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return tuple(_strict_int(item, field=field) for item in value)


def _list_of_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return tuple(_strict_string(item, field=field) for item in value)


def _binding_from_payload(value: object) -> RidgeForecastArtifactBinding:
    binding = _require_keys(
        value,
        {
            "dataset_id",
            "dataset_schema_version",
            "dataset_artifact_sha256",
            "fit_symbol_indices",
            "fit_symbol_names",
            "feature_names",
            "feature_indices",
            "fit_cutoff_ns",
            "horizon_hours",
            "alpha_hex",
            "n_samples",
            "vintage_id",
            "source_commit",
            "implementation_sha256",
            "runtime_sha256",
        },
        field="binding",
    )
    return RidgeForecastArtifactBinding(
        dataset_id=_strict_string(binding["dataset_id"], field="dataset_id"),
        dataset_schema_version=_strict_string(
            binding["dataset_schema_version"],
            field="dataset_schema_version",
        ),
        dataset_artifact_sha256=_strict_string(
            binding["dataset_artifact_sha256"],
            field="dataset_artifact_sha256",
        ),
        fit_symbol_indices=_list_of_ints(
            binding["fit_symbol_indices"],
            field="fit_symbol_indices",
        ),
        fit_symbol_names=_list_of_strings(
            binding["fit_symbol_names"],
            field="fit_symbol_names",
        ),
        feature_names=_list_of_strings(binding["feature_names"], field="feature_names"),
        feature_indices=_list_of_ints(
            binding["feature_indices"],
            field="feature_indices",
        ),
        fit_cutoff_ns=_strict_int(
            binding["fit_cutoff_ns"],
            field="fit_cutoff_ns",
            minimum=_NAT_NS + 1,
        ),
        horizon_hours=_strict_int(
            binding["horizon_hours"],
            field="horizon_hours",
            minimum=1,
        ),
        alpha_hex=_strict_string(binding["alpha_hex"], field="alpha_hex"),
        n_samples=_strict_int(binding["n_samples"], field="n_samples", minimum=1),
        vintage_id=_strict_string(binding["vintage_id"], field="vintage_id"),
        source_commit=_strict_string(binding["source_commit"], field="source_commit"),
        implementation_sha256=_strict_string(
            binding["implementation_sha256"],
            field="implementation_sha256",
        ),
        runtime_sha256=_strict_string(
            binding["runtime_sha256"], field="runtime_sha256"
        ),
    )


def _model_from_payload(
    value: object,
    binding: RidgeForecastArtifactBinding,
) -> RidgeForecastModel:
    model = _require_keys(
        value,
        {
            "feature_mean_hex",
            "feature_scale_hex",
            "coefficients_hex",
            "intercept_hex",
        },
        field="model",
    )
    indices = binding.feature_indices
    vectors: dict[str, np.ndarray] = {}
    for name in ("feature_mean", "feature_scale", "coefficients"):
        raw_values = model[f"{name}_hex"]
        if not isinstance(raw_values, list) or len(raw_values) != len(indices):
            raise ValueError(f"model {name} shape differs from feature indices")
        parsed = [_parse_float_hex(item, field=f"{name}_hex") for item in raw_values]
        vectors[name] = np.asarray(parsed, dtype=np.float64)
    intercept = _parse_float_hex(model["intercept_hex"], field="intercept_hex")
    if np.any(vectors["feature_scale"] <= 0.0):
        raise ValueError("model feature scales must be positive")
    return RidgeForecastModel(
        feature_indices=indices,
        feature_mean=vectors["feature_mean"],
        feature_scale=vectors["feature_scale"],
        coefficients=vectors["coefficients"],
        intercept=intercept,
        horizon_hours=binding.horizon_hours,
        alpha=_parse_float_hex(binding.alpha_hex, field="alpha_hex"),
        n_samples=binding.n_samples,
        fit_cutoff=np.datetime64(binding.fit_cutoff_ns, "ns"),
    )


def load_ridge_forecast_artifact(
    root: str | Path,
    *,
    expected_digest: str,
    expected_binding: RidgeForecastArtifactBinding,
) -> RidgeForecastModel:
    """Load only a canonical artifact matching an external digest and context."""

    require_sha256(expected_digest, field="expected_digest")
    if type(expected_binding) is not RidgeForecastArtifactBinding:
        raise TypeError("expected_binding must be a RidgeForecastArtifactBinding")
    payload = _read_artifact_payload(Path(root))
    if _artifact_digest(payload) != expected_digest:
        raise ValueError("Ridge artifact differs from its externally expected digest")
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Ridge artifact JSON is invalid") from error
    top = _require_keys(decoded, {"schema", "binding", "model"}, field="artifact")
    if top["schema"] != _SCHEMA:
        raise ValueError("Ridge artifact schema is unsupported")
    if canonical_json_bytes(decoded) != payload:
        raise ValueError("Ridge artifact JSON is not canonical")
    binding = _binding_from_payload(top["binding"])
    if binding != expected_binding:
        raise ValueError("Ridge artifact context differs from caller expectations")
    model = _model_from_payload(top["model"], binding)
    return model


__all__ = [
    "RidgeForecastArtifactBinding",
    "load_ridge_forecast_artifact",
    "publish_ridge_forecast_artifact",
]
