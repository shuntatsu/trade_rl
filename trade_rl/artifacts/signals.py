"""Canonical content-addressed point-in-time alpha and factor artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

SIGNAL_MANIFEST_NAME: Final = "manifest.json"
SIGNAL_ARRAYS_NAME: Final = "arrays.npz"
SIGNAL_ARTIFACT_SCHEMA: Final = "causal_signal_array_artifact_v2"
_FIXED_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
SignalKind = Literal["alpha", "factor"]
_ALLOWED_FILES = frozenset({SIGNAL_MANIFEST_NAME, SIGNAL_ARRAYS_NAME})
_LEGACY_GENERATOR_CONFIG = content_digest({"schema": "legacy_signal_generator_config_v1"})
_LEGACY_GENERATOR_CODE = content_digest({"schema": "legacy_signal_generator_code_v1"})


@dataclass(frozen=True, slots=True)
class SignalArrayManifest:
    artifact_digest: str
    arrays_digest: str
    dataset_id: str
    fit_start: int
    fit_stop: int
    prediction_start: int
    prediction_stop: int
    generator_config_digest: str
    generator_code_digest: str
    kind: SignalKind
    names: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    available_at_dtype: str
    schema_version: str = SIGNAL_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        for field_name, value in (
            ("artifact_digest", self.artifact_digest),
            ("arrays_digest", self.arrays_digest),
            ("dataset_id", self.dataset_id),
            ("generator_config_digest", self.generator_config_digest),
            ("generator_code_digest", self.generator_code_digest),
        ):
            require_sha256(value, field=field_name)
        if self.kind not in {"alpha", "factor"}:
            raise ValueError("signal artifact kind is unsupported")
        if self.fit_start < 0 or self.fit_stop <= self.fit_start:
            raise ValueError("signal fit range must be non-empty and half-open")
        if self.prediction_start < self.fit_stop:
            raise ValueError("prediction range must start at or after fit_stop")
        if self.prediction_stop <= self.prediction_start:
            raise ValueError("signal prediction range must be non-empty and half-open")
        if not self.names or any(not name for name in self.names):
            raise ValueError("signal names must be non-empty")
        if len(set(self.names)) != len(self.names):
            raise ValueError("signal names must be unique")
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError("signal shape must be positive")
        if self.prediction_stop > self.shape[0]:
            raise ValueError("signal prediction range exceeds values")
        if self.schema_version != SIGNAL_ARTIFACT_SCHEMA:
            raise ValueError("unsupported signal artifact schema")

    def digest_payload(self) -> dict[str, object]:
        return {
            "arrays_digest": self.arrays_digest,
            "arrays_file": SIGNAL_ARRAYS_NAME,
            "available_at_dtype": self.available_at_dtype,
            "dataset_id": self.dataset_id,
            "dtype": self.dtype,
            "fit_start": self.fit_start,
            "fit_stop": self.fit_stop,
            "generator_code_digest": self.generator_code_digest,
            "generator_config_digest": self.generator_config_digest,
            "kind": self.kind,
            "names": self.names,
            "prediction_start": self.prediction_start,
            "prediction_stop": self.prediction_stop,
            "schema_version": self.schema_version,
            "shape": self.shape,
        }


@dataclass(frozen=True, slots=True)
class SignalArrays:
    values: np.ndarray
    valid: np.ndarray
    available_at: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def dtype(self) -> np.dtype[np.generic]:
        return self.values.dtype

    def __getitem__(self, item: object) -> np.ndarray:
        return self.values[item]

    def __array__(self, dtype: np.dtype[np.generic] | None = None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))
    return output.getvalue()


def _verify_file_closure(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"signal artifact directory is missing: {root}")
    entries = tuple(root.iterdir())
    names = {entry.name for entry in entries}
    if names != _ALLOWED_FILES or any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("signal artifact file closure mismatch")


def _validate_arrays(
    *,
    kind: SignalKind,
    names: tuple[str, ...],
    values: np.ndarray,
    valid: np.ndarray | None,
    available_at: np.ndarray | None,
) -> SignalArrays:
    array = np.asarray(values)
    if array.ndim not in {2, 3} or not np.issubdtype(array.dtype, np.number):
        raise ValueError("signal values must be a numeric rank-2 or rank-3 array")
    if not np.isfinite(array).all():
        raise ValueError("signal values must be finite")
    if kind == "alpha" and (array.ndim != 2 or array.shape[1] != len(names)):
        raise ValueError("alpha values must have shape (bars, symbols)")
    if kind == "factor" and (array.ndim != 3 or array.shape[1] != len(names)):
        raise ValueError("factor values must have shape (bars, factors, symbols)")
    validity = np.ones(array.shape, dtype=np.bool_) if valid is None else np.asarray(valid, dtype=np.bool_)
    if validity.shape != array.shape:
        raise ValueError("signal validity shape must match values")
    availability = (
        np.arange(array.shape[0], dtype=np.int64)
        if available_at is None
        else np.asarray(available_at)
    )
    if availability.shape != (array.shape[0],):
        raise ValueError("signal available_at must have one value per bar")
    if not (
        np.issubdtype(availability.dtype, np.datetime64)
        or np.issubdtype(availability.dtype, np.integer)
    ):
        raise ValueError("signal available_at must use datetime64 or integer indices")
    if np.issubdtype(availability.dtype, np.datetime64):
        availability = availability.astype("datetime64[ns]")
        if np.any(availability.astype(np.int64) == np.iinfo(np.int64).min):
            raise ValueError("signal available_at must not contain NaT")
    else:
        availability = availability.astype(np.int64)
        if np.any(availability < 0):
            raise ValueError("signal available_at indices must be non-negative")
    return SignalArrays(array.copy(), validity.copy(), availability.copy())


def write_signal_artifact(
    root: str | Path,
    *,
    kind: SignalKind,
    dataset_id: str,
    fit_start: int,
    fit_stop: int,
    names: tuple[str, ...],
    values: np.ndarray,
    prediction_start: int | None = None,
    prediction_stop: int | None = None,
    generator_config_digest: str = _LEGACY_GENERATOR_CONFIG,
    generator_code_digest: str = _LEGACY_GENERATOR_CODE,
    valid: np.ndarray | None = None,
    available_at: np.ndarray | None = None,
) -> str:
    arrays = _validate_arrays(
        kind=kind,
        names=names,
        values=values,
        valid=valid,
        available_at=available_at,
    )
    resolved_prediction_start = fit_stop if prediction_start is None else prediction_start
    resolved_prediction_stop = arrays.shape[0] if prediction_stop is None else prediction_stop
    output = Path(root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"signal artifact destination is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    payload = _deterministic_npz(
        {
            "available_at": arrays.available_at,
            "valid": arrays.valid,
            "values": arrays.values,
        }
    )
    arrays_digest = _sha256(payload)
    base = {
        "arrays_digest": arrays_digest,
        "arrays_file": SIGNAL_ARRAYS_NAME,
        "available_at_dtype": arrays.available_at.dtype.str,
        "dataset_id": dataset_id,
        "dtype": arrays.values.dtype.str,
        "fit_start": fit_start,
        "fit_stop": fit_stop,
        "generator_code_digest": generator_code_digest,
        "generator_config_digest": generator_config_digest,
        "kind": kind,
        "names": names,
        "prediction_start": resolved_prediction_start,
        "prediction_stop": resolved_prediction_stop,
        "schema_version": SIGNAL_ARTIFACT_SCHEMA,
        "shape": tuple(int(size) for size in arrays.shape),
    }
    manifest = SignalArrayManifest(
        artifact_digest=content_digest(base),
        arrays_digest=arrays_digest,
        dataset_id=dataset_id,
        fit_start=fit_start,
        fit_stop=fit_stop,
        prediction_start=resolved_prediction_start,
        prediction_stop=resolved_prediction_stop,
        generator_config_digest=generator_config_digest,
        generator_code_digest=generator_code_digest,
        kind=kind,
        names=names,
        shape=tuple(int(size) for size in arrays.shape),
        dtype=arrays.values.dtype.str,
        available_at_dtype=arrays.available_at.dtype.str,
    )
    _atomic_write(output / SIGNAL_ARRAYS_NAME, payload)
    _atomic_write(output / SIGNAL_MANIFEST_NAME, canonical_json_bytes(manifest))
    return manifest.artifact_digest


def load_signal_artifact(
    root: str | Path,
    *,
    expected_kind: SignalKind | None = None,
) -> tuple[SignalArrayManifest, SignalArrays]:
    path = Path(root)
    _verify_file_closure(path)
    raw = json.loads((path / SIGNAL_MANIFEST_NAME).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("signal manifest must be a mapping")
    try:
        manifest = SignalArrayManifest(
            artifact_digest=str(raw["artifact_digest"]),
            arrays_digest=str(raw["arrays_digest"]),
            dataset_id=str(raw["dataset_id"]),
            fit_start=int(raw["fit_start"]),
            fit_stop=int(raw["fit_stop"]),
            prediction_start=int(raw["prediction_start"]),
            prediction_stop=int(raw["prediction_stop"]),
            generator_config_digest=str(raw["generator_config_digest"]),
            generator_code_digest=str(raw["generator_code_digest"]),
            kind=str(raw["kind"]),  # type: ignore[arg-type]
            names=tuple(str(value) for value in raw["names"]),
            shape=tuple(int(value) for value in raw["shape"]),
            dtype=str(raw["dtype"]),
            available_at_dtype=str(raw["available_at_dtype"]),
            schema_version=str(raw["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("signal manifest is invalid") from error
    if content_digest(manifest.digest_payload()) != manifest.artifact_digest:
        raise ValueError("signal manifest digest mismatch")
    if expected_kind is not None and manifest.kind != expected_kind:
        raise ValueError("signal artifact kind mismatch")
    payload = (path / SIGNAL_ARRAYS_NAME).read_bytes()
    if _sha256(payload) != manifest.arrays_digest:
        raise ValueError("signal arrays digest mismatch")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {"available_at", "valid", "values"}:
                raise ValueError("signal array allow-list mismatch")
            arrays = SignalArrays(
                values=np.asarray(archive["values"]),
                valid=np.asarray(archive["valid"], dtype=np.bool_),
                available_at=np.asarray(archive["available_at"]),
            )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("signal arrays are invalid") from error
    if arrays.shape != manifest.shape or arrays.dtype.str != manifest.dtype:
        raise ValueError("signal array shape or dtype mismatch")
    if arrays.valid.shape != arrays.shape:
        raise ValueError("signal validity shape mismatch")
    if arrays.available_at.shape != (arrays.shape[0],):
        raise ValueError("signal availability shape mismatch")
    if arrays.available_at.dtype.str != manifest.available_at_dtype:
        raise ValueError("signal availability dtype mismatch")
    if not np.isfinite(arrays.values).all():
        raise ValueError("signal values must be finite")
    return manifest, arrays


__all__ = [
    "SIGNAL_ARRAYS_NAME",
    "SIGNAL_ARTIFACT_SCHEMA",
    "SIGNAL_MANIFEST_NAME",
    "SignalArrayManifest",
    "SignalArrays",
    "load_signal_artifact",
    "write_signal_artifact",
]
