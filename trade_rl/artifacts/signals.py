"""Canonical content-addressed alpha and factor array artifacts."""

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
SIGNAL_ARTIFACT_SCHEMA: Final = "causal_signal_array_artifact_v1"
_FIXED_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
SignalKind = Literal["alpha", "factor"]


@dataclass(frozen=True, slots=True)
class SignalArrayManifest:
    artifact_digest: str
    arrays_digest: str
    dataset_id: str
    fit_start: int
    fit_stop: int
    kind: SignalKind
    names: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    schema_version: str = SIGNAL_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.artifact_digest, field="artifact_digest")
        require_sha256(self.arrays_digest, field="arrays_digest")
        require_sha256(self.dataset_id, field="dataset_id")
        if self.kind not in {"alpha", "factor"}:
            raise ValueError("signal artifact kind is unsupported")
        if self.fit_start < 0 or self.fit_stop <= self.fit_start:
            raise ValueError("signal fit range must be non-empty and half-open")
        if not self.names or any(not name for name in self.names):
            raise ValueError("signal names must be non-empty")
        if len(set(self.names)) != len(self.names):
            raise ValueError("signal names must be unique")
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError("signal shape must be positive")
        if self.schema_version != SIGNAL_ARTIFACT_SCHEMA:
            raise ValueError("unsupported signal artifact schema")

    def digest_payload(self) -> dict[str, object]:
        return {
            "arrays_digest": self.arrays_digest,
            "arrays_file": SIGNAL_ARRAYS_NAME,
            "dataset_id": self.dataset_id,
            "dtype": self.dtype,
            "fit_start": self.fit_start,
            "fit_stop": self.fit_stop,
            "kind": self.kind,
            "names": self.names,
            "schema_version": self.schema_version,
            "shape": self.shape,
        }


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


def _deterministic_npz(values: np.ndarray) -> bytes:
    npy = io.BytesIO()
    np.lib.format.write_array(npy, values, allow_pickle=False)
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo("values.npy", date_time=_FIXED_ZIP_TIMESTAMP)
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, npy.getvalue())
    return output.getvalue()


def write_signal_artifact(
    root: str | Path,
    *,
    kind: SignalKind,
    dataset_id: str,
    fit_start: int,
    fit_stop: int,
    names: tuple[str, ...],
    values: np.ndarray,
) -> str:
    array = np.asarray(values)
    if array.ndim not in {2, 3} or not np.issubdtype(array.dtype, np.number):
        raise ValueError("signal values must be a numeric rank-2 or rank-3 array")
    if not np.isfinite(array).all():
        raise ValueError("signal values must be finite")
    if kind == "alpha" and (array.ndim != 2 or array.shape[1] != len(names)):
        raise ValueError("alpha values must have shape (bars, symbols)")
    if kind == "factor" and (array.ndim != 3 or array.shape[1] != len(names)):
        raise ValueError("factor values must have shape (bars, factors, symbols)")
    if fit_stop > array.shape[0]:
        raise ValueError("signal fit range exceeds the artifact values")
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    payload = _deterministic_npz(array)
    arrays_digest = _sha256(payload)
    base = {
        "arrays_digest": arrays_digest,
        "arrays_file": SIGNAL_ARRAYS_NAME,
        "dataset_id": dataset_id,
        "dtype": array.dtype.str,
        "fit_start": fit_start,
        "fit_stop": fit_stop,
        "kind": kind,
        "names": names,
        "schema_version": SIGNAL_ARTIFACT_SCHEMA,
        "shape": tuple(int(size) for size in array.shape),
    }
    manifest = SignalArrayManifest(
        artifact_digest=content_digest(base),
        arrays_digest=arrays_digest,
        dataset_id=dataset_id,
        fit_start=fit_start,
        fit_stop=fit_stop,
        kind=kind,
        names=names,
        shape=tuple(int(size) for size in array.shape),
        dtype=array.dtype.str,
    )
    _atomic_write(output / SIGNAL_ARRAYS_NAME, payload)
    _atomic_write(output / SIGNAL_MANIFEST_NAME, canonical_json_bytes(manifest))
    return manifest.artifact_digest


def load_signal_artifact(
    root: str | Path,
    *,
    expected_kind: SignalKind | None = None,
) -> tuple[SignalArrayManifest, np.ndarray]:
    path = Path(root)
    manifest_path = path / SIGNAL_MANIFEST_NAME
    arrays_path = path / SIGNAL_ARRAYS_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"signal manifest is missing: {manifest_path}")
    if not arrays_path.is_file():
        raise FileNotFoundError(f"signal arrays are missing: {arrays_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("signal manifest must be a mapping")
    try:
        manifest = SignalArrayManifest(
            artifact_digest=str(raw["artifact_digest"]),
            arrays_digest=str(raw["arrays_digest"]),
            dataset_id=str(raw["dataset_id"]),
            fit_start=int(raw["fit_start"]),
            fit_stop=int(raw["fit_stop"]),
            kind=str(raw["kind"]),  # type: ignore[arg-type]
            names=tuple(str(value) for value in raw["names"]),
            shape=tuple(int(value) for value in raw["shape"]),
            dtype=str(raw["dtype"]),
            schema_version=str(raw["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("signal manifest is invalid") from error
    if content_digest(manifest.digest_payload()) != manifest.artifact_digest:
        raise ValueError("signal manifest digest mismatch")
    if expected_kind is not None and manifest.kind != expected_kind:
        raise ValueError("signal artifact kind mismatch")
    payload = arrays_path.read_bytes()
    if _sha256(payload) != manifest.arrays_digest:
        raise ValueError("signal arrays digest mismatch")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if archive.files != ["values"]:
                raise ValueError("signal array allow-list mismatch")
            values = np.asarray(archive["values"])
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("signal arrays are invalid") from error
    if values.shape != manifest.shape or values.dtype.str != manifest.dtype:
        raise ValueError("signal array shape or dtype mismatch")
    if not np.isfinite(values).all():
        raise ValueError("signal values must be finite")
    return manifest, values


__all__ = [
    "SIGNAL_ARRAYS_NAME",
    "SIGNAL_ARTIFACT_SCHEMA",
    "SIGNAL_MANIFEST_NAME",
    "SignalArrayManifest",
    "load_signal_artifact",
    "write_signal_artifact",
]
