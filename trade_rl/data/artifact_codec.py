"""Canonical deterministic codec shared by all market-dataset artifact workflows."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset

DATASET_MANIFEST_NAME: Final = "manifest.json"
DATASET_ARRAYS_NAME: Final = "arrays.npz"
DATASET_ARTIFACT_SCHEMA: Final = "market_dataset_artifact_v3"
_FIXED_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)


def _sha256_bytes(payload: bytes) -> str:
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
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))
    return buffer.getvalue()


def _encode_scalar(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_encode_scalar(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"unsupported dataset scalar type: {type(value).__name__}")


def _dataset_parts(
    dataset: MarketDataset,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, object] = {}
    for item in fields(MarketDataset):
        if not item.init or item.name.startswith("_"):
            continue
        value = getattr(dataset, item.name)
        if isinstance(value, np.ndarray):
            arrays[item.name] = np.asarray(value)
        else:
            scalars[item.name] = _encode_scalar(value)
    return arrays, scalars


def write_dataset_files(root: Path, dataset: MarketDataset) -> tuple[Path, str]:
    """Write canonical files inside an existing staging or destination directory."""

    if dataset.identity_payload_json is None:
        raise ValueError("formal dataset artifacts require a verified content identity")
    root.mkdir(parents=True, exist_ok=True)
    arrays, scalars = _dataset_parts(dataset)
    payload = _deterministic_npz(arrays)
    arrays_path = root / DATASET_ARRAYS_NAME
    _atomic_write(arrays_path, payload)
    manifest_payload = {
        "arrays": {
            name: {
                "dtype": np.asarray(value).dtype.str,
                "shape": tuple(int(size) for size in np.asarray(value).shape),
            }
            for name, value in sorted(arrays.items())
        },
        "arrays_digest": _sha256_bytes(payload),
        "arrays_file": DATASET_ARRAYS_NAME,
        "dataset": scalars,
        "dataset_id": dataset.dataset_id,
        "schema_version": DATASET_ARTIFACT_SCHEMA,
    }
    artifact_digest = content_digest(manifest_payload)
    manifest = {**manifest_payload, "artifact_digest": artifact_digest}
    path = root / DATASET_MANIFEST_NAME
    _atomic_write(path, canonical_json_bytes(manifest))
    return path, artifact_digest


@dataclass(frozen=True, slots=True)
class DatasetArtifactFiles:
    """Canonical files written for one market-dataset artifact."""

    manifest_path: Path
    arrays_path: Path
    artifact_digest: str


def write_market_dataset_files(
    root: str | Path, dataset: MarketDataset
) -> DatasetArtifactFiles:
    """Write deterministic dataset files and return their typed identities."""

    destination = Path(root)
    manifest_path, artifact_digest = write_dataset_files(destination, dataset)
    return DatasetArtifactFiles(
        manifest_path=manifest_path,
        arrays_path=destination / DATASET_ARRAYS_NAME,
        artifact_digest=artifact_digest,
    )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def load_dataset_files(root: Path) -> MarketDataset:
    """Load and verify one canonical market-dataset artifact directory."""

    manifest_path = root / DATASET_MANIFEST_NAME
    arrays_path = root / DATASET_ARRAYS_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest is missing: {manifest_path}")
    if not arrays_path.is_file():
        raise FileNotFoundError(f"dataset arrays are missing: {arrays_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = dict(_mapping(raw, field="dataset manifest"))
    artifact_digest = _string(
        manifest.pop("artifact_digest", None), field="artifact_digest"
    )
    if content_digest(manifest) != artifact_digest:
        raise ValueError("dataset manifest digest mismatch")
    if manifest.get("schema_version") != DATASET_ARTIFACT_SCHEMA:
        raise ValueError("unsupported dataset artifact schema")
    if manifest.get("arrays_file") != DATASET_ARRAYS_NAME:
        raise ValueError("dataset arrays file identity is invalid")
    payload = arrays_path.read_bytes()
    if _sha256_bytes(payload) != _string(
        manifest.get("arrays_digest"), field="arrays_digest"
    ):
        raise ValueError("dataset arrays digest mismatch")

    metadata = _mapping(manifest.get("arrays"), field="arrays")
    expected_names = set(metadata)
    arrays: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != expected_names:
            raise ValueError("dataset array names do not match manifest")
        for name in sorted(expected_names):
            description = _mapping(metadata[name], field=f"arrays.{name}")
            array = np.asarray(archive[name])
            raw_shape = description.get("shape")
            if not isinstance(raw_shape, list) or any(
                isinstance(size, bool) or not isinstance(size, int)
                for size in raw_shape
            ):
                raise ValueError(f"arrays.{name}.shape is invalid")
            expected_shape = tuple(raw_shape)
            expected_dtype = _string(
                description.get("dtype"), field=f"arrays.{name}.dtype"
            )
            if array.shape != expected_shape:
                raise ValueError(f"dataset array shape mismatch: {name}")
            if array.dtype.str != expected_dtype:
                raise ValueError(f"dataset array dtype mismatch: {name}")
            arrays[name] = array

    scalar_meta = dict(_mapping(manifest.get("dataset"), field="dataset"))
    top_level_dataset_id = _string(manifest.get("dataset_id"), field="dataset_id")
    if scalar_meta.get("dataset_id") != top_level_dataset_id:
        raise ValueError("dataset_id does not match canonical dataset metadata")
    kwargs: dict[str, Any] = {**scalar_meta, **arrays}
    allowed = {item.name for item in fields(MarketDataset) if item.init}
    if set(kwargs) != allowed:
        missing = sorted(allowed - set(kwargs))
        extra = sorted(set(kwargs) - allowed)
        raise ValueError(
            f"dataset artifact field contract mismatch: missing={missing}, extra={extra}"
        )
    return MarketDataset(**kwargs)
