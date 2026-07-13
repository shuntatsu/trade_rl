"""Deterministic market-dataset artifacts and range-scoped dataset views."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketCalendarKind, MarketDataset

DATASET_MANIFEST_NAME: Final = "manifest.json"
DATASET_ARRAYS_NAME: Final = "arrays.npz"
DATASET_ARTIFACT_SCHEMA: Final = "market_dataset_artifact_v1"
DATASET_VIEW_SCHEMA: Final = "market_dataset_view_v1"
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
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))
    return buffer.getvalue()


def _dataset_arrays(dataset: MarketDataset) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for item in fields(MarketDataset):
        if not item.init or item.name.startswith("_"):
            continue
        value = getattr(dataset, item.name)
        if isinstance(value, np.ndarray):
            arrays[item.name] = np.asarray(value)
    return arrays


def _dataset_scalars(dataset: MarketDataset) -> dict[str, object]:
    return {
        "calendar_kind": MarketCalendarKind(dataset.calendar_kind).value,
        "dataset_id": dataset.dataset_id,
        "feature_names": dataset.feature_names,
        "global_feature_names": dataset.global_feature_names,
        "nominal_bar_hours": dataset.nominal_bar_hours,
        "periods_per_year": dataset.periods_per_year,
        "symbols": dataset.symbols,
    }


def write_market_dataset_artifact(root: Path, dataset: MarketDataset) -> Path:
    """Write a deterministic ``manifest.json`` plus ``arrays.npz`` artifact."""

    root.mkdir(parents=True, exist_ok=True)
    arrays = _dataset_arrays(dataset)
    payload = _deterministic_npz(arrays)
    arrays_path = root / DATASET_ARRAYS_NAME
    _atomic_write(arrays_path, payload)
    manifest = {
        "arrays": {
            name: {
                "dtype": np.asarray(value).dtype.str,
                "shape": tuple(int(size) for size in np.asarray(value).shape),
            }
            for name, value in sorted(arrays.items())
        },
        "arrays_digest": _sha256_bytes(payload),
        "arrays_file": DATASET_ARRAYS_NAME,
        "dataset": _dataset_scalars(dataset),
        "schema_version": DATASET_ARTIFACT_SCHEMA,
    }
    path = root / DATASET_MANIFEST_NAME
    _atomic_write(path, canonical_json_bytes(manifest))
    return path


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def load_market_dataset_artifact(root: Path) -> MarketDataset:
    """Load and fully validate a market-dataset artifact."""

    manifest_path = root / DATASET_MANIFEST_NAME
    arrays_path = root / DATASET_ARRAYS_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest is missing: {manifest_path}")
    if not arrays_path.is_file():
        raise FileNotFoundError(f"dataset arrays are missing: {arrays_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _mapping(raw, field="dataset manifest")
    if manifest.get("schema_version") != DATASET_ARTIFACT_SCHEMA:
        raise ValueError("unsupported dataset artifact schema")
    if manifest.get("arrays_file") != DATASET_ARRAYS_NAME:
        raise ValueError("dataset arrays file identity is invalid")
    payload = arrays_path.read_bytes()
    expected_digest = _string(manifest.get("arrays_digest"), field="arrays_digest")
    if _sha256_bytes(payload) != expected_digest:
        raise ValueError("dataset arrays digest mismatch")

    raw_array_metadata = _mapping(manifest.get("arrays"), field="arrays")
    expected_names = set(raw_array_metadata)
    arrays: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        observed_names = set(archive.files)
        if observed_names != expected_names:
            raise ValueError("dataset array names do not match manifest")
        for name in sorted(expected_names):
            metadata = _mapping(raw_array_metadata[name], field=f"arrays.{name}")
            array = np.asarray(archive[name])
            expected_shape_raw = metadata.get("shape")
            if not isinstance(expected_shape_raw, list) or any(
                isinstance(size, bool) or not isinstance(size, int)
                for size in expected_shape_raw
            ):
                raise ValueError(f"arrays.{name}.shape is invalid")
            expected_shape = tuple(expected_shape_raw)
            expected_dtype = _string(metadata.get("dtype"), field=f"arrays.{name}.dtype")
            if array.shape != expected_shape:
                raise ValueError(f"dataset array shape mismatch: {name}")
            if array.dtype.str != expected_dtype:
                raise ValueError(f"dataset array dtype mismatch: {name}")
            arrays[name] = array

    dataset_meta = _mapping(manifest.get("dataset"), field="dataset")
    nominal = dataset_meta.get("nominal_bar_hours")
    if nominal is not None and (
        isinstance(nominal, bool) or not isinstance(nominal, int | float)
    ):
        raise ValueError("dataset.nominal_bar_hours must be numeric or null")
    kwargs: dict[str, Any] = {
        "dataset_id": _string(dataset_meta.get("dataset_id"), field="dataset.dataset_id"),
        "symbols": _string_tuple(dataset_meta.get("symbols"), field="dataset.symbols"),
        "feature_names": _string_tuple(
            dataset_meta.get("feature_names"), field="dataset.feature_names"
        ),
        "global_feature_names": _string_tuple(
            dataset_meta.get("global_feature_names"),
            field="dataset.global_feature_names",
        ),
        "periods_per_year": _integer(
            dataset_meta.get("periods_per_year"), field="dataset.periods_per_year"
        ),
        "calendar_kind": _string(
            dataset_meta.get("calendar_kind"), field="dataset.calendar_kind"
        ),
        "nominal_bar_hours": None if nominal is None else float(nominal),
        **arrays,
    }
    allowed = {item.name for item in fields(MarketDataset) if item.init}
    if not set(kwargs).issubset(allowed):
        raise ValueError("dataset artifact contains unsupported fields")
    return MarketDataset(**kwargs)


@dataclass(frozen=True, slots=True)
class MarketDatasetView:
    """A half-open, absolute-index view that cannot escape its assigned range."""

    dataset: MarketDataset
    start: int
    stop: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or isinstance(self.stop, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.stop, int)
            or not 0 <= self.start < self.stop <= self.dataset.n_bars
        ):
            raise ValueError("dataset view range is outside the dataset")

    @property
    def identity(self) -> str:
        return content_digest(
            {
                "dataset_id": self.dataset.dataset_id,
                "schema_version": DATASET_VIEW_SCHEMA,
                "start": self.start,
                "stop": self.stop,
            }
        )

    def subview(self, start: int, stop: int) -> MarketDatasetView:
        if not self.start <= start < stop <= self.stop:
            raise ValueError("requested subview is outside the parent range")
        return MarketDatasetView(self.dataset, start, stop)

    def materialize(self) -> MarketDataset:
        if self.stop - self.start < 3:
            raise ValueError("materialized dataset view requires at least three bars")
        kwargs: dict[str, Any] = {}
        for item in fields(MarketDataset):
            if not item.init or item.name.startswith("_"):
                continue
            value = getattr(self.dataset, item.name)
            if item.name == "dataset_id":
                kwargs[item.name] = self.identity
            elif isinstance(value, np.ndarray) and value.shape[:1] == (
                self.dataset.n_bars,
            ):
                kwargs[item.name] = value[self.start : self.stop]
            else:
                kwargs[item.name] = value
        return MarketDataset(**kwargs)
