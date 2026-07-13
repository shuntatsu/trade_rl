"""Deterministic on-disk artifacts for resolved market datasets."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset

MARKET_ARTIFACT_SCHEMA = "market_dataset_artifact_v1"
MARKET_ARTIFACT_MANIFEST = "manifest.json"
MARKET_ARTIFACT_ARRAYS = "arrays.npz"

_ARRAY_FIELDS = (
    "timestamps",
    "available_at",
    "features",
    "global_features",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "funding_rate",
    "tradable",
    "symbol_active",
    "information_available",
    "feature_available",
    "feature_staleness",
    "contract_multipliers",
)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _write_arrays(path: Path, dataset: MarketDataset) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for field_name in _ARRAY_FIELDS:
            value = getattr(dataset, field_name)
            if value is None:
                raise ValueError(f"dataset field is unresolved: {field_name}")
            info = zipfile.ZipInfo(
                filename=f"{field_name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                _npy_bytes(np.asarray(value)),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _manifest_payload(
    dataset: MarketDataset, *, arrays_digest: str
) -> dict[str, object]:
    return {
        "schema": MARKET_ARTIFACT_SCHEMA,
        "dataset_id": dataset.dataset_id,
        "symbols": dataset.symbols,
        "feature_names": dataset.feature_names,
        "global_feature_names": dataset.global_feature_names,
        "periods_per_year": dataset.periods_per_year,
        "volume_units": tuple(value.value for value in dataset.volume_units),
        "feature_config_digest": dataset.feature_config_digest,
        "normalization_digest": dataset.normalization_digest,
        "arrays_file": MARKET_ARTIFACT_ARRAYS,
        "arrays_digest": arrays_digest,
        "array_fields": _ARRAY_FIELDS,
    }


def write_market_dataset_artifact(root: str | Path, dataset: MarketDataset) -> str:
    """Write an atomic deterministic artifact and return its manifest digest."""

    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / MARKET_ARTIFACT_ARRAYS
    temporary_arrays = arrays_path.with_name(f".{arrays_path.name}.tmp")
    _write_arrays(temporary_arrays, dataset)
    temporary_arrays.replace(arrays_path)

    payload = _manifest_payload(dataset, arrays_digest=_file_digest(arrays_path))
    artifact_digest = content_digest(payload)
    manifest = {**payload, "artifact_digest": artifact_digest}
    manifest_path = output / MARKET_ARTIFACT_MANIFEST
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_bytes(canonical_json_bytes(manifest))
    temporary_manifest.replace(manifest_path)
    return artifact_digest


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


def _read_arrays(path: Path) -> dict[str, np.ndarray]:
    expected = {f"{name}.npy" for name in _ARRAY_FIELDS}
    result: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path, mode="r") as archive:
        names = set(archive.namelist())
        if names != expected:
            raise ValueError("market artifact arrays do not match the schema")
        for field_name in _ARRAY_FIELDS:
            payload = archive.read(f"{field_name}.npy")
            result[field_name] = np.load(io.BytesIO(payload), allow_pickle=False)
    return result


def load_market_dataset_artifact(root: str | Path) -> MarketDataset:
    """Verify and load one deterministic market dataset artifact."""

    artifact_root = Path(root)
    manifest_path = artifact_root / MARKET_ARTIFACT_MANIFEST
    arrays_path = artifact_root / MARKET_ARTIFACT_ARRAYS
    if not manifest_path.is_file():
        raise FileNotFoundError(f"market artifact manifest is missing: {manifest_path}")
    if not arrays_path.is_file():
        raise FileNotFoundError(f"market artifact arrays are missing: {arrays_path}")

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = dict(_mapping(raw_manifest, field="market artifact manifest"))
    artifact_digest = _string(
        manifest.pop("artifact_digest", None),
        field="artifact_digest",
    )
    if content_digest(manifest) != artifact_digest:
        raise ValueError("market artifact manifest digest mismatch")
    if _string(manifest.get("schema"), field="schema") != MARKET_ARTIFACT_SCHEMA:
        raise ValueError("unsupported market artifact schema")
    if (
        _string(manifest.get("arrays_file"), field="arrays_file")
        != MARKET_ARTIFACT_ARRAYS
    ):
        raise ValueError("market artifact arrays_file is unsupported")
    if _file_digest(arrays_path) != _string(
        manifest.get("arrays_digest"),
        field="arrays_digest",
    ):
        raise ValueError("market artifact arrays digest mismatch")
    if _strings(manifest.get("array_fields"), field="array_fields") != _ARRAY_FIELDS:
        raise ValueError("market artifact array field order is unsupported")

    arrays = _read_arrays(arrays_path)
    return MarketDataset(
        dataset_id=_string(manifest.get("dataset_id"), field="dataset_id"),
        symbols=_strings(manifest.get("symbols"), field="symbols"),
        timestamps=arrays["timestamps"],
        available_at=arrays["available_at"],
        features=arrays["features"],
        global_features=arrays["global_features"],
        open=arrays["open"],
        high=arrays["high"],
        low=arrays["low"],
        close=arrays["close"],
        volume=arrays["volume"],
        funding_rate=arrays["funding_rate"],
        tradable=arrays["tradable"],
        symbol_active=arrays["symbol_active"],
        information_available=arrays["information_available"],
        feature_available=arrays["feature_available"],
        feature_staleness=arrays["feature_staleness"],
        feature_names=_strings(manifest.get("feature_names"), field="feature_names"),
        global_feature_names=_strings(
            manifest.get("global_feature_names"),
            field="global_feature_names",
        ),
        volume_units=tuple(
            VolumeUnit(value)
            for value in _strings(manifest.get("volume_units"), field="volume_units")
        ),
        contract_multipliers=arrays["contract_multipliers"],
        feature_config_digest=_string(
            manifest.get("feature_config_digest"),
            field="feature_config_digest",
        ),
        normalization_digest=_string(
            manifest.get("normalization_digest"),
            field="normalization_digest",
        ),
        periods_per_year=_integer(
            manifest.get("periods_per_year"),
            field="periods_per_year",
        ),
    )
