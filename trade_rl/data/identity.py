"""Shared canonical identity computation for resolved market datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes

MARKET_DATASET_IDENTITY_SCHEMA = "market_dataset_identity_v5"

DATASET_ID_ARRAY_FIELDS = (
    "timestamps",
    "available_at",
    "information_available",
    "features",
    "global_features",
    "global_feature_available",
    "global_feature_staleness_hours",
    "global_feature_missing_reason",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "funding_rate",
    "tradable",
    "symbol_active",
    "feature_available",
    "feature_staleness",
)


def _update_digest_array(
    digest: "hashlib._Hash",
    name: str,
    value: np.ndarray,
) -> None:
    array = np.ascontiguousarray(value)
    if np.issubdtype(array.dtype, np.datetime64):
        array = array.astype("datetime64[ns]").astype("<i8")
    elif array.dtype == np.dtype(np.bool_):
        array = array.astype(np.uint8)
    elif np.issubdtype(array.dtype, np.floating):
        array = array.astype("<f8")
    elif np.issubdtype(array.dtype, np.integer):
        array = array.astype("<i8")
    descriptor = json.dumps(
        {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    payload = array.tobytes(order="C")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def content_and_arrays_digest(
    metadata: object,
    arrays: Iterable[tuple[str, np.ndarray]],
) -> str:
    """Hash canonical metadata followed by named canonical array payloads."""

    digest = hashlib.sha256(canonical_json_bytes(metadata))
    for name, array in arrays:
        _update_digest_array(digest, name, array)
    return digest.hexdigest()


def canonical_identity_json(payload: Mapping[str, object]) -> str:
    """Return the canonical UTF-8 JSON representation stored with a dataset."""

    return canonical_json_bytes(payload).decode("utf-8")


def parse_identity_json(value: str) -> dict[str, object]:
    """Parse and require an object-valued canonical identity payload."""

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("dataset identity payload must be a JSON object")
    return parsed


def compute_market_dataset_id(
    payload: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Recompute one dataset ID from its persisted payload and stored arrays."""

    if payload.get("schema") != MARKET_DATASET_IDENTITY_SCHEMA:
        raise ValueError("unsupported market dataset identity schema")
    missing = [name for name in DATASET_ID_ARRAY_FIELDS if name not in arrays]
    if missing:
        raise ValueError(f"dataset identity arrays are missing fields: {missing}")
    return content_and_arrays_digest(
        payload,
        ((name, arrays[name]) for name in DATASET_ID_ARRAY_FIELDS),
    )
