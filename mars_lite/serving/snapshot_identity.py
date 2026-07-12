"""Canonical content identities for inference feature snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

import numpy as np

_SCHEMA_MARKER = "trade-rl-feature-snapshot-v1"


class _Hasher(Protocol):
    def update(self, data: bytes) -> None: ...


def _update_field(hasher: _Hasher, tag: str, payload: bytes) -> None:
    tag_bytes = tag.encode("utf-8")
    hasher.update(len(tag_bytes).to_bytes(8, "big"))
    hasher.update(tag_bytes)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _strings(value: Sequence[str], *, field: str) -> tuple[str, ...]:
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise ValueError(f"{field} must contain non-empty strings")
    return result


def _numeric_array(value: np.ndarray, *, field: str) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    return result


def _update_array(hasher: _Hasher, tag: str, value: np.ndarray) -> None:
    _update_field(hasher, f"{tag}.dtype", b"<f8")
    _update_field(hasher, f"{tag}.shape", _json_bytes(list(value.shape)))
    _update_field(hasher, f"{tag}.data", value.tobytes(order="C"))


def compute_snapshot_id(
    *,
    bundle_digest: str,
    base_timeframe: str,
    timestamps: np.ndarray,
    symbols: Sequence[str],
    feature_names: Sequence[str],
    global_feature_names: Sequence[str],
    feature_history: np.ndarray,
    global_features: np.ndarray,
    close_history: np.ndarray,
) -> str:
    """Hash the exact ordered schema and values used for one inference snapshot."""

    if not bundle_digest:
        raise ValueError("bundle_digest is required")
    if not base_timeframe:
        raise ValueError("base_timeframe is required")

    timestamps_ns = np.ascontiguousarray(
        np.asarray(timestamps, dtype="datetime64[ns]").astype("<i8", copy=False)
    )
    if timestamps_ns.ndim != 1 or timestamps_ns.size == 0:
        raise ValueError("timestamps must be a non-empty one-dimensional array")
    if np.any(timestamps_ns == np.iinfo(np.int64).min):
        raise ValueError("timestamps must not contain NaT")

    symbol_order = _strings(symbols, field="symbols")
    feature_order = _strings(feature_names, field="feature_names")
    global_order = _strings(global_feature_names, field="global_feature_names")
    features = _numeric_array(feature_history, field="feature_history")
    globals_ = _numeric_array(global_features, field="global_features")
    close = _numeric_array(close_history, field="close_history")

    if features.ndim != 3:
        raise ValueError("feature_history must be three-dimensional")
    if features.shape != (
        timestamps_ns.size,
        len(symbol_order),
        len(feature_order),
    ):
        raise ValueError("feature_history shape does not match snapshot schema")
    if globals_.shape != (len(global_order),):
        raise ValueError("global_features shape does not match snapshot schema")
    if close.shape != (timestamps_ns.size, len(symbol_order)):
        raise ValueError("close_history shape does not match snapshot schema")

    hasher = hashlib.sha256()
    _update_field(hasher, "schema", _SCHEMA_MARKER.encode("utf-8"))
    _update_field(hasher, "bundle_digest", bundle_digest.encode("utf-8"))
    _update_field(hasher, "base_timeframe", base_timeframe.encode("utf-8"))
    _update_field(hasher, "symbols", _json_bytes(symbol_order))
    _update_field(hasher, "feature_names", _json_bytes(feature_order))
    _update_field(hasher, "global_feature_names", _json_bytes(global_order))
    _update_field(hasher, "timestamps.dtype", b"datetime64[ns]/<i8")
    _update_field(hasher, "timestamps.shape", _json_bytes(list(timestamps_ns.shape)))
    _update_field(hasher, "timestamps.data", timestamps_ns.tobytes(order="C"))
    _update_array(hasher, "feature_history", features)
    _update_array(hasher, "global_features", globals_)
    _update_array(hasher, "close_history", close)
    return hasher.hexdigest()
