"""Path-free decoder for preadmitted, hash-pinned market Dataset artifact bytes.

This software-only adapter lives outside the frozen general artifact codec source
closure. Keep its validation equivalent to the file-backed codec and prove parity
against real synthetic v3/v4 artifact files in the owning tests. It does not authorize
opening an artifact or any economic dataset.
"""

from __future__ import annotations

import io
import json
from dataclasses import fields
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts.codec import (
    DATASET_ARRAYS_NAME,
    DATASET_ARTIFACT_SCHEMA,
    DATASET_ARTIFACT_SCHEMA_V4,
    _artifact_schema_for_identity,
    _mapping,
    _sha256_bytes,
    _string,
)
from trade_rl.data.identity import parse_identity_json
from trade_rl.data.market import MarketDataset


def load_dataset_from_verified_bytes(
    manifest_bytes: bytes, arrays_bytes: bytes
) -> tuple[MarketDataset, str]:
    """Decode only the supplied snapshots; never re-open their original paths.

    The caller must first admit the manifest before reading arrays, then hash-pin
    the arrays bytes. This decoder independently checks the artifact schema,
    self-digest, array digest, shapes, dtypes and Dataset identity.
    """

    if type(manifest_bytes) is not bytes or type(arrays_bytes) is not bytes:
        raise TypeError("manifest and arrays must be bytes")
    manifest = dict(
        _mapping(json.loads(manifest_bytes.decode("utf-8")), field="dataset manifest")
    )
    artifact_digest = _string(
        manifest.pop("artifact_digest", None), field="artifact_digest"
    )
    if content_digest(manifest) != artifact_digest:
        raise ValueError("dataset manifest digest mismatch")
    artifact_schema = manifest.get("schema_version")
    if artifact_schema not in {DATASET_ARTIFACT_SCHEMA, DATASET_ARTIFACT_SCHEMA_V4}:
        raise ValueError("unsupported dataset artifact schema")
    scalar_meta = dict(_mapping(manifest.get("dataset"), field="dataset"))
    identity_payload_json = _string(
        scalar_meta.get("identity_payload_json"),
        field="dataset.identity_payload_json",
    )
    identity_schema = parse_identity_json(identity_payload_json).get("schema")
    if artifact_schema != _artifact_schema_for_identity(identity_schema):
        raise ValueError("artifact and identity schemas do not match")
    if manifest.get("arrays_file") != DATASET_ARRAYS_NAME:
        raise ValueError("dataset arrays file identity is invalid")
    metadata = _mapping(manifest.get("arrays"), field="arrays")
    overlapping_fields = sorted(set(scalar_meta).intersection(metadata))
    if overlapping_fields:
        raise ValueError(
            f"dataset artifact scalar/array field overlap: {overlapping_fields}"
        )
    has_funding_price_rate = "funding_price_rate" in metadata
    if artifact_schema == DATASET_ARTIFACT_SCHEMA_V4 and not has_funding_price_rate:
        raise ValueError("v4 artifact is missing funding_price_rate")
    if artifact_schema == DATASET_ARTIFACT_SCHEMA and has_funding_price_rate:
        raise ValueError("v3 artifact must not contain funding_price_rate")
    if _sha256_bytes(arrays_bytes) != _string(
        manifest.get("arrays_digest"), field="arrays_digest"
    ):
        raise ValueError("dataset arrays digest mismatch")

    expected_names = set(metadata)
    arrays: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as archive:
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

    top_level_dataset_id = _string(manifest.get("dataset_id"), field="dataset_id")
    if scalar_meta.get("dataset_id") != top_level_dataset_id:
        raise ValueError("dataset_id does not match canonical dataset metadata")
    kwargs: dict[str, Any] = {**scalar_meta, **arrays}
    allowed = {item.name for item in fields(MarketDataset) if item.init}
    if artifact_schema == DATASET_ARTIFACT_SCHEMA:
        allowed.remove("funding_price_rate")
    if set(kwargs) != allowed:
        missing = sorted(allowed - set(kwargs))
        extra = sorted(set(kwargs) - allowed)
        raise ValueError(
            f"dataset artifact field contract mismatch: missing={missing}, extra={extra}"
        )
    return MarketDataset(**kwargs), artifact_digest
