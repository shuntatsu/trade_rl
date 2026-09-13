"""Identity-only materialization for sealed aggTrades capacity calibration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.data.market import MarketDataset

_AUTHORITY_SCHEMA = "aggtrades_capacity_calibration_authority_v1"
_TRANSFORM_SCHEMA = "aggtrades_calibrated_capacity_dataset_transform_v1"
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_CAPS = (
    0.0021629560553901974,
    0.002044685341258238,
    0.002184898995567895,
    0.0020480213652913385,
    0.002346308308284808,
)
_PROTOCOL_DIGEST = "5fb013fb0a3d717846a701b23d2f4bfca8e742ebaef0e6f564a331053f2ed071"
_RESULT_DIGEST = "a5490f9209b16a8ba94f11506b83aa45ce44c7d61d13234480e2362f6df89809"
_RESULT_JSON_SHA256 = "397004e6a1a9c3b30b32212066ac41366f7af74edc0ae23372f847e50c6c5a4e"
_RESULT_ARTIFACT_ID = 10320428830
_RESULT_ARTIFACT_API_DIGEST = (
    "810e792636a696a0bc11e22ab53a21f378aaa0bfaeab8c3411123e9c6d9a5c35"
)
_VERIFIER_RUN_ID = 34767391268
_VERIFIER_ARTIFACT_ID = 10320394746
_VERIFIER_ARTIFACT_API_DIGEST = (
    "31702e68dddd9b4844eee6dc3b5b47ab4aa824d50200509a7bcdbf009c652bd1"
)


@dataclass(frozen=True, slots=True)
class AggTradesCapacityAuthority:
    symbols: tuple[str, ...]
    symbol_caps: tuple[float, ...]
    protocol_digest: str
    result_digest: str
    result_json_sha256: str
    result_artifact_id: int
    result_artifact_api_digest: str
    verifier_run_id: int
    verifier_artifact_id: int
    verifier_artifact_api_digest: str
    schema_version: str = _AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        for field, value in (
            ("protocol_digest", self.protocol_digest),
            ("result_digest", self.result_digest),
            ("result_json_sha256", self.result_json_sha256),
            ("result_artifact_api_digest", self.result_artifact_api_digest),
            ("verifier_artifact_api_digest", self.verifier_artifact_api_digest),
        ):
            require_sha256(value, field=field)
        if self.schema_version != _AUTHORITY_SCHEMA:
            raise ValueError("unsupported calibrated capacity authority schema")
        if self.symbols != _SYMBOLS:
            raise ValueError("symbol roster differs from sealed capacity authority")
        if self.symbol_caps != _CAPS:
            raise ValueError("symbol caps differ from sealed capacity authority")
        if len(self.symbol_caps) != len(self.symbols) or any(
            not math.isfinite(value) or not 0.0 < value <= 0.05
            for value in self.symbol_caps
        ):
            raise ValueError("sealed capacity caps are invalid")
        frozen = (
            (self.protocol_digest, _PROTOCOL_DIGEST),
            (self.result_digest, _RESULT_DIGEST),
            (self.result_json_sha256, _RESULT_JSON_SHA256),
            (self.result_artifact_id, _RESULT_ARTIFACT_ID),
            (self.result_artifact_api_digest, _RESULT_ARTIFACT_API_DIGEST),
            (self.verifier_run_id, _VERIFIER_RUN_ID),
            (self.verifier_artifact_id, _VERIFIER_ARTIFACT_ID),
            (self.verifier_artifact_api_digest, _VERIFIER_ARTIFACT_API_DIGEST),
        )
        if any(actual != expected for actual, expected in frozen):
            raise ValueError(
                "capacity authority differs from sealed calibration evidence"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "symbols": list(self.symbols),
            "symbol_caps": list(self.symbol_caps),
            "protocol_digest": self.protocol_digest,
            "result_digest": self.result_digest,
            "result_json_sha256": self.result_json_sha256,
            "result_artifact_id": self.result_artifact_id,
            "result_artifact_api_digest": self.result_artifact_api_digest,
            "verifier_run_id": self.verifier_run_id,
            "verifier_artifact_id": self.verifier_artifact_id,
            "verifier_artifact_api_digest": self.verifier_artifact_api_digest,
        }


def canonical_aggtrades_capacity_authority() -> AggTradesCapacityAuthority:
    return AggTradesCapacityAuthority(
        symbols=_SYMBOLS,
        symbol_caps=_CAPS,
        protocol_digest=_PROTOCOL_DIGEST,
        result_digest=_RESULT_DIGEST,
        result_json_sha256=_RESULT_JSON_SHA256,
        result_artifact_id=_RESULT_ARTIFACT_ID,
        result_artifact_api_digest=_RESULT_ARTIFACT_API_DIGEST,
        verifier_run_id=_VERIFIER_RUN_ID,
        verifier_artifact_id=_VERIFIER_ARTIFACT_ID,
        verifier_artifact_api_digest=_VERIFIER_ARTIFACT_API_DIGEST,
    )


def _source_identity_payload(source: MarketDataset) -> dict[str, object]:
    if source.identity_payload_json is None:
        raise ValueError("source Dataset must have a verified content identity payload")
    try:
        decoded = cast(object, json.loads(source.identity_payload_json))
    except json.JSONDecodeError as error:
        raise ValueError("source Dataset identity payload is malformed") from error
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise ValueError("source Dataset identity payload must be a JSON object")
    return cast(dict[str, object], decoded)


def build_aggtrades_calibrated_capacity_dataset(
    source: MarketDataset,
    *,
    source_artifact_schema: str,
    source_artifact_digest: str,
    authority: AggTradesCapacityAuthority,
) -> MarketDataset:
    """Replace only Dataset participation-capacity authority and re-content-address."""

    if tuple(source.symbols) != authority.symbols:
        raise ValueError(
            "source Dataset symbol roster differs from sealed capacity authority"
        )
    if not source_artifact_schema:
        raise ValueError("source_artifact_schema must be non-empty")
    require_sha256(source_artifact_digest, field="source_artifact_digest")
    source_identity = _source_identity_payload(source)
    source_identity_json = source.identity_payload_json
    if source_identity_json is None:
        raise RuntimeError("validated source Dataset lost its identity payload")
    source_identity_bytes = source_identity_json.encode("utf-8")
    cap_vector = np.asarray(authority.symbol_caps, dtype=np.float64)
    cap_matrix = np.broadcast_to(cap_vector, (source.n_bars, source.n_symbols)).copy()
    transformed = replace(
        source,
        max_participation_rate=cap_matrix,
        identity_payload_json=None,
    ).with_content_identity(
        {
            "source_dataset": {
                "dataset_id": source.dataset_id,
                "artifact_schema": source_artifact_schema,
                "artifact_digest": source_artifact_digest,
                "identity_payload_sha256": hashlib.sha256(
                    source_identity_bytes
                ).hexdigest(),
                "identity_payload": source_identity,
            },
            "execution_capacity_calibration": authority.to_payload(),
            "transformation": {
                "schema_version": _TRANSFORM_SCHEMA,
                "replaced_identity_array": "max_participation_rate",
            },
        }
    )

    source_arrays = source.identity_arrays()
    transformed_arrays = transformed.identity_arrays()
    if set(source_arrays) != set(transformed_arrays):
        raise RuntimeError(
            "Dataset identity-array roster changed during capacity transform"
        )
    for name, source_array in source_arrays.items():
        candidate = transformed_arrays[name]
        if (
            source_array.dtype != candidate.dtype
            or source_array.shape != candidate.shape
        ):
            raise RuntimeError(f"Dataset identity-array shape/dtype drifted: {name}")
        if name == "max_participation_rate":
            if not np.array_equal(candidate, cap_matrix):
                raise RuntimeError(
                    "calibrated participation array differs from sealed caps"
                )
        elif source_array.tobytes() != candidate.tobytes():
            raise RuntimeError(f"non-cap Dataset identity array changed: {name}")
    if transformed.dataset_id == source.dataset_id:
        raise RuntimeError("calibrated Dataset identity did not change")
    return transformed


__all__ = [
    "AggTradesCapacityAuthority",
    "build_aggtrades_calibrated_capacity_dataset",
    "canonical_aggtrades_capacity_authority",
]
