"""Fail-closed loader for the maintained PostgreSQL indicator cache."""

from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest

INDICATOR_CACHE_ID: Final = "binance-usds-m-native-indicators-15x-202101-202606-v1"
INDICATOR_ARTIFACT_TABLE: Final = (
    "market_raw.binance_usds_m_indicator_artifacts_202101_202606"
)
INDICATOR_MANIFEST_TABLE: Final = (
    "market_raw.binance_usds_m_indicator_manifests_202101_202606"
)
_MANIFEST_SCHEMA: Final = "native_indicator_cache_v1"
_PAYLOAD_SCHEMA_PREFIX: Final = "npz_native_indicator_v1:"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARRAY_NAMES: Final = frozenset({"event_time_ms", "values", "available"})


class _Cursor(Protocol):
    def execute(self, query: str, params: object = None) -> Any: ...

    def fetchone(self) -> Sequence[object] | None: ...

    def fetchall(self) -> Sequence[Sequence[object]]: ...

    def __enter__(self) -> _Cursor: ...

    def __exit__(self, *args: object) -> None: ...


class IndicatorArtifactConnection(Protocol):
    """Small DB-API boundary needed by the indicator loader."""

    def cursor(self) -> _Cursor: ...


@dataclass(frozen=True, slots=True)
class NativeIndicatorArtifact:
    """One verified symbol/timeframe indicator matrix."""

    symbol: str
    timeframe: str
    feature_names: tuple[str, ...]
    event_time_ms: np.ndarray
    values: np.ndarray
    available: np.ndarray
    payload_schema: str
    payload_sha256: str

    def __post_init__(self) -> None:
        for value in (self.event_time_ms, self.values, self.available):
            value.setflags(write=False)


@dataclass(frozen=True, slots=True)
class NativeIndicatorArtifactBundle:
    """Requested cache subset in caller-declared symbol/timeframe order."""

    cache_id: str
    market: str
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    feature_config_digest: str
    artifacts: tuple[NativeIndicatorArtifact, ...]

    def get(self, symbol: str, timeframe: str) -> NativeIndicatorArtifact:
        """Return one exact member without assuming a benchmark symbol."""

        for artifact in self.artifacts:
            if artifact.symbol == symbol and artifact.timeframe == timeframe:
                return artifact
        raise KeyError((symbol, timeframe))

    def by_symbol(self) -> dict[str, tuple[NativeIndicatorArtifact, ...]]:
        """Expose inputs in the same symbol order used by ``MarketDataset``."""

        return {
            symbol: tuple(self.get(symbol, timeframe) for timeframe in self.timeframes)
            for symbol in self.symbols
        }


def _ordered_unique(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    resolved = tuple(str(value) for value in values)
    if not resolved or any(not value for value in resolved):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} must be unique")
    return resolved


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must be unique")
    return result


def _sha256(value: object, *, field: str) -> str:
    resolved = str(value).strip()
    if _SHA256.fullmatch(resolved) is None:
        raise ValueError(f"{field} must be lowercase sha256")
    return resolved


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _feature_names_by_timeframe(
    feature_specs: Mapping[str, object],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    base = feature_specs.get("base_timeframe")
    if not isinstance(base, str) or not base:
        raise ValueError("feature_specs.base_timeframe must be a non-empty string")
    additional = _strings(
        feature_specs.get("feature_timeframes"),
        field="feature_specs.feature_timeframes",
    )
    timeframes = (base, *additional)
    raw_features = feature_specs.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise ValueError("feature_specs.features must be a non-empty list")
    grouped: dict[str, list[str]] = {timeframe: [] for timeframe in timeframes}
    all_names: set[str] = set()
    for index, raw in enumerate(raw_features):
        spec = _mapping(raw, field=f"feature_specs.features[{index}]")
        name = spec.get("name")
        if not isinstance(name, str) or not name or name in all_names:
            raise ValueError("indicator feature names must be non-empty and unique")
        timeframe = spec.get("timeframe", base)
        if not isinstance(timeframe, str) or timeframe not in grouped:
            raise ValueError(f"indicator feature has unsupported timeframe: {name}")
        grouped[timeframe].append(name)
        all_names.add(name)
    if any(not names for names in grouped.values()):
        raise ValueError("each maintained timeframe must contain features")
    return timeframes, {key: tuple(value) for key, value in grouped.items()}


def _load_npz(
    payload: bytes,
    *,
    row_count: int,
    feature_count: int,
    available_value_count: int,
    first_event_time_ms: int,
    last_event_time_ms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != _ARRAY_NAMES:
                raise ValueError("indicator NPZ array names are invalid")
            event_time_ms = np.asarray(archive["event_time_ms"])
            values = np.asarray(archive["values"])
            available = np.asarray(archive["available"])
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("indicator NPZ"):
            raise
        raise ValueError("indicator NPZ payload is invalid") from error
    expected_matrix_shape = (row_count, feature_count)
    if event_time_ms.dtype != np.dtype(np.int64) or event_time_ms.shape != (row_count,):
        raise ValueError("indicator event_time_ms contract mismatch")
    if values.dtype != np.dtype(np.float32) or values.shape != expected_matrix_shape:
        raise ValueError("indicator values contract mismatch")
    if (
        available.dtype != np.dtype(np.bool_)
        or available.shape != expected_matrix_shape
    ):
        raise ValueError("indicator available contract mismatch")
    if row_count <= 0 or feature_count <= 0:
        raise ValueError("indicator dimensions must be positive")
    if np.any(np.diff(event_time_ms) <= 0):
        raise ValueError("indicator event times must be strictly increasing")
    if int(event_time_ms[0]) != first_event_time_ms:
        raise ValueError("indicator first event time mismatch")
    if int(event_time_ms[-1]) != last_event_time_ms:
        raise ValueError("indicator last event time mismatch")
    if int(np.count_nonzero(available)) != available_value_count:
        raise ValueError("indicator available value count mismatch")
    if not np.isfinite(values[available]).all():
        raise ValueError("available indicator values must be finite")
    return event_time_ms, values, available


def load_postgres_indicator_artifacts(
    connection: IndicatorArtifactConnection,
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    cache_id: str = INDICATOR_CACHE_ID,
) -> NativeIndicatorArtifactBundle:
    """Load and verify only the requested PostgreSQL NPZ artifacts."""

    requested_symbols = _ordered_unique(symbols, field="symbols")
    requested_timeframes = _ordered_unique(timeframes, field="timeframes")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT cache_id, schema_version, market, symbols, start_time, end_time,
                   feature_config_digest, feature_specs, artifact_count
            FROM {INDICATOR_MANIFEST_TABLE}
            WHERE cache_id = %s
            """,
            (cache_id,),
        )
        manifest_row = cursor.fetchone()
        if manifest_row is None:
            raise FileNotFoundError(f"indicator manifest is missing: {cache_id}")
        cursor.execute(
            f"""
            SELECT symbol, timeframe, row_count, feature_count,
                   available_value_count, first_event_time_ms, last_event_time_ms,
                   payload_schema, payload_sha256, payload_bytes, npz_payload
            FROM {INDICATOR_ARTIFACT_TABLE}
            WHERE cache_id = %s
              AND symbol = ANY(%s)
              AND timeframe = ANY(%s)
            ORDER BY symbol, timeframe
            """,
            (cache_id, list(requested_symbols), list(requested_timeframes)),
        )
        artifact_rows = tuple(cursor.fetchall())

    if len(manifest_row) != 9:
        raise ValueError("indicator manifest row contract mismatch")
    (
        stored_cache_id,
        schema_version,
        market,
        raw_symbols,
        start_time,
        end_time,
        feature_config_digest,
        raw_feature_specs,
        artifact_count,
    ) = manifest_row
    if stored_cache_id != cache_id or schema_version != _MANIFEST_SCHEMA:
        raise ValueError("indicator manifest identity mismatch")
    if not isinstance(market, str) or not market:
        raise ValueError("indicator market identity is invalid")
    manifest_symbols = _strings(raw_symbols, field="manifest.symbols")
    unknown_symbols = sorted(set(requested_symbols) - set(manifest_symbols))
    if unknown_symbols:
        raise ValueError(f"indicator symbols are not in manifest: {unknown_symbols}")
    if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
        raise ValueError("indicator manifest times must be datetimes")
    if start_time.tzinfo is None or end_time.tzinfo is None or end_time <= start_time:
        raise ValueError("indicator manifest time range is invalid")
    feature_specs = _mapping(raw_feature_specs, field="feature_specs")
    config_digest = _sha256(feature_config_digest, field="feature_config_digest")
    if content_digest(feature_specs) != config_digest:
        raise ValueError("indicator feature config digest mismatch")
    maintained_timeframes, names_by_timeframe = _feature_names_by_timeframe(
        feature_specs
    )
    if _integer(artifact_count, field="artifact_count") != len(manifest_symbols) * len(
        maintained_timeframes
    ):
        raise ValueError("indicator manifest artifact count mismatch")
    unknown_timeframes = sorted(set(requested_timeframes) - set(maintained_timeframes))
    if unknown_timeframes:
        raise ValueError(
            f"indicator timeframes are not in manifest: {unknown_timeframes}"
        )

    rows_by_key: dict[tuple[str, str], Sequence[object]] = {}
    for row in artifact_rows:
        if len(row) != 11:
            raise ValueError("indicator artifact row contract mismatch")
        key = (str(row[0]), str(row[1]))
        if key in rows_by_key:
            raise ValueError(f"duplicate indicator artifact: {key}")
        rows_by_key[key] = row
    expected_keys = {
        (symbol, timeframe)
        for symbol in requested_symbols
        for timeframe in requested_timeframes
    }
    actual_keys = set(rows_by_key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise FileNotFoundError(
            f"indicator artifact set mismatch: missing={missing}, extra={extra}"
        )

    schema_by_timeframe: dict[str, str] = {}
    artifacts: list[NativeIndicatorArtifact] = []
    for symbol in requested_symbols:
        for timeframe in requested_timeframes:
            row = rows_by_key[(symbol, timeframe)]
            row_count = _integer(row[2], field="row_count")
            feature_count = _integer(row[3], field="feature_count")
            available_count = _integer(row[4], field="available_value_count")
            first_event = _integer(row[5], field="first_event_time_ms")
            last_event = _integer(row[6], field="last_event_time_ms")
            payload_schema = str(row[7]).strip()
            if not payload_schema.startswith(_PAYLOAD_SCHEMA_PREFIX):
                raise ValueError("unsupported indicator payload schema")
            _sha256(
                payload_schema.removeprefix(_PAYLOAD_SCHEMA_PREFIX),
                field="payload_schema digest",
            )
            previous_schema = schema_by_timeframe.setdefault(timeframe, payload_schema)
            if previous_schema != payload_schema:
                raise ValueError("indicator payload schema differs within timeframe")
            payload_digest = _sha256(row[8], field="payload_sha256")
            payload_size = _integer(row[9], field="payload_bytes")
            raw_payload = row[10]
            if not isinstance(raw_payload, bytes | bytearray | memoryview):
                raise ValueError("indicator NPZ payload must be binary")
            payload = bytes(raw_payload)
            if len(payload) != payload_size:
                raise ValueError("indicator payload byte count mismatch")
            if hashlib.sha256(payload).hexdigest() != payload_digest:
                raise ValueError("indicator payload digest mismatch")
            feature_names = names_by_timeframe[timeframe]
            if feature_count != len(feature_names):
                raise ValueError("indicator feature count differs from manifest")
            event_time_ms, values, available = _load_npz(
                payload,
                row_count=row_count,
                feature_count=feature_count,
                available_value_count=available_count,
                first_event_time_ms=first_event,
                last_event_time_ms=last_event,
            )
            artifacts.append(
                NativeIndicatorArtifact(
                    symbol=symbol,
                    timeframe=timeframe,
                    feature_names=feature_names,
                    event_time_ms=event_time_ms,
                    values=values,
                    available=available,
                    payload_schema=payload_schema,
                    payload_sha256=payload_digest,
                )
            )
    return NativeIndicatorArtifactBundle(
        cache_id=cache_id,
        market=market,
        symbols=requested_symbols,
        timeframes=requested_timeframes,
        start_time=start_time,
        end_time=end_time,
        feature_config_digest=config_digest,
        artifacts=tuple(artifacts),
    )


__all__ = [
    "INDICATOR_ARTIFACT_TABLE",
    "INDICATOR_CACHE_ID",
    "INDICATOR_MANIFEST_TABLE",
    "IndicatorArtifactConnection",
    "NativeIndicatorArtifact",
    "NativeIndicatorArtifactBundle",
    "load_postgres_indicator_artifacts",
]
