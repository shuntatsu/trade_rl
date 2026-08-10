"""Metadata-only inventory for verified PostgreSQL indicator artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_aware_datetime, require_sha256
from trade_rl.integrations.postgres_indicator_artifacts import (
    INDICATOR_ARTIFACT_TABLE,
    INDICATOR_CACHE_ID,
    INDICATOR_MANIFEST_TABLE,
    IndicatorArtifactConnection,
)

_MANIFEST_SCHEMA: Final = "native_indicator_cache_v1"
_PAYLOAD_SCHEMA_PREFIX: Final = "npz_native_indicator_v1:"
_SUPPORTED_MARKET: Final = "usds-m"


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field} must be a non-empty string list or tuple")
    resolved = tuple(value)
    if not resolved:
        raise ValueError(f"{field} must not be empty")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} must contain unique values")
    return resolved


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _digest(value: object, *, field: str) -> str:
    resolved = _string(value, field=field)
    require_sha256(resolved, field=field)
    return resolved


def _aware(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be a datetime")
    require_aware_datetime(value, field=field)
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _feature_contract(
    feature_specs: Mapping[str, object],
) -> tuple[tuple[str, ...], dict[str, int]]:
    base_timeframe = _string(
        feature_specs.get("base_timeframe"),
        field="feature_specs.base_timeframe",
    )
    additional = _strings(
        feature_specs.get("feature_timeframes"),
        field="feature_specs.feature_timeframes",
    )
    timeframes = (base_timeframe, *additional)
    if len(set(timeframes)) != len(timeframes):
        raise ValueError("indicator feature timeframes must be unique")
    raw_features = feature_specs.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise ValueError("feature_specs.features must be a non-empty list")
    counts = {timeframe: 0 for timeframe in timeframes}
    feature_names: set[str] = set()
    for index, raw_feature in enumerate(raw_features):
        feature = _mapping(raw_feature, field=f"feature_specs.features[{index}]")
        name = _string(
            feature.get("name"),
            field=f"feature_specs.features[{index}].name",
        )
        if name in feature_names:
            raise ValueError("indicator feature names must be unique")
        feature_names.add(name)
        timeframe = feature.get("timeframe", base_timeframe)
        if not isinstance(timeframe, str) or timeframe not in counts:
            raise ValueError(f"indicator feature has unsupported timeframe: {name}")
        counts[timeframe] += 1
    if any(count <= 0 for count in counts.values()):
        raise ValueError("each indicator timeframe must contain features")
    return timeframes, counts


@dataclass(frozen=True, slots=True)
class PostgresIndicatorArtifactMetadata:
    """Verified metadata for one stored symbol/timeframe artifact."""

    symbol: str
    timeframe: str
    row_count: int
    feature_count: int
    available_value_count: int
    first_event_time_ms: int
    last_event_time_ms: int
    payload_schema: str
    payload_sha256: str
    payload_bytes: int

    def __post_init__(self) -> None:
        symbol = _string(self.symbol, field="indicator metadata symbol")
        timeframe = _string(self.timeframe, field="indicator metadata timeframe")
        row_count = _integer(self.row_count, field="indicator row_count", minimum=1)
        feature_count = _integer(
            self.feature_count,
            field="indicator feature_count",
            minimum=1,
        )
        available_count = _integer(
            self.available_value_count,
            field="indicator available_value_count",
        )
        if available_count > row_count * feature_count:
            raise ValueError("indicator available value count exceeds matrix capacity")
        first_event = _integer(
            self.first_event_time_ms,
            field="indicator first_event_time_ms",
        )
        last_event = _integer(
            self.last_event_time_ms,
            field="indicator last_event_time_ms",
        )
        if last_event <= first_event:
            raise ValueError("indicator event-time range is invalid")
        payload_schema = _string(
            self.payload_schema,
            field="indicator payload_schema",
        )
        if not payload_schema.startswith(_PAYLOAD_SCHEMA_PREFIX):
            raise ValueError("indicator payload schema is unsupported")
        _digest(
            payload_schema.removeprefix(_PAYLOAD_SCHEMA_PREFIX),
            field="indicator payload schema digest",
        )
        payload_sha256 = _digest(
            self.payload_sha256,
            field="indicator payload_sha256",
        )
        payload_bytes = _integer(
            self.payload_bytes,
            field="indicator payload_bytes",
            minimum=1,
        )
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "feature_count", feature_count)
        object.__setattr__(self, "available_value_count", available_count)
        object.__setattr__(self, "first_event_time_ms", first_event)
        object.__setattr__(self, "last_event_time_ms", last_event)
        object.__setattr__(self, "payload_schema", payload_schema)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "payload_bytes", payload_bytes)

    def digest_payload(self) -> dict[str, object]:
        return {
            "available_value_count": self.available_value_count,
            "feature_count": self.feature_count,
            "first_event_time_ms": self.first_event_time_ms,
            "last_event_time_ms": self.last_event_time_ms,
            "payload_bytes": self.payload_bytes,
            "payload_schema": self.payload_schema,
            "payload_sha256": self.payload_sha256,
            "row_count": self.row_count,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        }


@dataclass(frozen=True, slots=True)
class PostgresIndicatorInventory:
    """Exact metadata closure for one verified PostgreSQL indicator cache."""

    cache_id: str
    source_manifest_digest: str
    market: str
    symbols: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    feature_config_digest: str
    timeframes: tuple[str, ...]
    artifacts: tuple[PostgresIndicatorArtifactMetadata, ...]

    def __post_init__(self) -> None:
        cache_id = _string(self.cache_id, field="indicator inventory cache_id")
        source_digest = _digest(
            self.source_manifest_digest,
            field="indicator inventory source_manifest_digest",
        )
        market = _string(self.market, field="indicator inventory market")
        if market != _SUPPORTED_MARKET:
            raise ValueError("indicator inventory supports only usds-m")
        symbols = _strings(self.symbols, field="indicator inventory symbols")
        timeframes = _strings(
            self.timeframes,
            field="indicator inventory timeframes",
        )
        start = _aware(self.start_time, field="indicator inventory start_time")
        end = _aware(self.end_time, field="indicator inventory end_time")
        if end <= start:
            raise ValueError("indicator inventory time range is invalid")
        feature_digest = _digest(
            self.feature_config_digest,
            field="indicator inventory feature_config_digest",
        )
        artifacts = tuple(self.artifacts)
        if any(
            not isinstance(item, PostgresIndicatorArtifactMetadata)
            for item in artifacts
        ):
            raise TypeError("indicator inventory contains invalid artifact metadata")
        expected_keys = tuple(
            (symbol, timeframe) for symbol in symbols for timeframe in timeframes
        )
        observed_keys = tuple((item.symbol, item.timeframe) for item in artifacts)
        if observed_keys != expected_keys:
            raise ValueError("indicator inventory artifact metadata closure mismatch")
        object.__setattr__(self, "cache_id", cache_id)
        object.__setattr__(self, "source_manifest_digest", source_digest)
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "feature_config_digest", feature_digest)
        object.__setattr__(self, "timeframes", timeframes)
        object.__setattr__(self, "artifacts", artifacts)

    def artifact_for(
        self,
        symbol: str,
        timeframe: str,
    ) -> PostgresIndicatorArtifactMetadata:
        for artifact in self.artifacts:
            if artifact.symbol == symbol and artifact.timeframe == timeframe:
                return artifact
        raise KeyError((symbol, timeframe))


def _artifact_metadata(
    row: Sequence[object],
    *,
    expected_feature_count: int,
) -> PostgresIndicatorArtifactMetadata:
    if len(row) != 10:
        raise ValueError("indicator artifact metadata row contract mismatch")
    artifact = PostgresIndicatorArtifactMetadata(
        symbol=row[0],
        timeframe=row[1],
        row_count=row[2],
        feature_count=row[3],
        available_value_count=row[4],
        first_event_time_ms=row[5],
        last_event_time_ms=row[6],
        payload_schema=row[7],
        payload_sha256=row[8],
        payload_bytes=row[9],
    )
    if artifact.feature_count != expected_feature_count:
        raise ValueError(
            f"indicator feature count mismatch for {artifact.timeframe}: "
            f"{artifact.feature_count} != {expected_feature_count}"
        )
    return artifact


def load_postgres_indicator_inventory(
    connection: IndicatorArtifactConnection,
    *,
    cache_id: str = INDICATOR_CACHE_ID,
) -> PostgresIndicatorInventory:
    """Load and verify manifest and artifact metadata without reading NPZ payloads."""

    resolved_cache_id = _string(cache_id, field="indicator inventory cache_id")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT cache_id, schema_version, market, symbols, start_time, end_time,
                   feature_config_digest, feature_specs, artifact_count
            FROM {INDICATOR_MANIFEST_TABLE}
            WHERE cache_id = %s
            """,
            (resolved_cache_id,),
        )
        manifest_row = cursor.fetchone()
        if manifest_row is None:
            raise FileNotFoundError(
                f"indicator manifest is missing: {resolved_cache_id}"
            )
        cursor.execute(
            f"""
            SELECT symbol, timeframe, row_count, feature_count,
                   available_value_count, first_event_time_ms,
                   last_event_time_ms, payload_schema, payload_sha256,
                   payload_bytes
            FROM {INDICATOR_ARTIFACT_TABLE}
            WHERE cache_id = %s
            ORDER BY symbol, timeframe
            """,
            (resolved_cache_id,),
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
    if stored_cache_id != resolved_cache_id or schema_version != _MANIFEST_SCHEMA:
        raise ValueError("indicator manifest identity mismatch")
    resolved_market = _string(market, field="indicator manifest market")
    if resolved_market != _SUPPORTED_MARKET:
        raise ValueError("indicator inventory supports only usds-m")
    symbols = _strings(raw_symbols, field="indicator manifest symbols")
    start = _aware(start_time, field="indicator manifest start_time")
    end = _aware(end_time, field="indicator manifest end_time")
    if end <= start:
        raise ValueError("indicator manifest time range is invalid")
    feature_specs = _mapping(raw_feature_specs, field="indicator feature_specs")
    feature_digest = _digest(
        feature_config_digest,
        field="indicator feature_config_digest",
    )
    if content_digest(feature_specs) != feature_digest:
        raise ValueError("indicator feature config digest mismatch")
    timeframes, feature_counts = _feature_contract(feature_specs)
    expected_artifact_count = len(symbols) * len(timeframes)
    if (
        _integer(artifact_count, field="indicator artifact_count", minimum=1)
        != expected_artifact_count
    ):
        raise ValueError("indicator manifest artifact count mismatch")

    rows_by_key: dict[tuple[str, str], Sequence[object]] = {}
    for row in artifact_rows:
        if len(row) != 10:
            raise ValueError("indicator artifact metadata row contract mismatch")
        symbol = _string(row[0], field="indicator artifact symbol")
        timeframe = _string(row[1], field="indicator artifact timeframe")
        key = (symbol, timeframe)
        if key in rows_by_key:
            raise ValueError(f"duplicate indicator artifact metadata: {key}")
        rows_by_key[key] = row
    expected_keys = {
        (symbol, timeframe) for symbol in symbols for timeframe in timeframes
    }
    actual_keys = set(rows_by_key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise FileNotFoundError(
            "indicator artifact metadata set mismatch: "
            f"missing={missing}, extra={extra}"
        )

    artifacts = tuple(
        _artifact_metadata(
            rows_by_key[(symbol, timeframe)],
            expected_feature_count=feature_counts[timeframe],
        )
        for symbol in symbols
        for timeframe in timeframes
    )
    source_manifest_digest = content_digest(
        {
            "artifact_count": expected_artifact_count,
            "artifacts": tuple(item.digest_payload() for item in artifacts),
            "cache_id": resolved_cache_id,
            "end_time": end.isoformat(),
            "feature_config_digest": feature_digest,
            "feature_specs": feature_specs,
            "market": resolved_market,
            "schema_version": _MANIFEST_SCHEMA,
            "start_time": start.isoformat(),
            "symbols": symbols,
            "timeframes": timeframes,
        }
    )
    return PostgresIndicatorInventory(
        cache_id=resolved_cache_id,
        source_manifest_digest=source_manifest_digest,
        market=resolved_market,
        symbols=symbols,
        start_time=start,
        end_time=end,
        feature_config_digest=feature_digest,
        timeframes=timeframes,
        artifacts=artifacts,
    )


__all__ = [
    "PostgresIndicatorArtifactMetadata",
    "PostgresIndicatorInventory",
    "load_postgres_indicator_inventory",
]
