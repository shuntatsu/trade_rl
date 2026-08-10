"""Metadata-only discovery for the maintained PostgreSQL indicator cache."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final, cast

from trade_rl.artifacts.hashing import content_digest
from trade_rl.catalog.stored_instrument_catalog import (
    StoredIndicatorArtifactEvidence,
    StoredIndicatorSourceInventory,
)
from trade_rl.domain.common import require_aware_datetime, require_sha256
from trade_rl.integrations.postgres_indicator_artifacts import (
    INDICATOR_ARTIFACT_TABLE,
    INDICATOR_CACHE_ID,
    INDICATOR_MANIFEST_TABLE,
    IndicatorArtifactConnection,
)

_MANIFEST_SCHEMA: Final = "native_indicator_cache_v1"
_SUPPORTED_MARKET: Final = "usds-m"
_SOURCE_DIGEST_SCHEMA: Final = "postgres_indicator_source_inventory_v1"


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _strings(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a string list or tuple")
    resolved = tuple(value)
    if (not resolved and not allow_empty) or any(
        not isinstance(item, str) or not item for item in resolved
    ):
        raise ValueError(f"{field} contains invalid values")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} must contain unique values")
    return cast(tuple[str, ...], resolved)


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _digest(value: object, *, field: str) -> str:
    resolved = _string(value, field=field)
    require_sha256(resolved, field=field)
    return resolved


def _aware_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be a datetime")
    require_aware_datetime(value, field=field)
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _feature_counts_by_timeframe(
    feature_specs: Mapping[str, object],
) -> tuple[tuple[str, ...], dict[str, int]]:
    base_timeframe = _string(
        feature_specs.get("base_timeframe"),
        field="feature_specs.base_timeframe",
    )
    additional_timeframes = _strings(
        feature_specs.get("feature_timeframes"),
        field="feature_specs.feature_timeframes",
        allow_empty=True,
    )
    timeframes = (base_timeframe, *additional_timeframes)
    if len(set(timeframes)) != len(timeframes):
        raise ValueError("maintained indicator timeframes must be unique")

    raw_features = feature_specs.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise ValueError("feature_specs.features must be a non-empty list")

    counts = {timeframe: 0 for timeframe in timeframes}
    feature_names: set[str] = set()
    for index, raw_feature in enumerate(raw_features):
        feature = _mapping(
            raw_feature,
            field=f"feature_specs.features[{index}]",
        )
        name = _string(
            feature.get("name"),
            field=f"feature_specs.features[{index}].name",
        )
        if name in feature_names:
            raise ValueError("indicator feature names must be unique")
        timeframe = feature.get("timeframe", base_timeframe)
        if not isinstance(timeframe, str) or timeframe not in counts:
            raise ValueError(f"indicator feature has unsupported timeframe: {name}")
        counts[timeframe] += 1
        feature_names.add(name)

    if any(count == 0 for count in counts.values()):
        raise ValueError("each maintained timeframe must contain features")
    return timeframes, counts


def _parse_manifest(
    row: Sequence[object],
    *,
    cache_id: str,
) -> tuple[
    str,
    tuple[str, ...],
    datetime,
    datetime,
    str,
    tuple[str, ...],
    dict[str, int],
    int,
]:
    if len(row) != 9:
        raise ValueError("indicator manifest row contract mismatch")

    stored_cache_id = _string(row[0], field="indicator manifest cache_id")
    schema_version = _string(row[1], field="indicator manifest schema_version")
    if stored_cache_id != cache_id or schema_version != _MANIFEST_SCHEMA:
        raise ValueError("indicator manifest identity mismatch")

    market = _string(row[2], field="indicator manifest market")
    if market != _SUPPORTED_MARKET:
        raise ValueError("indicator manifest market is unsupported")
    symbols = _strings(row[3], field="indicator manifest symbols")
    start_time = _aware_datetime(row[4], field="indicator manifest start_time")
    end_time = _aware_datetime(row[5], field="indicator manifest end_time")
    if end_time <= start_time:
        raise ValueError("indicator manifest time range is invalid")

    feature_config_digest = _digest(
        row[6],
        field="indicator manifest feature_config_digest",
    )
    feature_specs = _mapping(row[7], field="indicator manifest feature_specs")
    if content_digest(feature_specs) != feature_config_digest:
        raise ValueError("indicator manifest feature config digest mismatch")
    timeframes, feature_counts = _feature_counts_by_timeframe(feature_specs)

    artifact_count = _integer(
        row[8],
        field="indicator manifest artifact_count",
        minimum=1,
    )
    if artifact_count != len(symbols) * len(timeframes):
        raise ValueError("indicator manifest artifact count mismatch")

    return (
        market,
        symbols,
        start_time,
        end_time,
        feature_config_digest,
        timeframes,
        feature_counts,
        artifact_count,
    )


def _parse_artifact_metadata(
    row: Sequence[object],
    *,
    feature_counts: Mapping[str, int],
) -> StoredIndicatorArtifactEvidence:
    if len(row) != 10:
        raise ValueError("indicator artifact metadata row contract mismatch")

    symbol = _string(row[0], field="indicator artifact symbol")
    timeframe = _string(row[1], field="indicator artifact timeframe")
    row_count = _integer(
        row[2],
        field="indicator artifact row_count",
        minimum=1,
    )
    feature_count = _integer(
        row[3],
        field="indicator artifact feature_count",
        minimum=1,
    )
    expected_feature_count = feature_counts.get(timeframe)
    if expected_feature_count is not None and feature_count != expected_feature_count:
        raise ValueError("indicator artifact feature count mismatch")

    return StoredIndicatorArtifactEvidence(
        symbol=symbol,
        timeframe=timeframe,
        row_count=row_count,
        feature_count=feature_count,
        available_value_count=_integer(
            row[4],
            field="indicator artifact available_value_count",
        ),
        first_event_time_ms=_integer(
            row[5],
            field="indicator artifact first_event_time_ms",
        ),
        last_event_time_ms=_integer(
            row[6],
            field="indicator artifact last_event_time_ms",
        ),
        payload_schema=_string(
            row[7],
            field="indicator artifact payload_schema",
        ),
        payload_sha256=_digest(
            row[8],
            field="indicator artifact payload_sha256",
        ),
        payload_bytes=_integer(
            row[9],
            field="indicator artifact payload_bytes",
            minimum=1,
        ),
    )


def _artifact_digest_payload(
    artifact: StoredIndicatorArtifactEvidence,
) -> dict[str, object]:
    return {
        "available_value_count": artifact.available_value_count,
        "feature_count": artifact.feature_count,
        "first_event_time_ms": artifact.first_event_time_ms,
        "last_event_time_ms": artifact.last_event_time_ms,
        "payload_bytes": artifact.payload_bytes,
        "payload_schema": artifact.payload_schema,
        "payload_sha256": artifact.payload_sha256,
        "row_count": artifact.row_count,
        "symbol": artifact.symbol,
        "timeframe": artifact.timeframe,
    }


def load_postgres_indicator_source_inventory(
    connection: IndicatorArtifactConnection,
    *,
    cache_id: str = INDICATOR_CACHE_ID,
) -> StoredIndicatorSourceInventory:
    """Load verified manifest and artifact metadata without reading NPZ payloads."""

    requested_cache_id = _string(cache_id, field="indicator inventory cache_id")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT cache_id, schema_version, market, symbols, start_time, end_time,
                   feature_config_digest, feature_specs, artifact_count
            FROM {INDICATOR_MANIFEST_TABLE}
            WHERE cache_id = %s
            """,
            (requested_cache_id,),
        )
        manifest_row = cursor.fetchone()
        if manifest_row is None:
            raise FileNotFoundError(
                f"indicator manifest is missing: {requested_cache_id}"
            )

        cursor.execute(
            f"""
            SELECT symbol, timeframe, row_count, feature_count,
                   available_value_count, first_event_time_ms, last_event_time_ms,
                   payload_schema, payload_sha256, payload_bytes
            FROM {INDICATOR_ARTIFACT_TABLE}
            WHERE cache_id = %s
            ORDER BY symbol, timeframe
            """,
            (requested_cache_id,),
        )
        artifact_rows = tuple(cursor.fetchall())

    (
        market,
        symbols,
        start_time,
        end_time,
        feature_config_digest,
        timeframes,
        feature_counts,
        artifact_count,
    ) = _parse_manifest(manifest_row, cache_id=requested_cache_id)

    artifacts_by_key: dict[tuple[str, str], StoredIndicatorArtifactEvidence] = {}
    for row in artifact_rows:
        artifact = _parse_artifact_metadata(
            row,
            feature_counts=feature_counts,
        )
        key = (artifact.symbol, artifact.timeframe)
        if key in artifacts_by_key:
            raise ValueError(f"duplicate indicator artifact metadata: {key}")
        artifacts_by_key[key] = artifact

    expected_keys = tuple(
        (symbol, timeframe) for symbol in symbols for timeframe in timeframes
    )
    expected_key_set = set(expected_keys)
    observed_key_set = set(artifacts_by_key)
    if observed_key_set != expected_key_set or len(artifact_rows) != artifact_count:
        missing = sorted(expected_key_set - observed_key_set)
        extra = sorted(observed_key_set - expected_key_set)
        raise FileNotFoundError(
            "indicator artifact metadata set mismatch: "
            f"missing={missing}, extra={extra}"
        )

    artifacts = tuple(artifacts_by_key[key] for key in expected_keys)
    schema_by_timeframe: dict[str, str] = {}
    for artifact in artifacts:
        previous_schema = schema_by_timeframe.setdefault(
            artifact.timeframe,
            artifact.payload_schema,
        )
        if previous_schema != artifact.payload_schema:
            raise ValueError("indicator payload schema differs within timeframe")

    source_manifest_digest = content_digest(
        {
            "artifact_count": artifact_count,
            "artifacts": tuple(
                _artifact_digest_payload(artifact) for artifact in artifacts
            ),
            "cache_id": requested_cache_id,
            "end_time": end_time,
            "feature_config_digest": feature_config_digest,
            "market": market,
            "required_timeframes": timeframes,
            "schema_version": _SOURCE_DIGEST_SCHEMA,
            "source_schema_version": _MANIFEST_SCHEMA,
            "start_time": start_time,
            "symbols": symbols,
        }
    )
    return StoredIndicatorSourceInventory(
        cache_id=requested_cache_id,
        source_manifest_digest=source_manifest_digest,
        market=market,
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        feature_config_digest=feature_config_digest,
        required_timeframes=timeframes,
        artifacts=artifacts,
    )


__all__ = ["load_postgres_indicator_source_inventory"]
