"""Immutable contracts for verified instruments stored in research infrastructure."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_aware_datetime, require_sha256

STORED_INSTRUMENT_CATALOG_SCHEMA: Final = "stored_instrument_catalog_v1"
_INDICATOR_PAYLOAD_SCHEMA_PREFIX: Final = "npz_native_indicator_v1:"
_SUPPORTED_MARKET: Final = "usds-m"


def _non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _ordered_unique_strings(
    values: tuple[str, ...] | list[str],
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    resolved = tuple(values)
    if (not resolved and not allow_empty) or any(
        not isinstance(item, str) or not item for item in resolved
    ):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise ValueError(f"{field} must be an ordered {qualifier} string sequence")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} must contain unique values")
    return resolved


def _integer(
    value: object,
    *,
    field: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer greater than or equal to {minimum}")
    return value


def _aware(value: datetime, *, field: str) -> datetime:
    require_aware_datetime(value, field=field)
    return value


def _sha256(value: str, *, field: str) -> str:
    require_sha256(value, field=field)
    return value


def _immutable_write(path: Path, payload: bytes, *, field: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"{field} already exists with different content: {path}")
        return path
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


@dataclass(frozen=True, slots=True)
class StoredIndicatorArtifactEvidence:
    """Metadata-only evidence for one verified symbol/timeframe artifact."""

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
        symbol = _non_empty_string(self.symbol, field="indicator artifact symbol")
        timeframe = _non_empty_string(
            self.timeframe,
            field="indicator artifact timeframe",
        )
        row_count = _integer(
            self.row_count,
            field="indicator artifact row_count",
            minimum=1,
        )
        feature_count = _integer(
            self.feature_count,
            field="indicator artifact feature_count",
            minimum=1,
        )
        available_count = _integer(
            self.available_value_count,
            field="indicator artifact available_value_count",
        )
        if available_count > row_count * feature_count:
            raise ValueError(
                "indicator artifact available_value_count exceeds matrix capacity"
            )
        first_event = _integer(
            self.first_event_time_ms,
            field="indicator artifact first_event_time_ms",
        )
        last_event = _integer(
            self.last_event_time_ms,
            field="indicator artifact last_event_time_ms",
        )
        if last_event <= first_event:
            raise ValueError("indicator artifact event-time range is invalid")
        payload_schema = _non_empty_string(
            self.payload_schema,
            field="indicator artifact payload_schema",
        )
        if not payload_schema.startswith(_INDICATOR_PAYLOAD_SCHEMA_PREFIX):
            raise ValueError("indicator artifact payload_schema is unsupported")
        _sha256(
            payload_schema.removeprefix(_INDICATOR_PAYLOAD_SCHEMA_PREFIX),
            field="indicator artifact payload schema digest",
        )
        payload_digest = _sha256(
            self.payload_sha256,
            field="indicator artifact payload_sha256",
        )
        payload_bytes = _integer(
            self.payload_bytes,
            field="indicator artifact payload_bytes",
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
        object.__setattr__(self, "payload_sha256", payload_digest)
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
class StoredIndicatorSourceInventory:
    """Exact metadata closure for a verified indicator-cache manifest."""

    cache_id: str
    source_manifest_digest: str
    market: str
    symbols: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    feature_config_digest: str
    required_timeframes: tuple[str, ...]
    artifacts: tuple[StoredIndicatorArtifactEvidence, ...]

    def __post_init__(self) -> None:
        cache_id = _non_empty_string(self.cache_id, field="source inventory cache_id")
        source_digest = _sha256(
            self.source_manifest_digest,
            field="source inventory manifest digest",
        )
        market = _non_empty_string(self.market, field="source inventory market")
        symbols = _ordered_unique_strings(
            self.symbols,
            field="source inventory symbols",
        )
        timeframes = _ordered_unique_strings(
            self.required_timeframes,
            field="source inventory required_timeframes",
        )
        start = _aware(self.start_time, field="source inventory start_time")
        end = _aware(self.end_time, field="source inventory end_time")
        if end <= start:
            raise ValueError("source inventory time range is invalid")
        feature_digest = _sha256(
            self.feature_config_digest,
            field="source inventory feature_config_digest",
        )
        artifacts = tuple(self.artifacts)
        if any(
            not isinstance(item, StoredIndicatorArtifactEvidence)
            for item in artifacts
        ):
            raise TypeError(
                "source inventory artifacts must be StoredIndicatorArtifactEvidence"
            )
        expected_keys = tuple(
            (symbol, timeframe) for symbol in symbols for timeframe in timeframes
        )
        actual_keys = tuple((item.symbol, item.timeframe) for item in artifacts)
        if actual_keys != expected_keys:
            raise ValueError("stored indicator artifact closure mismatch")
        object.__setattr__(self, "cache_id", cache_id)
        object.__setattr__(self, "source_manifest_digest", source_digest)
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "feature_config_digest", feature_digest)
        object.__setattr__(self, "required_timeframes", timeframes)
        object.__setattr__(self, "artifacts", artifacts)

    def artifact_for(
        self,
        symbol: str,
        timeframe: str,
    ) -> StoredIndicatorArtifactEvidence:
        for artifact in self.artifacts:
            if artifact.symbol == symbol and artifact.timeframe == timeframe:
                return artifact
        raise KeyError((symbol, timeframe))


@dataclass(frozen=True, slots=True)
class StoredInstrumentExclusion:
    """Predeclared reasons one stored symbol cannot enter the eligible universe."""

    symbol: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        symbol = _non_empty_string(self.symbol, field="excluded instrument symbol")
        reasons = _ordered_unique_strings(
            self.reasons,
            field="excluded instrument reasons",
        )
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "reasons", reasons)

    def to_json_dict(self) -> dict[str, object]:
        return {"reasons": list(self.reasons), "symbol": self.symbol}


@dataclass(frozen=True, slots=True)
class StoredInstrumentCatalog:
    """Frozen eligible/excluded universe for one future research generation."""

    source_cache_id: str
    source_manifest_digest: str
    market: str
    feature_config_digest: str
    required_timeframes: tuple[str, ...]
    research_start: datetime
    research_end: datetime
    eligible_symbols: tuple[str, ...]
    excluded_symbols: tuple[StoredInstrumentExclusion, ...]
    per_symbol_artifact_digests: tuple[
        tuple[str, tuple[tuple[str, str], ...]], ...
    ]
    per_symbol_metadata_digests: tuple[tuple[str, str], ...]
    schema_version: str = STORED_INSTRUMENT_CATALOG_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STORED_INSTRUMENT_CATALOG_SCHEMA:
            raise ValueError("unsupported stored instrument catalog schema")
        source_cache_id = _non_empty_string(
            self.source_cache_id,
            field="stored instrument source_cache_id",
        )
        source_digest = _sha256(
            self.source_manifest_digest,
            field="stored instrument source_manifest_digest",
        )
        market = _non_empty_string(self.market, field="stored instrument market")
        if market != _SUPPORTED_MARKET:
            raise ValueError("stored instrument catalog supports only usds-m")
        feature_digest = _sha256(
            self.feature_config_digest,
            field="stored instrument feature_config_digest",
        )
        timeframes = _ordered_unique_strings(
            self.required_timeframes,
            field="stored instrument required_timeframes",
        )
        research_start = _aware(
            self.research_start,
            field="stored instrument research_start",
        )
        research_end = _aware(
            self.research_end,
            field="stored instrument research_end",
        )
        if research_end <= research_start:
            raise ValueError("stored instrument research interval is invalid")
        eligible = _ordered_unique_strings(
            self.eligible_symbols,
            field="stored instrument eligible_symbols",
            allow_empty=True,
        )
        excluded = tuple(self.excluded_symbols)
        if any(not isinstance(item, StoredInstrumentExclusion) for item in excluded):
            raise TypeError(
                "stored instrument exclusions must be StoredInstrumentExclusion"
            )
        excluded_names = tuple(item.symbol for item in excluded)
        _ordered_unique_strings(
            excluded_names,
            field="stored instrument excluded symbols",
            allow_empty=True,
        )
        if not set(eligible).isdisjoint(excluded_names):
            raise ValueError("stored instrument eligible/excluded sets overlap")

        artifact_entries: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for symbol, raw_digests in self.per_symbol_artifact_digests:
            resolved_symbol = _non_empty_string(
                symbol,
                field="stored instrument artifact symbol",
            )
            digests = tuple(raw_digests)
            if tuple(timeframe for timeframe, _ in digests) != timeframes:
                raise ValueError(
                    "stored instrument per-symbol artifact timeframe closure mismatch"
                )
            resolved_digests = tuple(
                (
                    _non_empty_string(
                        timeframe,
                        field="stored instrument artifact timeframe",
                    ),
                    _sha256(
                        digest,
                        field="stored instrument artifact payload digest",
                    ),
                )
                for timeframe, digest in digests
            )
            artifact_entries.append((resolved_symbol, resolved_digests))
        artifact_symbols = tuple(symbol for symbol, _ in artifact_entries)
        _ordered_unique_strings(
            artifact_symbols,
            field="stored instrument artifact symbols",
        )
        if tuple((*eligible, *excluded_names)) != artifact_symbols:
            raise ValueError("stored instrument catalog symbol closure mismatch")

        metadata_entries = tuple(
            (
                _non_empty_string(
                    symbol,
                    field="stored instrument metadata symbol",
                ),
                _sha256(
                    digest,
                    field="stored instrument metadata digest",
                ),
            )
            for symbol, digest in self.per_symbol_metadata_digests
        )
        metadata_symbols = tuple(symbol for symbol, _ in metadata_entries)
        _ordered_unique_strings(
            metadata_symbols,
            field="stored instrument metadata symbols",
            allow_empty=True,
        )
        if not set(metadata_symbols) <= set(artifact_symbols):
            raise ValueError("stored instrument metadata contains unknown symbols")
        if not set(eligible) <= set(metadata_symbols):
            raise ValueError("eligible stored instruments require metadata evidence")

        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("stored instrument catalog digest mismatch")
        object.__setattr__(self, "source_cache_id", source_cache_id)
        object.__setattr__(self, "source_manifest_digest", source_digest)
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "feature_config_digest", feature_digest)
        object.__setattr__(self, "required_timeframes", timeframes)
        object.__setattr__(self, "research_start", research_start)
        object.__setattr__(self, "research_end", research_end)
        object.__setattr__(self, "eligible_symbols", eligible)
        object.__setattr__(self, "excluded_symbols", excluded)
        object.__setattr__(
            self,
            "per_symbol_artifact_digests",
            tuple(artifact_entries),
        )
        object.__setattr__(
            self,
            "per_symbol_metadata_digests",
            metadata_entries,
        )
        object.__setattr__(self, "digest", expected_digest)

    def digest_payload(self) -> dict[str, object]:
        return {
            "eligible_symbols": self.eligible_symbols,
            "excluded_symbols": tuple(
                item.to_json_dict() for item in self.excluded_symbols
            ),
            "feature_config_digest": self.feature_config_digest,
            "market": self.market,
            "per_symbol_artifact_digests": self.per_symbol_artifact_digests,
            "per_symbol_metadata_digests": self.per_symbol_metadata_digests,
            "required_timeframes": self.required_timeframes,
            "research_end": self.research_end.isoformat(),
            "research_start": self.research_start.isoformat(),
            "schema_version": self.schema_version,
            "source_cache_id": self.source_cache_id,
            "source_manifest_digest": self.source_manifest_digest,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "eligible_symbols": list(self.eligible_symbols),
            "excluded_symbols": [
                item.to_json_dict() for item in self.excluded_symbols
            ],
            "feature_config_digest": self.feature_config_digest,
            "market": self.market,
            "per_symbol_artifact_digests": [
                {
                    "artifacts": [
                        {"payload_sha256": digest, "timeframe": timeframe}
                        for timeframe, digest in digests
                    ],
                    "symbol": symbol,
                }
                for symbol, digests in self.per_symbol_artifact_digests
            ],
            "per_symbol_metadata_digests": [
                {"metadata_digest": digest, "symbol": symbol}
                for symbol, digest in self.per_symbol_metadata_digests
            ],
            "required_timeframes": list(self.required_timeframes),
            "research_end": self.research_end.isoformat(),
            "research_start": self.research_start.isoformat(),
            "schema_version": self.schema_version,
            "source_cache_id": self.source_cache_id,
            "source_manifest_digest": self.source_manifest_digest,
        }


def build_stored_instrument_catalog(
    source: StoredIndicatorSourceInventory,
    *,
    research_start: datetime,
    research_end: datetime,
    metadata_digests: Mapping[str, str],
) -> StoredInstrumentCatalog:
    """Freeze one eligible universe from verified source and metadata evidence."""

    if not isinstance(source, StoredIndicatorSourceInventory):
        raise TypeError("source must be StoredIndicatorSourceInventory")
    start = _aware(research_start, field="catalog research_start")
    end = _aware(research_end, field="catalog research_end")
    if end <= start or start < source.start_time or end > source.end_time:
        raise ValueError("catalog research interval is outside source coverage")
    if source.market != _SUPPORTED_MARKET:
        raise ValueError("stored instrument catalog supports only usds-m")
    unknown_metadata = set(metadata_digests) - set(source.symbols)
    if unknown_metadata:
        raise ValueError(
            f"stored instrument metadata contains unknown symbols: "
            f"{sorted(unknown_metadata)}"
        )
    resolved_metadata: dict[str, str] = {}
    for symbol in source.symbols:
        raw_digest = metadata_digests.get(symbol)
        if raw_digest is not None:
            resolved_metadata[symbol] = _sha256(
                raw_digest,
                field=f"stored instrument metadata digest for {symbol}",
            )

    eligible: list[str] = []
    excluded: list[StoredInstrumentExclusion] = []
    per_symbol_artifacts: list[
        tuple[str, tuple[tuple[str, str], ...]]
    ] = []
    for symbol in source.symbols:
        reasons: list[str] = []
        if symbol not in resolved_metadata:
            reasons.append("missing_execution_metadata")
        artifact_digests: list[tuple[str, str]] = []
        for timeframe in source.required_timeframes:
            artifact = source.artifact_for(symbol, timeframe)
            artifact_digests.append((timeframe, artifact.payload_sha256))
            if artifact.available_value_count == 0:
                reasons.append(f"no_available_values:{timeframe}")
        per_symbol_artifacts.append((symbol, tuple(artifact_digests)))
        if reasons:
            excluded.append(
                StoredInstrumentExclusion(symbol=symbol, reasons=tuple(reasons))
            )
        else:
            eligible.append(symbol)

    return StoredInstrumentCatalog(
        source_cache_id=source.cache_id,
        source_manifest_digest=source.source_manifest_digest,
        market=source.market,
        feature_config_digest=source.feature_config_digest,
        required_timeframes=source.required_timeframes,
        research_start=start,
        research_end=end,
        eligible_symbols=tuple(eligible),
        excluded_symbols=tuple(excluded),
        per_symbol_artifact_digests=tuple(per_symbol_artifacts),
        per_symbol_metadata_digests=tuple(
            (symbol, resolved_metadata[symbol])
            for symbol in source.symbols
            if symbol in resolved_metadata
        ),
    )


def write_stored_instrument_catalog(
    path: str | Path,
    catalog: StoredInstrumentCatalog,
) -> Path:
    """Write one canonical catalog or require exact immutable reuse."""

    if not isinstance(catalog, StoredInstrumentCatalog):
        raise TypeError("catalog must be StoredInstrumentCatalog")
    return _immutable_write(
        Path(path),
        canonical_json_bytes(catalog.to_json_dict()),
        field="stored instrument catalog",
    )


def _json_object(path: str | Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("stored instrument catalog must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("stored instrument catalog must be a JSON object")
    return dict(payload)


def _string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field} must be a string list")
    return tuple(value)


def _parse_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be an ISO datetime string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO datetime string") from error
    return _aware(parsed, field=field)


def load_stored_instrument_catalog(path: str | Path) -> StoredInstrumentCatalog:
    """Load a strict catalog and revalidate all closure and digest contracts."""

    payload = _json_object(path)
    required = {
        "digest",
        "eligible_symbols",
        "excluded_symbols",
        "feature_config_digest",
        "market",
        "per_symbol_artifact_digests",
        "per_symbol_metadata_digests",
        "required_timeframes",
        "research_end",
        "research_start",
        "schema_version",
        "source_cache_id",
        "source_manifest_digest",
    }
    if set(payload) != required:
        raise ValueError("stored instrument catalog field closure mismatch")

    raw_excluded = payload["excluded_symbols"]
    if not isinstance(raw_excluded, list):
        raise ValueError("stored instrument exclusions must be a list")
    excluded: list[StoredInstrumentExclusion] = []
    for item in raw_excluded:
        if not isinstance(item, dict) or set(item) != {"reasons", "symbol"}:
            raise ValueError("stored instrument exclusion field closure mismatch")
        excluded.append(
            StoredInstrumentExclusion(
                symbol=_non_empty_string(
                    item["symbol"],
                    field="excluded instrument symbol",
                ),
                reasons=_string_list(
                    item["reasons"],
                    field="excluded instrument reasons",
                ),
            )
        )

    raw_artifacts = payload["per_symbol_artifact_digests"]
    if not isinstance(raw_artifacts, list):
        raise ValueError("stored instrument artifact digests must be a list")
    artifact_entries: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for item in raw_artifacts:
        if not isinstance(item, dict) or set(item) != {"artifacts", "symbol"}:
            raise ValueError("stored instrument artifact entry closure mismatch")
        raw_members = item["artifacts"]
        if not isinstance(raw_members, list):
            raise ValueError("stored instrument artifact members must be a list")
        members: list[tuple[str, str]] = []
        for member in raw_members:
            if not isinstance(member, dict) or set(member) != {
                "payload_sha256",
                "timeframe",
            }:
                raise ValueError("stored instrument artifact member closure mismatch")
            members.append(
                (
                    _non_empty_string(
                        member["timeframe"],
                        field="stored instrument artifact timeframe",
                    ),
                    _non_empty_string(
                        member["payload_sha256"],
                        field="stored instrument artifact digest",
                    ),
                )
            )
        artifact_entries.append(
            (
                _non_empty_string(
                    item["symbol"],
                    field="stored instrument artifact symbol",
                ),
                tuple(members),
            )
        )

    raw_metadata = payload["per_symbol_metadata_digests"]
    if not isinstance(raw_metadata, list):
        raise ValueError("stored instrument metadata digests must be a list")
    metadata_entries: list[tuple[str, str]] = []
    for item in raw_metadata:
        if not isinstance(item, dict) or set(item) != {
            "metadata_digest",
            "symbol",
        }:
            raise ValueError("stored instrument metadata entry closure mismatch")
        metadata_entries.append(
            (
                _non_empty_string(
                    item["symbol"],
                    field="stored instrument metadata symbol",
                ),
                _non_empty_string(
                    item["metadata_digest"],
                    field="stored instrument metadata digest",
                ),
            )
        )

    return StoredInstrumentCatalog(
        source_cache_id=_non_empty_string(
            payload["source_cache_id"],
            field="stored instrument source_cache_id",
        ),
        source_manifest_digest=_non_empty_string(
            payload["source_manifest_digest"],
            field="stored instrument source_manifest_digest",
        ),
        market=_non_empty_string(
            payload["market"],
            field="stored instrument market",
        ),
        feature_config_digest=_non_empty_string(
            payload["feature_config_digest"],
            field="stored instrument feature_config_digest",
        ),
        required_timeframes=_string_list(
            payload["required_timeframes"],
            field="stored instrument required_timeframes",
        ),
        research_start=_parse_datetime(
            payload["research_start"],
            field="stored instrument research_start",
        ),
        research_end=_parse_datetime(
            payload["research_end"],
            field="stored instrument research_end",
        ),
        eligible_symbols=_string_list(
            payload["eligible_symbols"],
            field="stored instrument eligible_symbols",
        ),
        excluded_symbols=tuple(excluded),
        per_symbol_artifact_digests=tuple(artifact_entries),
        per_symbol_metadata_digests=tuple(metadata_entries),
        schema_version=_non_empty_string(
            payload["schema_version"],
            field="stored instrument schema_version",
        ),
        digest=_non_empty_string(
            payload["digest"],
            field="stored instrument digest",
        ),
    )


__all__ = [
    "STORED_INSTRUMENT_CATALOG_SCHEMA",
    "StoredIndicatorArtifactEvidence",
    "StoredIndicatorSourceInventory",
    "StoredInstrumentCatalog",
    "StoredInstrumentExclusion",
    "build_stored_instrument_catalog",
    "load_stored_instrument_catalog",
    "write_stored_instrument_catalog",
]
