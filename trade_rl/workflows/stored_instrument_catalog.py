"""Immutable contracts for verified instruments stored in research infrastructure."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _strings(
    values: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field} must be a string list or tuple")
    resolved = tuple(values)
    if (not resolved and not allow_empty) or any(
        not isinstance(item, str) or not item for item in resolved
    ):
        raise ValueError(f"{field} contains invalid values")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} must contain unique values")
    return resolved


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _aware(value: datetime, *, field: str) -> datetime:
    require_aware_datetime(value, field=field)
    return value


def _digest(value: object, *, field: str) -> str:
    resolved = _string(value, field=field)
    require_sha256(resolved, field=field)
    return resolved


def _immutable_write(path: Path, payload: bytes, *, field: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"{field} already has different content: {path}")
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
        symbol = _string(self.symbol, field="indicator artifact symbol")
        timeframe = _string(self.timeframe, field="indicator artifact timeframe")
        row_count = _integer(self.row_count, field="artifact row_count", minimum=1)
        feature_count = _integer(
            self.feature_count,
            field="artifact feature_count",
            minimum=1,
        )
        available_count = _integer(
            self.available_value_count,
            field="artifact available_value_count",
        )
        if available_count > row_count * feature_count:
            raise ValueError("artifact available values exceed matrix capacity")
        first_event = _integer(
            self.first_event_time_ms,
            field="artifact first_event_time_ms",
        )
        last_event = _integer(
            self.last_event_time_ms,
            field="artifact last_event_time_ms",
        )
        if last_event <= first_event:
            raise ValueError("indicator artifact event-time range is invalid")
        payload_schema = _string(
            self.payload_schema,
            field="indicator artifact payload_schema",
        )
        if not payload_schema.startswith(_INDICATOR_PAYLOAD_SCHEMA_PREFIX):
            raise ValueError("indicator artifact payload_schema is unsupported")
        _digest(
            payload_schema.removeprefix(_INDICATOR_PAYLOAD_SCHEMA_PREFIX),
            field="indicator artifact schema digest",
        )
        payload_sha256 = _digest(
            self.payload_sha256,
            field="indicator artifact payload_sha256",
        )
        payload_bytes = _integer(
            self.payload_bytes,
            field="artifact payload_bytes",
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


@dataclass(frozen=True, slots=True)
class StoredIndicatorSourceInventory:
    """Exact metadata closure for one verified indicator-cache manifest."""

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
        cache_id = _string(self.cache_id, field="source inventory cache_id")
        source_digest = _digest(
            self.source_manifest_digest,
            field="source inventory manifest digest",
        )
        market = _string(self.market, field="source inventory market")
        symbols = _strings(self.symbols, field="source inventory symbols")
        timeframes = _strings(
            self.required_timeframes,
            field="source inventory required_timeframes",
        )
        start = _aware(self.start_time, field="source inventory start_time")
        end = _aware(self.end_time, field="source inventory end_time")
        if end <= start:
            raise ValueError("source inventory time range is invalid")
        feature_digest = _digest(
            self.feature_config_digest,
            field="source inventory feature_config_digest",
        )
        artifacts = tuple(self.artifacts)
        if any(
            not isinstance(item, StoredIndicatorArtifactEvidence) for item in artifacts
        ):
            raise TypeError("source inventory contains an invalid artifact")
        expected = tuple(
            (symbol, timeframe) for symbol in symbols for timeframe in timeframes
        )
        observed = tuple((item.symbol, item.timeframe) for item in artifacts)
        if observed != expected:
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
    """Reasons one stored symbol cannot enter the eligible universe."""

    symbol: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "symbol",
            _string(self.symbol, field="excluded instrument symbol"),
        )
        object.__setattr__(
            self,
            "reasons",
            _strings(self.reasons, field="excluded instrument reasons"),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {"reasons": list(self.reasons), "symbol": self.symbol}


ArtifactDigestRows = tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
MetadataDigestRows = tuple[tuple[str, str], ...]


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
    per_symbol_artifact_digests: ArtifactDigestRows
    per_symbol_metadata_digests: MetadataDigestRows
    schema_version: str = STORED_INSTRUMENT_CATALOG_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STORED_INSTRUMENT_CATALOG_SCHEMA:
            raise ValueError("unsupported stored instrument catalog schema")
        source_cache_id = _string(
            self.source_cache_id,
            field="stored instrument source_cache_id",
        )
        source_digest = _digest(
            self.source_manifest_digest,
            field="stored instrument source manifest digest",
        )
        market = _string(self.market, field="stored instrument market")
        if market != _SUPPORTED_MARKET:
            raise ValueError("stored instrument catalog supports only usds-m")
        feature_digest = _digest(
            self.feature_config_digest,
            field="stored instrument feature digest",
        )
        timeframes = _strings(
            self.required_timeframes,
            field="stored instrument required timeframes",
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
        eligible = _strings(
            self.eligible_symbols,
            field="stored instrument eligible symbols",
            allow_empty=True,
        )
        excluded = tuple(self.excluded_symbols)
        if any(not isinstance(item, StoredInstrumentExclusion) for item in excluded):
            raise TypeError("stored instrument exclusions contain an invalid item")
        excluded_names = _strings(
            tuple(item.symbol for item in excluded),
            field="stored instrument excluded symbols",
            allow_empty=True,
        )
        if not set(eligible).isdisjoint(excluded_names):
            raise ValueError("stored instrument eligible/excluded sets overlap")

        artifacts = _validate_artifact_rows(
            self.per_symbol_artifact_digests,
            timeframes=timeframes,
        )
        artifact_symbols = tuple(symbol for symbol, _ in artifacts)
        declared = set(eligible) | set(excluded_names)
        if set(artifact_symbols) != declared:
            raise ValueError("stored instrument catalog symbol closure mismatch")
        if (
            tuple(symbol for symbol in artifact_symbols if symbol in eligible)
            != eligible
        ):
            raise ValueError("stored instrument eligible symbol order mismatch")
        if (
            tuple(symbol for symbol in artifact_symbols if symbol in excluded_names)
            != excluded_names
        ):
            raise ValueError("stored instrument excluded symbol order mismatch")

        metadata = _validate_metadata_rows(self.per_symbol_metadata_digests)
        metadata_symbols = tuple(symbol for symbol, _ in metadata)
        if not set(metadata_symbols) <= set(artifact_symbols):
            raise ValueError("stored instrument metadata contains unknown symbols")
        if not set(eligible) <= set(metadata_symbols):
            raise ValueError("eligible stored instruments require metadata evidence")
        expected_metadata_order = tuple(
            symbol for symbol in artifact_symbols if symbol in metadata_symbols
        )
        if metadata_symbols != expected_metadata_order:
            raise ValueError("stored instrument metadata symbol order mismatch")

        object.__setattr__(self, "source_cache_id", source_cache_id)
        object.__setattr__(self, "source_manifest_digest", source_digest)
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "feature_config_digest", feature_digest)
        object.__setattr__(self, "required_timeframes", timeframes)
        object.__setattr__(self, "research_start", research_start)
        object.__setattr__(self, "research_end", research_end)
        object.__setattr__(self, "eligible_symbols", eligible)
        object.__setattr__(self, "excluded_symbols", excluded)
        object.__setattr__(self, "per_symbol_artifact_digests", artifacts)
        object.__setattr__(self, "per_symbol_metadata_digests", metadata)
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("stored instrument catalog digest mismatch")
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
            "excluded_symbols": [item.to_json_dict() for item in self.excluded_symbols],
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


def _validate_artifact_rows(
    rows: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    *,
    timeframes: tuple[str, ...],
) -> ArtifactDigestRows:
    resolved: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for raw_symbol, raw_digests in rows:
        symbol = _string(raw_symbol, field="stored artifact symbol")
        digests = tuple(raw_digests)
        if tuple(timeframe for timeframe, _ in digests) != timeframes:
            raise ValueError("stored artifact timeframe closure mismatch")
        resolved.append(
            (
                symbol,
                tuple(
                    (
                        _string(timeframe, field="stored artifact timeframe"),
                        _digest(digest, field="stored artifact digest"),
                    )
                    for timeframe, digest in digests
                ),
            )
        )
    symbols = tuple(symbol for symbol, _ in resolved)
    _strings(symbols, field="stored artifact symbols")
    return tuple(resolved)


def _validate_metadata_rows(rows: Sequence[tuple[str, str]]) -> MetadataDigestRows:
    resolved = tuple(
        (
            _string(symbol, field="stored metadata symbol"),
            _digest(digest, field="stored metadata digest"),
        )
        for symbol, digest in rows
    )
    _strings(
        tuple(symbol for symbol, _ in resolved),
        field="stored metadata symbols",
        allow_empty=True,
    )
    return resolved


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
    unknown = set(metadata_digests) - set(source.symbols)
    if unknown:
        raise ValueError(f"stored metadata contains unknown symbols: {sorted(unknown)}")
    metadata = {
        symbol: _digest(digest, field=f"metadata digest for {symbol}")
        for symbol, digest in metadata_digests.items()
    }

    eligible: list[str] = []
    excluded: list[StoredInstrumentExclusion] = []
    artifacts: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for symbol in source.symbols:
        reasons: list[str] = []
        if symbol not in metadata:
            reasons.append("missing_execution_metadata")
        symbol_artifacts: list[tuple[str, str]] = []
        for timeframe in source.required_timeframes:
            artifact = source.artifact_for(symbol, timeframe)
            symbol_artifacts.append((timeframe, artifact.payload_sha256))
            if artifact.available_value_count == 0:
                reasons.append(f"no_available_values:{timeframe}")
        artifacts.append((symbol, tuple(symbol_artifacts)))
        if reasons:
            excluded.append(StoredInstrumentExclusion(symbol, tuple(reasons)))
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
        per_symbol_artifact_digests=tuple(artifacts),
        per_symbol_metadata_digests=tuple(
            (symbol, metadata[symbol])
            for symbol in source.symbols
            if symbol in metadata
        ),
    )


def write_stored_instrument_catalog(
    path: str | Path,
    catalog: StoredInstrumentCatalog,
) -> Path:
    if not isinstance(catalog, StoredInstrumentCatalog):
        raise TypeError("catalog must be StoredInstrumentCatalog")
    return _immutable_write(
        Path(path),
        canonical_json_bytes(catalog.to_json_dict()),
        field="stored instrument catalog",
    )


def load_stored_instrument_catalog(path: str | Path) -> StoredInstrumentCatalog:
    """Load a strict catalog and revalidate all closure and digest contracts."""

    payload = _json_object(path, field="stored instrument catalog")
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
    return StoredInstrumentCatalog(
        source_cache_id=_string(
            payload["source_cache_id"],
            field="stored source cache ID",
        ),
        source_manifest_digest=_string(
            payload["source_manifest_digest"],
            field="stored source manifest digest",
        ),
        market=_string(payload["market"], field="stored market"),
        feature_config_digest=_string(
            payload["feature_config_digest"],
            field="stored feature digest",
        ),
        required_timeframes=_strings(
            payload["required_timeframes"],
            field="stored required timeframes",
        ),
        research_start=_datetime(
            payload["research_start"],
            field="stored research start",
        ),
        research_end=_datetime(
            payload["research_end"],
            field="stored research end",
        ),
        eligible_symbols=_strings(
            payload["eligible_symbols"],
            field="stored eligible symbols",
            allow_empty=True,
        ),
        excluded_symbols=_load_exclusions(payload["excluded_symbols"]),
        per_symbol_artifact_digests=_load_artifact_rows(
            payload["per_symbol_artifact_digests"]
        ),
        per_symbol_metadata_digests=_load_metadata_rows(
            payload["per_symbol_metadata_digests"]
        ),
        schema_version=_string(
            payload["schema_version"],
            field="stored catalog schema",
        ),
        digest=_string(payload["digest"], field="stored catalog digest"),
    )


def _json_object(path: str | Path, *, field: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must be a JSON object")
    return dict(payload)


def _datetime(value: object, *, field: str) -> datetime:
    raw = _string(value, field=field)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO datetime") from error
    return _aware(parsed, field=field)


def _load_exclusions(value: object) -> tuple[StoredInstrumentExclusion, ...]:
    if not isinstance(value, list):
        raise ValueError("stored exclusions must be a list")
    result: list[StoredInstrumentExclusion] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"reasons", "symbol"}:
            raise ValueError("stored exclusion field closure mismatch")
        result.append(
            StoredInstrumentExclusion(
                _string(item["symbol"], field="stored excluded symbol"),
                _strings(item["reasons"], field="stored exclusion reasons"),
            )
        )
    return tuple(result)


def _load_artifact_rows(value: object) -> ArtifactDigestRows:
    if not isinstance(value, list):
        raise ValueError("stored artifact rows must be a list")
    result: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"artifacts", "symbol"}:
            raise ValueError("stored artifact row closure mismatch")
        raw_artifacts = row["artifacts"]
        if not isinstance(raw_artifacts, list):
            raise ValueError("stored artifact members must be a list")
        members: list[tuple[str, str]] = []
        for item in raw_artifacts:
            if not isinstance(item, dict) or set(item) != {
                "payload_sha256",
                "timeframe",
            }:
                raise ValueError("stored artifact member closure mismatch")
            members.append(
                (
                    _string(item["timeframe"], field="stored timeframe"),
                    _string(item["payload_sha256"], field="stored payload digest"),
                )
            )
        result.append(
            (
                _string(row["symbol"], field="stored artifact symbol"),
                tuple(members),
            )
        )
    return tuple(result)


def _load_metadata_rows(value: object) -> MetadataDigestRows:
    if not isinstance(value, list):
        raise ValueError("stored metadata rows must be a list")
    result: list[tuple[str, str]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "metadata_digest",
            "symbol",
        }:
            raise ValueError("stored metadata row closure mismatch")
        result.append(
            (
                _string(row["symbol"], field="stored metadata symbol"),
                _string(row["metadata_digest"], field="stored metadata digest"),
            )
        )
    return tuple(result)


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
