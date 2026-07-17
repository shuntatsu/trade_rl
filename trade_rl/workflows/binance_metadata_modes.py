"""Explicit, identity-bound Binance execution-metadata research modes."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.domain.common import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)
from trade_rl.integrations.binance import (
    BinanceExchangeInfoSnapshot,
    BinanceMarket,
    BinanceTransportMode,
)

_METADATA_EVIDENCE_SCHEMA = "binance_metadata_evidence_v1"
_CONSERVATIVE_STATIC_SCHEMA = "binance_conservative_static_v1"
_POLICY_VERSION = "binance_metadata_modes_v1"
_LIMITATION = (
    "Current exchange rules are applied statically across the research interval; "
    "they are not historical point-in-time observations."
)

MetadataValue: TypeAlias = str | float
MetadataEntry: TypeAlias = Mapping[str, MetadataValue]
MetadataMap: TypeAlias = Mapping[str, MetadataEntry]
HistoryMap: TypeAlias = Mapping[str, tuple[InstrumentExecutionRule, ...]]


class BinanceMetadataMode(StrEnum):
    """Declared source and integrity mode for Binance execution metadata."""

    HISTORICAL_SIGNED = "historical_signed"
    FROZEN_SNAPSHOT = "frozen_snapshot"
    CONSERVATIVE_STATIC = "conservative_static"


class ExchangeSnapshotTransport(Protocol):
    """Narrow transport contract needed by frozen-snapshot resolution."""

    def load_exchange_information_snapshot(
        self,
        *,
        market: BinanceMarket | str,
        mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
    ) -> BinanceExchangeInfoSnapshot: ...


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return cast(Mapping[str, object], _freeze(value))


def _freeze_metadata(value: Mapping[str, Mapping[str, MetadataValue]]) -> MetadataMap:
    return cast(MetadataMap, _freeze(value))


def _freeze_histories(
    value: Mapping[str, Sequence[InstrumentExecutionRule]] | None,
) -> HistoryMap | None:
    if value is None:
        return None
    return MappingProxyType(
        {str(symbol): tuple(rules) for symbol, rules in value.items()}
    )


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be a valid ISO-8601 timestamp") from error
    require_aware_datetime(parsed, field=field)
    return parsed.astimezone(UTC)


def _utc(value: datetime, *, field: str) -> datetime:
    return require_aware_datetime(value, field=field).astimezone(UTC)


def _iso(value: datetime, *, field: str) -> str:
    return _utc(value, field=field).isoformat()


def _positive(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{field} must be positive")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be positive") from error
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{field} must be positive")
    return resolved


def _strictly_positive(value: object, *, field: str) -> float:
    try:
        return _positive(value, field=field)
    except ValueError as error:
        raise ValueError(f"{field} must be strictly positive") from error


def _epoch_milliseconds(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive epoch timestamp")
    try:
        resolved = int(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a positive epoch timestamp") from error
    while abs(resolved) >= 10_000_000_000_000:
        resolved //= 1_000
    if resolved <= 0:
        raise ValueError(f"{field} must be a positive epoch timestamp")
    return resolved


def _filter_value(
    filters: Sequence[object],
    *,
    symbol: str,
    filter_type: str,
    fields: tuple[str, ...],
    semantic_field: str,
    required: bool = True,
) -> float:
    for item in filters:
        if not isinstance(item, Mapping) or item.get("filterType") != filter_type:
            continue
        for name in fields:
            if name in item:
                return _positive(item[name], field=f"{symbol}.{semantic_field}")
    if required:
        raise ValueError(f"Binance symbol {symbol} is missing {filter_type}")
    return 0.0


def _metadata_from_snapshot(
    snapshot: BinanceExchangeInfoSnapshot,
    *,
    symbols: tuple[str, ...],
) -> dict[str, dict[str, MetadataValue]]:
    raw_symbols = snapshot.payload.get("symbols")
    if not isinstance(raw_symbols, Sequence) or isinstance(
        raw_symbols, (str, bytes, bytearray)
    ):
        raise ValueError("Binance exchange information lacks symbols")
    by_symbol: dict[str, Mapping[str, object]] = {}
    for raw in raw_symbols:
        if isinstance(raw, Mapping) and isinstance(raw.get("symbol"), str):
            by_symbol[str(raw["symbol"])] = cast(Mapping[str, object], raw)

    result: dict[str, dict[str, MetadataValue]] = {}
    for symbol in symbols:
        item = by_symbol.get(symbol)
        if item is None:
            raise ValueError(f"Binance exchange information has no symbol {symbol}")
        status = item.get("contractStatus", item.get("status"))
        if status != "TRADING":
            raise ValueError(f"Binance symbol {symbol} is not trading: {status}")
        raw_filters = item.get("filters")
        if not isinstance(raw_filters, Sequence) or isinstance(
            raw_filters, (str, bytes, bytearray)
        ):
            raise ValueError(f"Binance symbol {symbol} lacks filters")
        tick_size = _filter_value(
            raw_filters,
            symbol=symbol,
            filter_type="PRICE_FILTER",
            fields=("tickSize",),
            semantic_field="tick_size",
        )
        lot_size = _filter_value(
            raw_filters,
            symbol=symbol,
            filter_type="LOT_SIZE",
            fields=("stepSize",),
            semantic_field="lot_size",
        )
        minimum_notional = max(
            _filter_value(
                raw_filters,
                symbol=symbol,
                filter_type="MIN_NOTIONAL",
                fields=("notional", "minNotional"),
                semantic_field="minimum_notional",
                required=False,
            ),
            _filter_value(
                raw_filters,
                symbol=symbol,
                filter_type="NOTIONAL",
                fields=("minNotional",),
                semantic_field="minimum_notional",
                required=False,
            ),
        )
        if minimum_notional <= 0.0:
            raise ValueError(f"Binance symbol {symbol} is missing minimum_notional")
        listed_ms = _epoch_milliseconds(
            item.get("onboardDate"), field=f"{symbol}.onboardDate"
        )
        result[symbol] = {
            "listed_at": datetime.fromtimestamp(listed_ms / 1_000, tz=UTC).isoformat(),
            "tick_size": tick_size,
            "lot_size": lot_size,
            "minimum_notional": minimum_notional,
        }
    return result


def _validate_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    result = tuple(
        require_non_empty(str(symbol), field="symbols") for symbol in symbols
    )
    if not result:
        raise ValueError("symbols must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("symbols must be unique")
    return result


def _identity(
    *,
    mode: BinanceMetadataMode,
    market: str,
    symbols: tuple[str, ...],
    source_uri: str,
    as_of: str,
    start_time: datetime,
    end_time: datetime,
    authentication: str,
    point_in_time: bool,
    limitations: tuple[str, ...],
    source_payload_digest: str,
    raw_payload_sha256: str | None = None,
    stress_factors: Mapping[str, float] | None = None,
) -> Mapping[str, object]:
    if end_time <= start_time:
        raise ValueError("metadata coverage end_time must be later than start_time")
    payload: dict[str, object] = {
        "schema_version": _METADATA_EVIDENCE_SCHEMA,
        "policy_version": _POLICY_VERSION,
        "mode": mode.value,
        "market": market,
        "symbols": symbols,
        "source_uri": require_non_empty(source_uri, field="source_uri"),
        "as_of": as_of,
        "coverage": {
            "start_time": _iso(start_time, field="start_time"),
            "end_time": _iso(end_time, field="end_time"),
            "application": (
                "effective-dated-full-interval"
                if point_in_time
                else "static-full-interval"
            ),
        },
        "authentication": authentication,
        "point_in_time": point_in_time,
        "limitations": limitations,
        "source_payload_digest": require_sha256(
            source_payload_digest, field="source_payload_digest"
        ),
    }
    if raw_payload_sha256 is not None:
        payload["raw_payload_sha256"] = require_sha256(
            raw_payload_sha256, field="raw_payload_sha256"
        )
    if stress_factors is not None:
        payload["stress_factors"] = dict(stress_factors)
    return _freeze_mapping(payload)


def _write_new(path: Path, content: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"metadata evidence artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"metadata evidence temporary file already exists: {temporary}"
        )
    try:
        temporary.write_bytes(content)
        if path.exists():
            raise FileExistsError(f"metadata evidence artifact already exists: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class BinanceHistoricalSignedScope:
    """Authenticated semantic scope for one point-in-time Binance rule history."""

    market: str
    symbols: tuple[str, ...]
    coverage_start: datetime
    coverage_end: datetime
    issued_at: datetime
    source_uri: str
    payload_digest: str
    policy_version: str = _POLICY_VERSION

    def __post_init__(self) -> None:
        try:
            market = BinanceMarket(self.market)
        except ValueError as error:
            raise ValueError("signed history market is unsupported") from error
        if market is not BinanceMarket.USDS_M:
            raise ValueError("signed history market must be Binance USD-M")
        symbols = _validate_symbols(self.symbols)
        coverage_start = _utc(self.coverage_start, field="coverage_start")
        coverage_end = _utc(self.coverage_end, field="coverage_end")
        issued_at = _utc(self.issued_at, field="issued_at")
        if coverage_end <= coverage_start:
            raise ValueError("signed history coverage end must follow start")
        if self.policy_version != _POLICY_VERSION:
            raise ValueError("signed history policy version is unsupported")
        object.__setattr__(self, "market", market.value)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "coverage_start", coverage_start)
        object.__setattr__(self, "coverage_end", coverage_end)
        object.__setattr__(self, "issued_at", issued_at)
        require_non_empty(self.source_uri, field="source_uri")
        require_sha256(self.payload_digest, field="payload_digest")


@dataclass(frozen=True, slots=True)
class BinanceMetadataResolution:
    """One immutable metadata resolution reused by every dataset build in a run."""

    mode: BinanceMetadataMode
    metadata: MetadataMap
    execution_rule_histories: HistoryMap | None
    identity_evidence: Mapping[str, object]
    evidence_digest: str
    source_uri: str
    raw_payload: bytes | None = None

    def report_payload(self) -> Mapping[str, object]:
        evidence = dict(self.identity_evidence)
        return _freeze_mapping(
            {
                **evidence,
                "evidence_digest": self.evidence_digest,
                "metadata": self.metadata,
                "production_status": "NO-GO",
            }
        )

    def write_artifacts(self, root: Path) -> None:
        report_path = root / "exchange-info.json"
        raw_path = root / "exchange-info.raw.json"
        targets = [report_path]
        if self.raw_payload is not None:
            targets.append(raw_path)
        existing = next((path for path in targets if path.exists()), None)
        if existing is not None:
            raise FileExistsError(
                f"metadata evidence artifact already exists: {existing}"
            )
        if self.raw_payload is not None:
            _write_new(raw_path, self.raw_payload)
        _write_new(report_path, canonical_json_bytes(self.report_payload()) + b"\n")


class BinanceMetadataResolutionProvider:
    """Lazily resolve metadata once and return the same object thereafter."""

    def __init__(self, resolver: Callable[[], BinanceMetadataResolution]) -> None:
        self._resolver = resolver
        self._resolution: BinanceMetadataResolution | None = None

    def get(self) -> BinanceMetadataResolution:
        if self._resolution is None:
            self._resolution = self._resolver()
        return self._resolution


def resolve_frozen_snapshot(
    *,
    transport: ExchangeSnapshotTransport,
    market: BinanceMarket | str,
    symbols: Sequence[str],
    start_time: datetime,
    end_time: datetime,
) -> BinanceMetadataResolution:
    resolved_symbols = _validate_symbols(symbols)
    resolved_market = BinanceMarket(market)
    if resolved_market is not BinanceMarket.USDS_M:
        raise ValueError("frozen_snapshot currently supports Binance USD-M only")
    snapshot = transport.load_exchange_information_snapshot(
        market=resolved_market, mode=BinanceTransportMode.REST
    )
    observed_digest = sha256(snapshot.raw_payload).hexdigest()
    if observed_digest != snapshot.raw_payload_sha256:
        raise ValueError("Binance exchange snapshot raw payload digest mismatch")
    metadata = _metadata_from_snapshot(snapshot, symbols=resolved_symbols)
    identity = _identity(
        mode=BinanceMetadataMode.FROZEN_SNAPSHOT,
        market=resolved_market.value,
        symbols=resolved_symbols,
        source_uri=snapshot.source_uri,
        as_of=_iso(snapshot.retrieved_at, field="retrieved_at"),
        start_time=start_time,
        end_time=end_time,
        authentication="none",
        point_in_time=False,
        limitations=(_LIMITATION,),
        source_payload_digest=snapshot.raw_payload_sha256,
        raw_payload_sha256=snapshot.raw_payload_sha256,
    )
    return BinanceMetadataResolution(
        mode=BinanceMetadataMode.FROZEN_SNAPSHOT,
        metadata=_freeze_metadata(metadata),
        execution_rule_histories=None,
        identity_evidence=identity,
        evidence_digest=content_digest(identity),
        source_uri=snapshot.source_uri,
        raw_payload=bytes(snapshot.raw_payload),
    )


def _signed_metadata(
    metadata: Mapping[str, Mapping[str, MetadataValue]],
    *,
    symbols: tuple[str, ...],
) -> dict[str, dict[str, MetadataValue]]:
    if tuple(metadata) != symbols:
        raise ValueError("signed history metadata symbol order does not match scope")
    result: dict[str, dict[str, MetadataValue]] = {}
    for symbol in symbols:
        item = metadata[symbol]
        listed_at = _parse_utc(item.get("listed_at"), field=f"{symbol}.listed_at")
        result[symbol] = {
            "listed_at": listed_at.isoformat(),
            "tick_size": _strictly_positive(
                item.get("tick_size"), field=f"{symbol}.tick_size"
            ),
            "lot_size": _strictly_positive(
                item.get("lot_size"), field=f"{symbol}.lot_size"
            ),
            "minimum_notional": _strictly_positive(
                item.get("minimum_notional"), field=f"{symbol}.minimum_notional"
            ),
        }
    return result


def _signed_histories(
    histories: Mapping[str, Sequence[InstrumentExecutionRule]],
    *,
    symbols: tuple[str, ...],
    start_time: datetime,
    end_time: datetime,
) -> dict[str, tuple[InstrumentExecutionRule, ...]]:
    if tuple(histories) != symbols:
        raise ValueError("signed history execution-rule symbol order does not match scope")
    result: dict[str, tuple[InstrumentExecutionRule, ...]] = {}
    for symbol in symbols:
        rules = tuple(histories[symbol])
        if not rules:
            raise ValueError(f"historical execution-rule history is empty for {symbol}")
        effective = tuple(_utc(rule.effective_at, field="effective_at") for rule in rules)
        if effective != tuple(sorted(effective)) or len(set(effective)) != len(effective):
            raise ValueError(f"historical execution-rule history is not ordered for {symbol}")
        if effective[0] > start_time:
            raise ValueError(
                f"historical execution-rule history does not cover start for {symbol}"
            )
        if effective[-1] > end_time:
            raise ValueError(
                f"historical execution-rule history exceeds coverage for {symbol}"
            )
        for index, rule in enumerate(rules):
            _strictly_positive(
                rule.tick_size, field=f"{symbol}.rules[{index}].tick_size"
            )
            _strictly_positive(
                rule.lot_size, field=f"{symbol}.rules[{index}].lot_size"
            )
            _strictly_positive(
                rule.minimum_notional,
                field=f"{symbol}.rules[{index}].minimum_notional",
            )
        result[symbol] = rules
    return result


def resolution_from_historical_signed(
    *,
    metadata: Mapping[str, Mapping[str, MetadataValue]],
    execution_rule_histories: Mapping[str, Sequence[InstrumentExecutionRule]],
    signed_scope: BinanceHistoricalSignedScope,
    start_time: datetime,
    end_time: datetime,
) -> BinanceMetadataResolution:
    """Resolve an authenticated history only for its exact signed semantic scope."""

    requested_start = _utc(start_time, field="start_time")
    requested_end = _utc(end_time, field="end_time")
    if signed_scope.coverage_start != requested_start or signed_scope.coverage_end != requested_end:
        raise ValueError("signed history coverage does not match the research interval")
    symbols = signed_scope.symbols
    resolved_metadata = _signed_metadata(metadata, symbols=symbols)
    histories = _signed_histories(
        execution_rule_histories,
        symbols=symbols,
        start_time=requested_start,
        end_time=requested_end,
    )
    identity = _identity(
        mode=BinanceMetadataMode.HISTORICAL_SIGNED,
        market=signed_scope.market,
        symbols=symbols,
        source_uri=signed_scope.source_uri,
        as_of=_iso(signed_scope.issued_at, field="issued_at"),
        start_time=requested_start,
        end_time=requested_end,
        authentication="hmac-sha256",
        point_in_time=True,
        limitations=(),
        source_payload_digest=signed_scope.payload_digest,
    )
    return BinanceMetadataResolution(
        mode=BinanceMetadataMode.HISTORICAL_SIGNED,
        metadata=_freeze_metadata(resolved_metadata),
        execution_rule_histories=_freeze_histories(histories),
        identity_evidence=identity,
        evidence_digest=content_digest(identity),
        source_uri=signed_scope.source_uri,
    )


def _static_metadata(
    raw: object,
    *,
    symbols: tuple[str, ...],
) -> dict[str, dict[str, MetadataValue]]:
    if not isinstance(raw, Mapping):
        raise ValueError("conservative static symbols must be an object")
    unknown = set(str(symbol) for symbol in raw) - set(symbols)
    missing = set(symbols) - set(str(symbol) for symbol in raw)
    if unknown:
        raise ValueError(
            f"conservative static metadata has unknown symbols: {sorted(unknown)}"
        )
    if missing:
        raise ValueError(
            f"conservative static metadata is missing symbols: {sorted(missing)}"
        )
    result: dict[str, dict[str, MetadataValue]] = {}
    for symbol in symbols:
        item = raw.get(symbol)
        if not isinstance(item, Mapping):
            raise ValueError(
                f"conservative static metadata for {symbol} must be an object"
            )
        listed = _parse_utc(item.get("listed_at"), field=f"symbols.{symbol}.listed_at")
        result[symbol] = {
            "listed_at": listed.isoformat(),
            "tick_size": _positive(
                item.get("tick_size"), field=f"symbols.{symbol}.tick_size"
            ),
            "lot_size": _positive(
                item.get("lot_size"), field=f"symbols.{symbol}.lot_size"
            ),
            "minimum_notional": _positive(
                item.get("minimum_notional"),
                field=f"symbols.{symbol}.minimum_notional",
            ),
        }
    return result


def resolve_conservative_static(
    *,
    path: Path,
    symbols: Sequence[str],
    start_time: datetime,
    end_time: datetime,
) -> BinanceMetadataResolution:
    raw_payload = path.read_bytes()
    try:
        payload = json.loads(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("conservative static payload must be valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("conservative static payload must be an object")
    if payload.get("schema_version") != _CONSERVATIVE_STATIC_SCHEMA:
        raise ValueError("unsupported conservative static schema_version")
    resolved_symbols = _validate_symbols(symbols)
    as_of = _parse_utc(payload.get("as_of"), field="as_of")
    source_uri = str(payload.get("source_uri", f"file://{path.resolve().as_posix()}"))
    metadata = _static_metadata(payload.get("symbols"), symbols=resolved_symbols)
    raw_factors = payload.get("stress_factors")
    if not isinstance(raw_factors, Mapping):
        raise ValueError("stress_factors must be an object")
    required = ("tick_size", "lot_size", "minimum_notional")
    if set(str(key) for key in raw_factors) != set(required):
        raise ValueError(f"stress_factors must contain exactly {list(required)}")
    factors: dict[str, float] = {}
    for name in required:
        value = _positive(raw_factors.get(name), field=f"stress_factors.{name}")
        if value < 1.0:
            raise ValueError(f"stress_factors.{name} must be at least 1.0")
        factors[name] = value
    raw_digest = sha256(raw_payload).hexdigest()
    identity = _identity(
        mode=BinanceMetadataMode.CONSERVATIVE_STATIC,
        market=BinanceMarket.USDS_M.value,
        symbols=resolved_symbols,
        source_uri=source_uri,
        as_of=_iso(as_of, field="as_of"),
        start_time=start_time,
        end_time=end_time,
        authentication="operator-declared",
        point_in_time=False,
        limitations=(_LIMITATION,),
        source_payload_digest=raw_digest,
        raw_payload_sha256=raw_digest,
        stress_factors=factors,
    )
    return BinanceMetadataResolution(
        mode=BinanceMetadataMode.CONSERVATIVE_STATIC,
        metadata=_freeze_metadata(metadata),
        execution_rule_histories=None,
        identity_evidence=identity,
        evidence_digest=content_digest(identity),
        source_uri=source_uri,
        raw_payload=raw_payload,
    )


__all__ = [
    "BinanceHistoricalSignedScope",
    "BinanceMetadataMode",
    "BinanceMetadataResolution",
    "BinanceMetadataResolutionProvider",
    "resolve_conservative_static",
    "resolve_frozen_snapshot",
    "resolution_from_historical_signed",
]
