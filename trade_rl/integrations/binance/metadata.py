"""Binance exchange metadata contracts and verified frozen snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from trade_rl.data.contracts import (
    InstrumentContract,
    InstrumentExecutionRule,
    VolumeUnit,
)
from trade_rl.integrations.binance.types import (
    BinanceMarket,
    BinanceTransportMode,
    _finite_float,
)
from trade_rl.integrations.binance.vision import _normalize_epoch_ms


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_json_object(payload: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {key: _freeze_json(value) for key, value in payload.items()}
    )


def _mutable_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _mutable_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_json(item) for item in value]
    return value


def _mutable_json_object(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: _mutable_json(value) for key, value in payload.items()}


@dataclass(frozen=True, slots=True)
class BinanceInstrumentMetadata:
    symbol: str
    listed_at: datetime
    tick_size: float
    lot_size: float
    minimum_notional: float
    volume_unit: VolumeUnit
    contract_multiplier: float = 1.0
    execution_rules: tuple[InstrumentExecutionRule, ...] = ()

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("Binance symbol must not be empty")
        if self.listed_at.tzinfo is None or self.listed_at.utcoffset() is None:
            raise ValueError("listed_at must be timezone-aware")
        for field_name, value in (
            ("tick_size", self.tick_size),
            ("lot_size", self.lot_size),
            ("minimum_notional", self.minimum_notional),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if (
            not math.isfinite(self.contract_multiplier)
            or self.contract_multiplier <= 0.0
        ):
            raise ValueError("contract_multiplier must be finite and positive")

    def to_contract(self) -> InstrumentContract:
        return InstrumentContract(
            symbol=self.symbol,
            listed_at=self.listed_at.astimezone(UTC),
            volume_unit=self.volume_unit,
            contract_multiplier=self.contract_multiplier,
            tick_size=self.tick_size,
            lot_size=self.lot_size,
            minimum_notional=self.minimum_notional,
            execution_rules=self.execution_rules,
        )


@dataclass(frozen=True, slots=True)
class BinanceExchangeInfoSnapshot:
    payload: Mapping[str, object]
    raw_payload: bytes
    source_uri: str
    retrieved_at: datetime
    raw_payload_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_json_object(self.payload))


def _filter_value(
    values: Sequence[Mapping[str, object]],
    *,
    filter_type: str,
    fields: Sequence[str],
) -> float:
    for item in values:
        if item.get("filterType") != filter_type:
            continue
        for field in fields:
            raw = item.get(field)
            if raw is not None:
                return _finite_float(raw, field=f"{filter_type}.{field}")
    return 0.0


def _metadata_from_exchange_info(
    payload: Mapping[str, object],
    *,
    market: BinanceMarket,
    symbols: tuple[str, ...],
) -> tuple[BinanceInstrumentMetadata, ...]:
    raw_symbols = payload.get("symbols")
    if not isinstance(raw_symbols, list):
        raise ValueError("Binance exchange information lacks symbols")
    by_symbol: dict[str, Mapping[str, object]] = {}
    for item in raw_symbols:
        if isinstance(item, dict) and isinstance(item.get("symbol"), str):
            by_symbol[str(item["symbol"])] = item
    result: list[BinanceInstrumentMetadata] = []
    for symbol in symbols:
        item = by_symbol.get(symbol)
        if item is None:
            raise ValueError(f"Binance exchange information has no symbol {symbol}")
        status = item.get("status", item.get("contractStatus"))
        if status != "TRADING":
            raise ValueError(f"Binance symbol {symbol} is not trading: {status}")
        raw_filters = item.get("filters")
        if not isinstance(raw_filters, list):
            raise ValueError(f"Binance symbol {symbol} lacks filters")
        filters = tuple(value for value in raw_filters if isinstance(value, dict))
        listed_raw = item.get("onboardDate", 0)
        listed_ms = _normalize_epoch_ms(listed_raw)
        listed_at = datetime.fromtimestamp(listed_ms / 1_000, tz=UTC)
        result.append(
            BinanceInstrumentMetadata(
                symbol=symbol,
                listed_at=listed_at,
                tick_size=_filter_value(
                    filters,
                    filter_type="PRICE_FILTER",
                    fields=("tickSize",),
                ),
                lot_size=_filter_value(
                    filters,
                    filter_type="LOT_SIZE",
                    fields=("stepSize",),
                ),
                minimum_notional=max(
                    _filter_value(
                        filters,
                        filter_type="MIN_NOTIONAL",
                        fields=("notional", "minNotional"),
                    ),
                    _filter_value(
                        filters,
                        filter_type="NOTIONAL",
                        fields=("minNotional",),
                    ),
                ),
                volume_unit=VolumeUnit.QUOTE_NOTIONAL,
                contract_multiplier=1.0,
            )
        )
    return tuple(result)


class _ExchangeInfoTransport(Protocol):
    def load_exchange_information_snapshot(
        self,
        *,
        market: BinanceMarket | str,
        mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
    ) -> BinanceExchangeInfoSnapshot: ...


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_non_empty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _parse_utc(value: object) -> datetime:
    text = _require_non_empty(value, field="retrieved_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("retrieved_at must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("retrieved_at must include a timezone")
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class FrozenBinanceExchangeInfoTransport:
    """Load exact cache bytes, optionally freezing one explicit live delegate."""

    root: Path
    delegate: _ExchangeInfoTransport | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def _load(self, *, market: BinanceMarket) -> BinanceExchangeInfoSnapshot | None:
        raw_path = self.root / "exchange-info.raw.json"
        manifest_path = self.root / "manifest.json"
        if raw_path.is_file() != manifest_path.is_file():
            raise RuntimeError("frozen metadata cache is incomplete")
        if not raw_path.is_file():
            return None
        raw = raw_path.read_bytes()
        try:
            manifest = _require_mapping(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                field="frozen metadata cache manifest",
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("frozen metadata cache manifest is invalid") from error
        digest = hashlib.sha256(raw).hexdigest()
        if manifest.get("schema_version") != "frozen_metadata_cache_v1":
            raise ValueError("frozen metadata cache schema mismatch")
        if manifest.get("market") != market.value:
            raise ValueError("frozen metadata cache market mismatch")
        if manifest.get("raw_payload_sha256") != digest:
            raise ValueError("frozen metadata cache digest mismatch")
        try:
            payload = _require_mapping(
                json.loads(raw), field="cached exchange information"
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("cached exchange information is invalid") from error
        return BinanceExchangeInfoSnapshot(
            payload=payload,
            raw_payload=raw,
            source_uri=_require_non_empty(
                manifest.get("source_uri"), field="source_uri"
            ),
            retrieved_at=_parse_utc(manifest.get("retrieved_at")),
            raw_payload_sha256=digest,
        )

    def _freeze(
        self,
        snapshot: BinanceExchangeInfoSnapshot,
        *,
        market: BinanceMarket,
    ) -> None:
        digest = hashlib.sha256(snapshot.raw_payload).hexdigest()
        if digest != snapshot.raw_payload_sha256:
            raise ValueError("live exchange information digest mismatch")
        source_uri = _require_non_empty(snapshot.source_uri, field="source_uri")
        retrieved_at = snapshot.retrieved_at
        if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        self.root.mkdir(parents=True, exist_ok=True)
        raw_path = self.root / "exchange-info.raw.json"
        manifest_path = self.root / "manifest.json"
        manifest = {
            "market": market.value,
            "raw_payload_sha256": digest,
            "retrieved_at": retrieved_at.astimezone(UTC).isoformat(),
            "schema_version": "frozen_metadata_cache_v1",
            "source_uri": source_uri,
        }
        raw_temporary = raw_path.with_suffix(f".raw.{os.getpid()}.tmp")
        manifest_temporary = manifest_path.with_suffix(f".{os.getpid()}.tmp")
        raw_temporary.write_bytes(snapshot.raw_payload)
        manifest_temporary.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        raw_temporary.replace(raw_path)
        manifest_temporary.replace(manifest_path)

    def load_exchange_information_snapshot(
        self,
        *,
        market: BinanceMarket | str,
        mode: BinanceTransportMode | str = BinanceTransportMode.REST,
    ) -> BinanceExchangeInfoSnapshot:
        resolved_market = BinanceMarket(market)
        cached = self._load(market=resolved_market)
        if cached is not None:
            return cached
        if self.delegate is None:
            raise RuntimeError("frozen metadata cache is incomplete")
        snapshot = self.delegate.load_exchange_information_snapshot(
            market=resolved_market,
            mode=mode,
        )
        self._freeze(snapshot, market=resolved_market)
        verified = self._load(market=resolved_market)
        if verified is None:  # pragma: no cover - atomic write invariant
            raise RuntimeError("frozen metadata cache publication failed")
        return verified


__all__ = [
    "BinanceExchangeInfoSnapshot",
    "BinanceInstrumentMetadata",
    "FrozenBinanceExchangeInfoTransport",
]
