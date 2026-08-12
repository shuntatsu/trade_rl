"""Verified offline transport for frozen Binance exchange-information evidence."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from trade_rl.integrations.binance import (
    BinanceExchangeInfoSnapshot,
    BinanceMarket,
    BinanceTransportMode,
)


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


__all__ = ["FrozenBinanceExchangeInfoTransport"]
