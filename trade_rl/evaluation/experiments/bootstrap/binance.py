"""Freeze and inspect exact Binance source evidence for canonical M2 bootstrap."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.config import CanonicalM2BootstrapConfig
from trade_rl.integrations.binance import (
    BinanceExchangeInfoSnapshot,
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportMode,
    FrozenBinanceExchangeInfoTransport,
    plan_binance_vision_cache,
    require_complete_binance_vision_cache,
    sync_binance_vision_cache,
    validate_cached_vision_payload,
    vision_cache_path,
)

_PLAN_SCHEMA = "canonical_m2_vision_plan_v1"
_METADATA_EVIDENCE_SCHEMA = "frozen_binance_exchange_info_evidence_v1"


class _LiveSourceTransport(Protocol):
    cache_root: Path | None

    def _request_bytes(self, url: str) -> bytes: ...

    def load_exchange_information_snapshot(
        self,
        *,
        market: BinanceMarket | str,
        mode: BinanceTransportMode | str = BinanceTransportMode.REST,
    ) -> BinanceExchangeInfoSnapshot: ...


@dataclass(frozen=True, slots=True)
class _FrozenDatasetTransport:
    market_data: BinancePublicTransport
    metadata: FrozenBinanceExchangeInfoTransport

    def load_klines(
        self,
        *,
        market: BinanceMarket | str,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        mode: BinanceTransportMode | str = BinanceTransportMode.VISION,
    ) -> tuple[list[list[object]], str]:
        return self.market_data.load_klines(
            market=market,
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            mode=mode,
        )

    def load_funding_rates(
        self,
        *,
        market: BinanceMarket | str,
        symbol: str,
        start_ms: int,
        end_ms: int,
        mode: BinanceTransportMode | str = BinanceTransportMode.VISION,
    ) -> tuple[list[tuple[int, float]], str]:
        return self.market_data.load_funding_rates(
            market=market,
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            mode=mode,
        )

    def load_exchange_information(
        self,
        *,
        market: BinanceMarket | str,
        mode: BinanceTransportMode | str = BinanceTransportMode.REST,
    ) -> tuple[dict[str, object], str]:
        return self.metadata.load_exchange_information(market=market, mode=mode)


@dataclass(frozen=True, slots=True)
class FrozenBinanceSource:
    """Verified, network-cut Binance source composition for one bootstrap."""

    cache_transport: BinancePublicTransport
    metadata_transport: FrozenBinanceExchangeInfoTransport
    composite_transport: _FrozenDatasetTransport
    vision_plan_payload: dict[str, object]
    vision_plan_digest: str
    raw_source_roster: tuple[dict[str, object], ...]
    raw_source_roster_digest: str
    metadata_evidence: dict[str, object]


def _vision_plan_payload(config: CanonicalM2BootstrapConfig) -> dict[str, object]:
    plan = plan_binance_vision_cache(
        market=config.market,
        symbols=config.symbols,
        intervals=(config.base_timeframe, *config.feature_timeframes),
        start_time=config.data_start,
        end_time=config.data_stop_exclusive,
    )
    return {
        "schema_version": _PLAN_SCHEMA,
        "market": plan.market.value,
        "symbols": list(plan.symbols),
        "intervals": list(plan.intervals),
        "start_time": plan.start_time.astimezone(UTC).isoformat(),
        "end_time": plan.end_time.astimezone(UTC).isoformat(),
        "urls": list(plan.urls),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _read_plan(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Vision plan must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Vision plan is invalid JSON") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError("Vision plan must be a JSON object")
    return payload


def _raw_source_roster(
    urls: tuple[str, ...],
    *,
    cache_root: Path,
) -> tuple[dict[str, object], ...]:
    roster: list[dict[str, object]] = []
    for url in urls:
        path = vision_cache_path(cache_root, url)
        if path.is_symlink():
            raise ValueError("Binance Vision cache member must be a regular file")
        payload = validate_cached_vision_payload(url, path)
        roster.append(
            {
                "url": url,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return tuple(roster)


def _metadata_evidence(
    snapshot: BinanceExchangeInfoSnapshot,
) -> dict[str, object]:
    return {
        "schema_version": _METADATA_EVIDENCE_SCHEMA,
        "market": BinanceMarket.USDS_M.value,
        "source_uri": snapshot.source_uri,
        "raw_payload_sha256": snapshot.raw_payload_sha256,
        "retrieved_at": snapshot.retrieved_at.astimezone(UTC).isoformat(),
    }


def _inspect_frozen_binance_source(
    config: CanonicalM2BootstrapConfig,
    source_root: str | Path,
) -> FrozenBinanceSource:
    """Reconstruct one frozen Binance source without network access."""

    root = Path(source_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("frozen Binance source root must be a regular directory")

    expected_plan_payload = _vision_plan_payload(config)
    observed_plan_payload = _read_plan(root / "vision-plan.json")
    if observed_plan_payload != expected_plan_payload:
        raise ValueError("Vision plan differs from pre-registered source contract")

    plan = plan_binance_vision_cache(
        market=config.market,
        symbols=config.symbols,
        intervals=(config.base_timeframe, *config.feature_timeframes),
        start_time=config.data_start,
        end_time=config.data_stop_exclusive,
    )
    cache_root = root / "vision-cache"
    require_complete_binance_vision_cache(plan, cache_root=cache_root)
    roster = _raw_source_roster(plan.urls, cache_root=cache_root)

    metadata_root = root / "exchange-info"
    if metadata_root.is_symlink():
        raise ValueError("frozen metadata root must be a regular directory")
    metadata_transport = FrozenBinanceExchangeInfoTransport(metadata_root)
    snapshot = metadata_transport.load_exchange_information_snapshot(
        market=config.market,
        mode=BinanceTransportMode.REST,
    )
    cache_transport = BinancePublicTransport(cache_root=cache_root, allow_network=False)
    composite = _FrozenDatasetTransport(
        market_data=cache_transport,
        metadata=metadata_transport,
    )
    return FrozenBinanceSource(
        cache_transport=cache_transport,
        metadata_transport=metadata_transport,
        composite_transport=composite,
        vision_plan_payload=observed_plan_payload,
        vision_plan_digest=content_digest(observed_plan_payload),
        raw_source_roster=roster,
        raw_source_roster_digest=content_digest(list(roster)),
        metadata_evidence=_metadata_evidence(snapshot),
    )


def _freeze_binance_source(
    config: CanonicalM2BootstrapConfig,
    source_root: str | Path,
    *,
    live_transport: _LiveSourceTransport | None = None,
) -> FrozenBinanceSource:
    """Acquire exact Binance bytes once, then return only cache-only readers."""

    root = Path(source_root)
    if root.exists():
        raise FileExistsError(f"frozen Binance source root already exists: {root}")
    root.mkdir(parents=True)
    cache_root = root / "vision-cache"
    metadata_root = root / "exchange-info"

    live: _LiveSourceTransport
    if live_transport is None:
        live = BinancePublicTransport(cache_root=cache_root)
    else:
        live = live_transport
        if live.cache_root is None or Path(live.cache_root) != cache_root:
            raise ValueError("live transport cache_root must equal bootstrap Vision cache root")

    frozen_metadata = FrozenBinanceExchangeInfoTransport(
        metadata_root,
        delegate=live,
    )
    frozen_metadata.load_exchange_information_snapshot(
        market=config.market,
        mode=BinanceTransportMode.REST,
    )

    plan = plan_binance_vision_cache(
        market=config.market,
        symbols=config.symbols,
        intervals=(config.base_timeframe, *config.feature_timeframes),
        start_time=config.data_start,
        end_time=config.data_stop_exclusive,
    )
    _write_json(root / "vision-plan.json", _vision_plan_payload(config))
    sync_binance_vision_cache(plan, transport=live)
    require_complete_binance_vision_cache(plan, cache_root=cache_root)

    return _inspect_frozen_binance_source(config, root)


__all__ = ["FrozenBinanceSource"]
