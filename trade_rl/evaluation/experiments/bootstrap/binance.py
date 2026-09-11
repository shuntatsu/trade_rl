"""Freeze and inspect exact Binance source evidence for canonical M2 bootstrap."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.config import CanonicalM2BootstrapConfig
from trade_rl.integrations.binance import (
    BinanceExchangeInfoSnapshot,
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportMode,
    BinanceVisionCachePlan,
    FrozenBinanceExchangeInfoTransport,
    binance_interval_milliseconds,
    plan_binance_vision_cache,
    require_complete_binance_vision_cache,
    sync_binance_vision_cache,
    sync_binance_vision_urls,
    validate_cached_vision_payload,
    vision_cache_path,
    vision_kline_url,
)
from trade_rl.integrations.binance.vision import _normalize_epoch_ms

_PLAN_SCHEMA = "canonical_m2_vision_plan_v1"
_RESOLUTION_SCHEMA = "canonical_m2_vision_resolution_v1"
_METADATA_EVIDENCE_SCHEMA = "frozen_binance_exchange_info_evidence_v1"
_DAY = timedelta(days=1)


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
class _VisionRepair:
    symbol: str
    timeframe: str
    missing_open_ms: tuple[int, ...]
    daily_urls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FrozenDatasetTransport:
    market_data: BinancePublicTransport
    metadata: FrozenBinanceExchangeInfoTransport
    repairs: tuple[_VisionRepair, ...] = ()

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
        primary, source = self.market_data.load_klines(
            market=market,
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            mode=mode,
        )
        if BinanceTransportMode(mode) is not BinanceTransportMode.VISION:
            return primary, source
        repair = next(
            (
                item
                for item in self.repairs
                if item.symbol == symbol and item.timeframe == interval
            ),
            None,
        )
        if repair is None:
            return primary, source
        relevant_missing = tuple(
            value for value in repair.missing_open_ms if start_ms <= value < end_ms
        )
        if not relevant_missing:
            return primary, source

        step = binance_interval_milliseconds(interval)
        rows_by_open = _index_kline_rows(
            primary,
            start_ms=start_ms,
            end_ms=end_ms,
            interval_ms=step,
            source="primary Vision evidence",
        )
        days = tuple(dict.fromkeys(_utc_day(value) for value in relevant_missing))
        for day in days:
            day_start_ms = int(day.timestamp() * 1_000)
            daily_rows, daily_source = self.market_data.load_klines(
                market=market,
                symbol=symbol,
                interval=interval,
                start_ms=day_start_ms,
                end_ms=day_start_ms + 24 * 60 * 60 * 1_000,
                mode=BinanceTransportMode.VISION,
            )
            if daily_source != "vision":
                raise ValueError("daily Vision repair must use Vision evidence")
            daily_by_open = _index_kline_rows(
                daily_rows,
                start_ms=max(start_ms, day_start_ms),
                end_ms=min(end_ms, day_start_ms + 24 * 60 * 60 * 1_000),
                interval_ms=step,
                source="daily Vision repair",
            )
            for open_ms, row in daily_by_open.items():
                existing = rows_by_open.get(open_ms)
                if existing is not None:
                    if _kline_semantics(existing) != _kline_semantics(row):
                        raise ValueError(
                            "daily Vision repair overlap conflicts with primary evidence"
                        )
                    continue
                rows_by_open[open_ms] = row

        expected = _expected_open_times(start_ms, end_ms, step)
        observed = tuple(sorted(rows_by_open))
        if observed != expected:
            missing = tuple(value for value in expected if value not in rows_by_open)
            raise ValueError(
                "daily Vision repair did not produce a complete regular clock; "
                f"missing={missing[:20]}"
            )
        return [rows_by_open[value] for value in expected], "vision"

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
    vision_resolution_payload: dict[str, object] | None
    vision_resolution_digest: str | None
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


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) for key in payload
    ):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _read_plan(path: Path) -> dict[str, object]:
    return _read_json_object(path, label="Vision plan")


def _utc_day(open_ms: int) -> datetime:
    return datetime.fromtimestamp(open_ms / 1_000, tz=UTC).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )


def _expected_open_times(
    start_ms: int, end_ms: int, interval_ms: int
) -> tuple[int, ...]:
    if end_ms <= start_ms or (end_ms - start_ms) % interval_ms != 0:
        raise ValueError("Vision kline range must align to its native interval")
    return tuple(range(start_ms, end_ms, interval_ms))


def _index_kline_rows(
    rows: list[list[object]],
    *,
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    source: str,
) -> dict[int, list[object]]:
    indexed: dict[int, list[object]] = {}
    previous: int | None = None
    for raw_row in rows:
        if len(raw_row) < 8:
            raise ValueError(f"{source} kline row must contain at least eight fields")
        row = list(raw_row)
        open_ms = _normalize_epoch_ms(row[0])
        if not start_ms <= open_ms < end_ms:
            continue
        if (open_ms - start_ms) % interval_ms != 0:
            raise ValueError(f"{source} kline timestamp is off the native clock")
        if previous is not None and open_ms <= previous:
            raise ValueError(f"{source} kline timestamps must be strictly increasing")
        previous = open_ms
        if open_ms in indexed:
            raise ValueError(f"{source} kline timestamps must be unique")
        indexed[open_ms] = row
    return indexed


def _finite_number(value: object, *, field: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} in Vision kline evidence") from error
    if not math.isfinite(number):
        raise ValueError(f"non-finite {field} in Vision kline evidence")
    return number


def _kline_semantics(
    row: list[object],
) -> tuple[int, float, float, float, float, float]:
    return (
        _normalize_epoch_ms(row[0]),
        _finite_number(row[1], field="open"),
        _finite_number(row[2], field="high"),
        _finite_number(row[3], field="low"),
        _finite_number(row[4], field="close"),
        _finite_number(row[7], field="quote volume"),
    )


def _resolve_vision_repairs(
    config: CanonicalM2BootstrapConfig,
    *,
    primary_plan: BinanceVisionCachePlan,
    cache_root: Path,
) -> tuple[_VisionRepair, ...]:
    transport = BinancePublicTransport(cache_root=cache_root, allow_network=False)
    start_ms = int(config.data_start.timestamp() * 1_000)
    end_ms = int(config.data_stop_exclusive.timestamp() * 1_000)
    repairs: list[_VisionRepair] = []
    primary_urls = frozenset(primary_plan.urls)
    for symbol in config.symbols:
        for timeframe in (config.base_timeframe, *config.feature_timeframes):
            interval_ms = binance_interval_milliseconds(timeframe)
            rows, source = transport.load_klines(
                market=config.market,
                symbol=symbol,
                interval=timeframe,
                start_ms=start_ms,
                end_ms=end_ms,
                mode=BinanceTransportMode.VISION,
            )
            if source != "vision":
                raise ValueError("primary source resolution must use Vision evidence")
            indexed = _index_kline_rows(
                rows,
                start_ms=start_ms,
                end_ms=end_ms,
                interval_ms=interval_ms,
                source="primary Vision evidence",
            )
            expected = _expected_open_times(start_ms, end_ms, interval_ms)
            missing = tuple(value for value in expected if value not in indexed)
            if not missing:
                continue
            daily_urls = tuple(
                dict.fromkeys(
                    vision_kline_url(
                        config.market,
                        symbol,
                        timeframe,
                        _utc_day(value),
                    )
                    for value in missing
                )
            )
            if any(url in primary_urls for url in daily_urls):
                raise ValueError(
                    "incomplete primary daily Vision archive has no distinct repair source"
                )
            repairs.append(
                _VisionRepair(
                    symbol=symbol,
                    timeframe=timeframe,
                    missing_open_ms=missing,
                    daily_urls=daily_urls,
                )
            )
    return tuple(repairs)


def _vision_resolution_payload(
    config: CanonicalM2BootstrapConfig,
    repairs: tuple[_VisionRepair, ...],
) -> dict[str, object]:
    return {
        "schema_version": _RESOLUTION_SCHEMA,
        "primary_plan_digest": content_digest(_vision_plan_payload(config)),
        "repairs": [
            {
                "symbol": repair.symbol,
                "timeframe": repair.timeframe,
                "missing_open_ms": list(repair.missing_open_ms),
                "daily_urls": list(repair.daily_urls),
            }
            for repair in repairs
        ],
    }


def _repair_urls(repairs: tuple[_VisionRepair, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(url for repair in repairs for url in repair.daily_urls))


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
    *,
    require_resolution: bool = True,
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

    resolution_payload: dict[str, object] | None
    resolution_digest: str | None
    repairs: tuple[_VisionRepair, ...]
    resolution_path = root / "vision-resolution.json"
    if require_resolution:
        repairs = _resolve_vision_repairs(
            config,
            primary_plan=plan,
            cache_root=cache_root,
        )
        expected_resolution = _vision_resolution_payload(config, repairs)
        observed_resolution = _read_json_object(
            resolution_path,
            label="Vision resolution",
        )
        if observed_resolution != expected_resolution:
            raise ValueError("Vision resolution differs from cached primary evidence")
        resolution_payload = observed_resolution
        resolution_digest = content_digest(observed_resolution)
    else:
        if resolution_path.exists():
            raise ValueError(
                "legacy primary-only Vision source must not have resolution"
            )
        repairs = ()
        resolution_payload = None
        resolution_digest = None

    repair_urls = _repair_urls(repairs)
    roster_urls = tuple(dict.fromkeys((*plan.urls, *repair_urls)))
    roster = _raw_source_roster(roster_urls, cache_root=cache_root)

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
        repairs=repairs,
    )
    return FrozenBinanceSource(
        cache_transport=cache_transport,
        metadata_transport=metadata_transport,
        composite_transport=composite,
        vision_plan_payload=observed_plan_payload,
        vision_plan_digest=content_digest(observed_plan_payload),
        vision_resolution_payload=resolution_payload,
        vision_resolution_digest=resolution_digest,
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
            raise ValueError(
                "live transport cache_root must equal bootstrap Vision cache root"
            )

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

    repairs = _resolve_vision_repairs(
        config,
        primary_plan=plan,
        cache_root=cache_root,
    )
    repair_urls = _repair_urls(repairs)
    if repair_urls:
        sync_binance_vision_urls(repair_urls, transport=live)
    resolution_payload = _vision_resolution_payload(config, repairs)
    _write_json(root / "vision-resolution.json", resolution_payload)

    frozen = _inspect_frozen_binance_source(config, root)
    start_ms = int(config.data_start.timestamp() * 1_000)
    end_ms = int(config.data_stop_exclusive.timestamp() * 1_000)
    for symbol in config.symbols:
        for timeframe in (config.base_timeframe, *config.feature_timeframes):
            frozen.composite_transport.load_klines(
                market=config.market,
                symbol=symbol,
                interval=timeframe,
                start_ms=start_ms,
                end_ms=end_ms,
                mode=BinanceTransportMode.VISION,
            )
    return frozen


__all__ = ["FrozenBinanceSource"]
