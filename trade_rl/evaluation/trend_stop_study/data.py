"""Multi-symbol 4h / 1d bar Datasets for the trend study (offline, read-only).

Primary path: a published Dataset artifact written by the Market Store export CLI
(one export per timeframe; the command is documented in
``docs/specs/trend-following-stops.md``), loaded through the verified artifact reader.
Fallback: a
local Binance Vision cache of native 4h / 1d klines and funding archives, built with the
unchanged ``build_binance_market_dataset`` and never any network access.

Neither path reads a price into this module's logic: it only validates the roster, the
bar cadence and the hard data boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.data import MarketDataset
from trade_rl.data.artifacts.publication import load_published_market_dataset_artifact
from trade_rl.data.build.economics import ExecutionEconomicsProfile
from trade_rl.evaluation.trend_stop_study.protocol import TrendStopProtocol
from trade_rl.integrations.binance import (
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportMode,
    FrozenBinanceExchangeInfoTransport,
    plan_binance_vision_cache,
    require_complete_binance_vision_cache,
)
from trade_rl.integrations.binance.dataset import (
    BinanceGapPolicy,
    build_binance_market_dataset,
)
from trade_rl.integrations.binance.derived_klines import (
    BaseIntervalAggregatingTransport,
)

_BAR_NS = {"4h": 4 * 3_600 * 10**9, "1d": 24 * 3_600 * 10**9}
ECONOMICS_PROFILE_NAME = "bot_overlay_only_assumptions_v1"


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _ns(text: str) -> int:
    return int(np.datetime64(text, "ns").astype(np.int64))


class RosterTooSmallError(ValueError):
    """The recorded exclusions leave fewer than ``min_symbols``: the run is INVALID."""


@dataclass(frozen=True)
class SymbolExclusion:
    """Evidence that one roster symbol has no archive (never a transient failure)."""

    symbol: str
    url: str
    http_status: int

    def to_mapping(self) -> dict[str, object]:
        return {"symbol": self.symbol, "url": self.url, "http_status": self.http_status}


_ARCHIVE_PREFIX = "https://data.binance.vision/"


def parse_exclusions(
    raw: object, protocol: TrendStopProtocol
) -> tuple[SymbolExclusion, ...]:
    """Strictly parse missing-archive exclusion records.

    Only an HTTP 404 for a Binance Vision archive URL of that symbol is evidence of a
    missing archive. Transient fetch failures, cache misses and malformed or checksum
    errors are not representable here: they must halt the run, never exclude a symbol.
    """

    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("exclusions must be a list of records")
    parsed: list[SymbolExclusion] = []
    for record in raw:
        if not isinstance(record, Mapping) or set(record) != {
            "symbol",
            "url",
            "http_status",
        }:
            raise ValueError("an exclusion needs exactly symbol, url and http_status")
        symbol, url, status = record["symbol"], record["url"], record["http_status"]
        if symbol not in protocol.symbols:
            raise ValueError(f"excluded symbol is not in the roster: {symbol!r}")
        if symbol == protocol.benchmark_symbol:
            raise ValueError("the benchmark symbol cannot be excluded")
        if (
            not isinstance(url, str)
            or not url.startswith(_ARCHIVE_PREFIX)
            or f"/{symbol}/" not in url
        ):
            raise ValueError(
                f"exclusion evidence for {symbol} must be its Binance Vision archive URL"
            )
        if (
            isinstance(status, bool)
            or not isinstance(status, int)
            or status != protocol.exclusion_http_status
        ):
            raise ValueError(
                f"exclusion for {symbol} needs HTTP {protocol.exclusion_http_status}; "
                "any other failure must halt the run"
            )
        parsed.append(SymbolExclusion(symbol, url, status))
    if len({item.symbol for item in parsed}) != len(parsed):
        raise ValueError("duplicate exclusion records")
    return tuple(parsed)


def active_symbols(
    protocol: TrendStopProtocol, exclusions: Sequence[SymbolExclusion] = ()
) -> tuple[str, ...]:
    excluded = {item.symbol for item in exclusions}
    return tuple(symbol for symbol in protocol.symbols if symbol not in excluded)


def load_exported_dataset(path: str | Path) -> MarketDataset:
    """Load and verify a published Dataset artifact (``<export-dir>/dataset``)."""

    return load_published_market_dataset_artifact(Path(path))[1]


def validate_study_dataset(
    dataset: MarketDataset,
    protocol: TrendStopProtocol,
    timeframe: str,
    exclusions: Sequence[SymbolExclusion] = (),
) -> tuple[str, ...]:
    """Check roster, cadence, warm-up and the data boundary; return the symbols used.

    The roster must be complete: the Dataset symbols plus the recorded exclusions must
    equal the preregistered roster exactly. A symbol absent without a recorded
    missing-archive exclusion is refused (``ValueError``); fewer than ``min_symbols``
    remaining raises :class:`RosterTooSmallError` so the caller can write an INVALID
    result.

    The boundary check is on the loaded data, not only on the requested window: a
    Dataset holding any bar that closes after ``data_end_exclusive`` is refused even if
    the evaluation window would not reach it.
    """

    if timeframe not in _BAR_NS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    stamps = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    if stamps.size < 2:
        raise ValueError("Dataset needs at least two bars")
    if int(stamps[-1]) > _ns(protocol.data_end_exclusive):
        raise ValueError(
            "Dataset holds bars after the exploratory data boundary "
            f"{protocol.data_end_exclusive}; refusing to load it"
        )
    steps = np.unique(np.diff(stamps))
    if steps.size != 1 or int(steps[0]) != _BAR_NS[timeframe]:
        raise ValueError(f"Dataset cadence is not a regular {timeframe} grid")
    if int(stamps[0]) > _ns(protocol.evaluation_start):
        raise ValueError("Dataset starts after the evaluation start (no warm-up)")
    unknown = sorted(set(dataset.symbols) - set(protocol.symbols))
    if unknown:
        raise ValueError(f"Dataset holds symbols outside the roster: {unknown}")
    excluded = {item.symbol for item in exclusions}
    present_excluded = sorted(excluded & set(dataset.symbols))
    if present_excluded:
        raise ValueError(
            f"symbols recorded as excluded are present in the Dataset: {present_excluded}"
        )
    unrecorded = sorted(set(protocol.symbols) - excluded - set(dataset.symbols))
    if unrecorded:
        raise ValueError(
            "roster symbols absent without a recorded missing-archive exclusion "
            f"(URL and HTTP {protocol.exclusion_http_status}): {unrecorded}"
        )
    if protocol.benchmark_symbol not in dataset.symbols:
        raise ValueError("the benchmark symbol must be part of the Dataset")
    if len(dataset.symbols) < protocol.min_symbols:
        raise RosterTooSmallError(
            f"Dataset has {len(dataset.symbols)} symbols; at least "
            f"{protocol.min_symbols} are required"
        )
    return tuple(dataset.symbols)


def plan_vision_cache(
    protocol: TrendStopProtocol,
    timeframe: str,
    exclusions: Sequence[SymbolExclusion] = (),
) -> Any:
    """The Vision archives a fallback build needs (pure; touches no network)."""

    return plan_binance_vision_cache(
        market=BinanceMarket.USDS_M,
        symbols=active_symbols(protocol, exclusions),
        intervals=(timeframe,),
        start_time=_utc(protocol.data_start),
        end_time=_utc(protocol.data_end_exclusive),
    )


def build_dataset_from_transport(
    protocol: TrendStopProtocol,
    timeframe: str,
    transport: Any,
    *,
    evidence: Mapping[str, object],
    exclusions: Sequence[SymbolExclusion] = (),
) -> MarketDataset:
    """Build with the unchanged Binance builder from any Vision-shaped transport."""

    if timeframe not in _BAR_NS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    built = build_binance_market_dataset(
        market=BinanceMarket.USDS_M,
        symbols=active_symbols(protocol, exclusions),
        interval=timeframe,
        start_time=_utc(protocol.data_start),
        end_time=_utc(protocol.data_end_exclusive),
        transport_mode=BinanceTransportMode.VISION,
        transport=transport,
        metadata_evidence=dict(evidence),
        execution_economics=ExecutionEconomicsProfile(name=ECONOMICS_PROFILE_NAME),
        gap_policy=BinanceGapPolicy.NON_TRADABLE_ROWS,
    )
    validate_study_dataset(built.dataset, protocol, timeframe, exclusions)
    return built.dataset


def build_vision_dataset(
    protocol: TrendStopProtocol,
    timeframe: str,
    *,
    cache_root: Path,
    exchange_info_dir: Path,
    exclusions: Sequence[SymbolExclusion] = (),
) -> MarketDataset:
    """Fallback: build from a complete local Vision cache; never uses the network."""

    if timeframe not in _BAR_NS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    plan = plan_vision_cache(protocol, timeframe, exclusions)
    require_complete_binance_vision_cache(plan, cache_root=cache_root)
    frozen = FrozenBinanceExchangeInfoTransport(exchange_info_dir, delegate=None)
    transport = BaseIntervalAggregatingTransport(
        BinancePublicTransport(cache_root=cache_root, allow_network=False),
        exchange_information=frozen,
        base_interval=timeframe,
    )
    return build_dataset_from_transport(
        protocol,
        timeframe,
        transport,
        evidence={
            "schema_version": "trend_stop_vision_fallback_v1",
            "timeframe": timeframe,
            "network": "never",
            "exclusions": [item.to_mapping() for item in exclusions],
        },
        exclusions=exclusions,
    )


__all__ = [
    "RosterTooSmallError",
    "SymbolExclusion",
    "active_symbols",
    "build_dataset_from_transport",
    "parse_exclusions",
    "build_vision_dataset",
    "load_exported_dataset",
    "plan_vision_cache",
    "validate_study_dataset",
]
