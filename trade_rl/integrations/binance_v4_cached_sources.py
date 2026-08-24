"""Verified cached-source loading for Causal Alpha V4 Binance context."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.v4_context import V4CrossMarketInputs
from trade_rl.integrations.binance import validate_cached_vision_payload
from trade_rl.integrations.binance_cache import vision_cache_path
from trade_rl.integrations.binance_v4_context import (
    BinanceFundingEventSeries,
    BinanceFuturesMetricsSeries,
    BinanceV4KlineSeries,
    align_futures_metrics_to_decisions,
    parse_binance_funding_archive,
    parse_binance_futures_metrics_archive,
    parse_binance_v4_kline_archive,
)
from trade_rl.integrations.binance_v4_context_assembly import (
    align_funding_events_to_decisions,
    align_v4_kline_to_decisions,
    assemble_v4_cross_market_inputs,
)


def _urls(value: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(item) for item in value)
    if not result or any(not item for item in result):
        raise ValueError("V4 cached source URLs must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError("V4 cached source URLs must be unique")
    return result


def _cached_payload(cache_root: str | Path, url: str) -> bytes:
    path = vision_cache_path(cache_root, url)
    if not path.is_file():
        raise FileNotFoundError(f"V4 cached source is missing: {path}")
    return validate_cached_vision_payload(url, path)


def _child_source_digest(
    *, schema_version: str, urls: tuple[str, ...], child_digests: tuple[str, ...]
) -> str:
    return content_digest(
        {
            "child_digests": child_digests,
            "schema_version": schema_version,
            "urls": urls,
        }
    )


def load_cached_v4_kline_series(
    urls: Sequence[str],
    *,
    cache_root: str | Path,
) -> BinanceV4KlineSeries:
    """Validate, parse, and concatenate authored kline archives without de-duplication."""

    source_urls = _urls(urls)
    blocks = tuple(
        parse_binance_v4_kline_archive(
            _cached_payload(cache_root, url),
            source_uri=url,
        )
        for url in source_urls
    )
    return BinanceV4KlineSeries(
        open_time_ms=np.concatenate(tuple(item.open_time_ms for item in blocks)),
        close_time_ms=np.concatenate(tuple(item.close_time_ms for item in blocks)),
        close=np.concatenate(tuple(item.close for item in blocks)),
        quote_volume=np.concatenate(tuple(item.quote_volume for item in blocks)),
        taker_buy_quote_volume=np.concatenate(
            tuple(item.taker_buy_quote_volume for item in blocks)
        ),
        source_digest=_child_source_digest(
            schema_version="binance_v4_cached_kline_series_v1",
            urls=source_urls,
            child_digests=tuple(item.digest for item in blocks),
        ),
    )


def load_cached_v4_funding_events(
    urls: Sequence[str],
    *,
    cache_root: str | Path,
) -> BinanceFundingEventSeries:
    """Validate, parse, and concatenate authored funding-event archives."""

    source_urls = _urls(urls)
    blocks = tuple(
        parse_binance_funding_archive(
            _cached_payload(cache_root, url),
            source_uri=url,
        )
        for url in source_urls
    )
    return BinanceFundingEventSeries(
        event_time_ms=np.concatenate(tuple(item.event_time_ms for item in blocks)),
        rate=np.concatenate(tuple(item.rate for item in blocks)),
        source_digest=_child_source_digest(
            schema_version="binance_v4_cached_funding_series_v1",
            urls=source_urls,
            child_digests=tuple(item.digest for item in blocks),
        ),
    )


def load_cached_v4_metrics_series(
    urls: Sequence[str],
    *,
    cache_root: str | Path,
    expected_symbol: str,
) -> BinanceFuturesMetricsSeries:
    """Validate, parse, and concatenate authored daily USD-M metrics archives."""

    source_urls = _urls(urls)
    blocks = tuple(
        parse_binance_futures_metrics_archive(
            _cached_payload(cache_root, url),
            source_uri=url,
            expected_symbol=expected_symbol,
        )
        for url in source_urls
    )
    return BinanceFuturesMetricsSeries(
        create_time=np.concatenate(tuple(item.create_time for item in blocks)),
        open_interest_value=np.concatenate(
            tuple(item.open_interest_value for item in blocks)
        ),
        global_long_short_ratio=np.concatenate(
            tuple(item.global_long_short_ratio for item in blocks)
        ),
        top_position_long_short_ratio=np.concatenate(
            tuple(item.top_position_long_short_ratio for item in blocks)
        ),
        source_digest=_child_source_digest(
            schema_version="binance_v4_cached_metrics_series_v1",
            urls=source_urls,
            child_digests=tuple(item.digest for item in blocks),
        ),
    )


def build_cached_v4_cross_market_inputs(
    *,
    decision_indices: object,
    decision_timestamps: object,
    cache_root: str | Path,
    spot_kline_urls: Sequence[str],
    perp_kline_urls: Sequence[str],
    mark_price_kline_urls: Sequence[str],
    funding_urls: Sequence[str],
    metrics_urls: Sequence[str] = (),
    expected_symbol: str,
) -> V4CrossMarketInputs:
    """Build exact decision-clock V4 input arrays only from verified cache bytes."""

    spot_raw = load_cached_v4_kline_series(spot_kline_urls, cache_root=cache_root)
    perp_raw = load_cached_v4_kline_series(perp_kline_urls, cache_root=cache_root)
    mark_raw = load_cached_v4_kline_series(mark_price_kline_urls, cache_root=cache_root)
    funding_raw = load_cached_v4_funding_events(funding_urls, cache_root=cache_root)

    spot = align_v4_kline_to_decisions(
        decision_timestamps,
        spot_raw,
        interval_minutes=15,
    )
    perp = align_v4_kline_to_decisions(
        decision_timestamps,
        perp_raw,
        interval_minutes=15,
    )
    mark = align_v4_kline_to_decisions(
        decision_timestamps,
        mark_raw,
        interval_minutes=15,
    )
    funding_rate, funding_available, _ = align_funding_events_to_decisions(
        decision_timestamps,
        funding_raw,
    )

    metrics = None
    metric_urls = tuple(metrics_urls)
    if metric_urls:
        metric_series = load_cached_v4_metrics_series(
            metric_urls,
            cache_root=cache_root,
            expected_symbol=expected_symbol,
        )
        metrics = align_futures_metrics_to_decisions(
            decision_timestamps,
            metric_series,
        )

    return assemble_v4_cross_market_inputs(
        decision_indices=decision_indices,
        decision_timestamps=decision_timestamps,
        spot=spot,
        perp=perp,
        mark=mark,
        funding_event_rate=funding_rate,
        funding_event_available=funding_available,
        metrics=metrics,
    )


__all__ = [
    "build_cached_v4_cross_market_inputs",
    "load_cached_v4_funding_events",
    "load_cached_v4_kline_series",
    "load_cached_v4_metrics_series",
]
