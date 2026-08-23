from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from trade_rl.integrations.binance_v4_context import (
    BinanceFundingEventSeries,
    BinanceV4KlineSeries,
)
from trade_rl.integrations.binance_v4_context_assembly import (
    AlignedV4KlineSeries,
    align_funding_events_to_decisions,
    align_v4_kline_to_decisions,
    assemble_v4_cross_market_inputs,
    vision_mark_price_kline_url,
)


def _digest(char: str) -> str:
    return char * 64


def _decisions() -> np.ndarray:
    return np.asarray(
        [
            np.datetime64("2026-01-01T00:15"),
            np.datetime64("2026-01-01T00:30"),
            np.datetime64("2026-01-01T00:45"),
        ],
        dtype="datetime64[ns]",
    )


def _series(
    *,
    opens: tuple[int, ...] = (1767225600000, 1767226500000, 1767227400000),
    closes: tuple[float, ...] = (100.0, 101.0, 102.0),
    source: str = "1",
) -> BinanceV4KlineSeries:
    return BinanceV4KlineSeries(
        open_time_ms=np.asarray(opens, dtype=np.int64),
        close_time_ms=np.asarray([value + 899_999 for value in opens], dtype=np.int64),
        close=np.asarray(closes, dtype=np.float64),
        quote_volume=np.asarray([1_000.0, 1_100.0, 1_200.0], dtype=np.float64)[
            : len(opens)
        ],
        taker_buy_quote_volume=np.asarray([550.0, 500.0, 650.0], dtype=np.float64)[
            : len(opens)
        ],
        source_digest=_digest(source),
    )


def test_mark_price_kline_url_is_frozen() -> None:
    assert vision_mark_price_kline_url(
        "BTCUSDT",
        "15m",
        datetime(2026, 1, 2, tzinfo=UTC),
    ) == (
        "https://data.binance.vision/data/futures/um/daily/markPriceKlines/"
        "BTCUSDT/15m/BTCUSDT-15m-2026-01-02.zip"
    )


def test_kline_alignment_uses_exact_closed_bar_without_carry() -> None:
    source = _series(
        opens=(1767225600000, 1767227400000),
        closes=(100.0, 102.0),
    )
    aligned = align_v4_kline_to_decisions(
        _decisions(),
        source,
        interval_minutes=15,
    )
    np.testing.assert_array_equal(aligned.available, [True, False, True])
    np.testing.assert_allclose(aligned.close, [100.0, 0.0, 102.0])
    assert aligned.quote_volume[1] == 0.0
    assert aligned.taker_buy_quote_volume[1] == 0.0


def test_kline_alignment_rejects_source_bar_not_closed_by_decision() -> None:
    source = _series()
    close_time = source.close_time_ms.copy()
    close_time[0] = 1767226500001
    source = replace(source, close_time_ms=close_time, digest="")
    aligned = align_v4_kline_to_decisions(
        _decisions(),
        source,
        interval_minutes=15,
    )
    assert not aligned.available[0]


def test_funding_event_maps_once_to_first_observable_decision() -> None:
    events = BinanceFundingEventSeries(
        event_time_ms=np.asarray([1767226200000], dtype=np.int64),
        rate=np.asarray([0.001], dtype=np.float64),
        source_digest=_digest("2"),
    )
    rate, available, digest = align_funding_events_to_decisions(_decisions(), events)
    np.testing.assert_allclose(rate, [0.001, 0.0, 0.0])
    np.testing.assert_array_equal(available, [True, False, False])
    assert len(digest) == 64


def test_funding_event_after_last_decision_is_not_visible() -> None:
    events = BinanceFundingEventSeries(
        event_time_ms=np.asarray([1767229200000], dtype=np.int64),
        rate=np.asarray([0.001], dtype=np.float64),
        source_digest=_digest("2"),
    )
    rate, available, _ = align_funding_events_to_decisions(_decisions(), events)
    np.testing.assert_allclose(rate, [0.0, 0.0, 0.0])
    assert not np.any(available)


def _aligned(
    *,
    values: tuple[float, float, float],
    available: tuple[bool, bool, bool] = (True, True, True),
    source: str,
) -> AlignedV4KlineSeries:
    return AlignedV4KlineSeries(
        close=np.asarray(values, dtype=np.float64),
        quote_volume=np.asarray([1000.0, 1000.0, 1000.0], dtype=np.float64),
        taker_buy_quote_volume=np.asarray([500.0, 500.0, 500.0], dtype=np.float64),
        available=np.asarray(available, dtype=np.bool_),
        source_digest=_digest(source),
    )


def test_cross_market_assembly_requires_mark_price_for_perp_row() -> None:
    decisions = _decisions()
    spot = _aligned(values=(100.0, 101.0, 102.0), source="1")
    perp = _aligned(values=(100.5, 101.5, 102.5), source="2")
    mark = _aligned(
        values=(100.25, 101.25, 102.25),
        available=(True, False, True),
        source="3",
    )
    funding_rate = np.asarray([0.001, 0.0, 0.0], dtype=np.float64)
    funding_available = np.asarray([True, False, False], dtype=np.bool_)

    result = assemble_v4_cross_market_inputs(
        decision_indices=np.asarray([10, 11, 12], dtype=np.int64),
        decision_timestamps=decisions,
        spot=spot,
        perp=perp,
        mark=mark,
        funding_event_rate=funding_rate,
        funding_event_available=funding_available,
        metrics=None,
    )
    np.testing.assert_array_equal(result.spot_row_available, [True, True, True])
    np.testing.assert_array_equal(result.perp_row_available, [True, False, True])
    np.testing.assert_allclose(result.perp_mark_price, [100.25, 0.0, 102.25])
    assert result.open_interest_value is None
    assert result.derivatives_available is None
    assert len(result.source_digest) == 64


def test_cross_market_assembly_source_digest_binds_mark_source() -> None:
    decisions = _decisions()
    common = dict(
        decision_indices=np.asarray([10, 11, 12], dtype=np.int64),
        decision_timestamps=decisions,
        spot=_aligned(values=(100.0, 101.0, 102.0), source="1"),
        perp=_aligned(values=(100.5, 101.5, 102.5), source="2"),
        funding_event_rate=np.asarray([0.001, 0.0, 0.0], dtype=np.float64),
        funding_event_available=np.asarray([True, False, False], dtype=np.bool_),
        metrics=None,
    )
    left = assemble_v4_cross_market_inputs(
        mark=_aligned(values=(100.25, 101.25, 102.25), source="3"),
        **common,
    )
    right = assemble_v4_cross_market_inputs(
        mark=_aligned(values=(100.25, 101.25, 102.25), source="4"),
        **common,
    )
    assert left.source_digest != right.source_digest
