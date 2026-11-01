"""Gap policy and base-interval aggregation for offline development builds."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from trade_rl.integrations.binance.dataset import (
    BinanceGapPolicy,
    BinanceMarketDataSource,
    build_binance_market_dataset,
)
from trade_rl.integrations.binance.derived_klines import (
    BaseIntervalAggregatingTransport,
    aggregate_klines,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
HOUR_MS = 3_600_000
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _row(open_ms: int, close: float, *, quote_volume: float = 1_000.0) -> list[Any]:
    return [
        open_ms,
        str(close * 0.999),
        str(close * 1.01),
        str(close * 0.99),
        str(close),
        "10.0",
        open_ms + HOUR_MS - 1,
        str(quote_volume),
        5,
        "1",
        "1",
        "0",
    ]


class HourlyTransport:
    """Deterministic 1h klines; ``missing`` maps symbol -> missing hour indices."""

    def __init__(
        self,
        missing: dict[str, set[int]] | None = None,
        funding_hours: tuple[int, ...] = (),
    ) -> None:
        self.missing = missing or {}
        self.funding_hours = funding_hours
        self.kline_intervals: list[str] = []

    def load_klines(
        self, *, symbol: str, interval: str, start_ms: int, end_ms: int, **_: object
    ) -> tuple[list[list[Any]], str]:
        self.kline_intervals.append(interval)
        assert interval == "1h"
        rng = np.random.default_rng(1 if symbol == SYMBOLS[0] else 2)
        price = 100.0
        rows: list[list[Any]] = []
        for index, open_ms in enumerate(range(start_ms, end_ms, HOUR_MS)):
            price *= float(np.exp(rng.normal(0.0, 0.01)))
            if index in self.missing.get(symbol, set()):
                continue
            rows.append(_row(open_ms, price, quote_volume=1_000.0 + index))
        return rows, "fixture"

    def load_funding_rates(
        self, *, start_ms: int, **_: object
    ) -> tuple[list[tuple[int, float]], str]:
        return (
            [(start_ms + hour * HOUR_MS + 1, 0.0001) for hour in self.funding_hours],
            "fixture",
        )

    def load_exchange_information(self, **_: object) -> tuple[dict[str, object], str]:
        raise AssertionError("metadata is supplied explicitly in these tests")


def _source(
    transport: object,
    *,
    hours: int = 48,
    policy: BinanceGapPolicy | str = BinanceGapPolicy.STRICT,
) -> BinanceMarketDataSource:
    return BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=hours),
        transport_mode="vision",
        transport=transport,
        gap_policy=policy,
    )


def _build(
    transport: object,
    *,
    days: int = 20,
    policy: BinanceGapPolicy | str = BinanceGapPolicy.NON_TRADABLE_ROWS,
) -> Any:
    return build_binance_market_dataset(
        market="usds-m",
        symbols=SYMBOLS,
        interval="1h",
        start_time=START,
        end_time=START + timedelta(days=days),
        transport_mode="vision",
        transport=transport,
        tick_sizes=(0.01, 0.01),
        lot_sizes=(0.001, 0.001),
        minimum_notionals=(5.0, 5.0),
        listed_ats=(START - timedelta(days=100),) * 2,
        feature_timeframes=("4h", "1d"),
        gap_policy=policy,
    )


# --- strict default ---------------------------------------------------------


def test_strict_is_the_default_and_rejects_a_missing_bar() -> None:
    transport = HourlyTransport({SYMBOLS[1]: {10}})
    source = _source(transport)

    assert source.gap_policy is BinanceGapPolicy.STRICT
    with pytest.raises(ValueError, match="complete and exactly regular"):
        source.load(SYMBOLS[1])
    assert source.gap_report == {}


def test_strict_builder_default_rejects_a_missing_bar() -> None:
    transport = BaseIntervalAggregatingTransport(HourlyTransport({SYMBOLS[1]: {40}}))

    with pytest.raises(ValueError, match="complete and exactly regular"):
        _build(transport, policy=BinanceGapPolicy.STRICT)


def test_strict_series_is_unchanged_by_the_option_for_a_complete_range() -> None:
    complete = _source(HourlyTransport()).load(SYMBOLS[0])
    explicit = _source(HourlyTransport(), policy="strict").load(SYMBOLS[0])
    tolerant = _source(HourlyTransport(), policy="non_tradable_rows").load(SYMBOLS[0])

    for other in (explicit, tolerant):
        assert np.array_equal(complete.timestamps, other.timestamps)
        assert np.array_equal(complete.available_at, other.available_at)
        for name in ("open", "high", "low", "close", "volume", "funding_rate"):
            assert np.array_equal(getattr(complete, name), getattr(other, name))
        assert np.array_equal(complete.tradable, other.tradable)


# --- gap policy -------------------------------------------------------------


def test_missing_base_bars_are_absent_rows_and_funding_in_them_is_not_accrued() -> None:
    # Hour 10 closes at hour 11; the funding event at +10h+1ms lands in that bar.
    transport = HourlyTransport({SYMBOLS[1]: {10, 11}}, funding_hours=(5, 10, 20))
    source = _source(transport, policy=BinanceGapPolicy.NON_TRADABLE_ROWS)

    series = source.load(SYMBOLS[1])
    complete = source.load(SYMBOLS[0])

    assert len(complete.timestamps) == 48
    assert len(series.timestamps) == 46
    origin = np.datetime64(START.replace(tzinfo=None), "ns")
    missing_closes = np.array([origin + np.timedelta64(h, "h") for h in (11, 12)])
    assert not np.isin(series.timestamps, missing_closes).any()
    assert np.isin(missing_closes, complete.timestamps).all()
    assert bool(series.tradable.all())
    assert source.gap_report[f"{SYMBOLS[1]}:1h"] == {
        "expected_bars": 48,
        "present_bars": 46,
        "missing_bars": 2,
        "funding_events_in_missing_bars": 1,
    }
    # The two surviving events accrue; the one inside the gap does not.
    assert int(series.funding_event_count.sum()) == 2
    assert int(complete.funding_event_count.sum()) == 3


def test_higher_clock_placeholder_is_price_carried_and_unavailable_until_range_end() -> (
    None
):
    transport = BaseIntervalAggregatingTransport(HourlyTransport({SYMBOLS[1]: {5}}))
    source = _source(transport, hours=24, policy=BinanceGapPolicy.NON_TRADABLE_ROWS)
    range_end = np.datetime64((START + timedelta(hours=24)).replace(tzinfo=None), "ns")

    four_hour = source.load_timeframe(SYMBOLS[1], "4h")
    clean = source.load_timeframe(SYMBOLS[0], "4h")

    assert len(four_hour.timestamps) == 6
    # Window 1 (hours 4..7) contains the missing hour 5: placeholder slot.
    assert four_hour.available_at[1] == range_end
    assert four_hour.volume[1] == 0.0
    assert four_hour.open[1] == four_hour.close[1] == four_hour.close[0]
    assert bool(
        (
            four_hour.available_at[[0, 2, 3, 4, 5]]
            == four_hour.timestamps[[0, 2, 3, 4, 5]]
        ).all()
    )
    assert bool((clean.available_at == clean.timestamps).all())
    assert transport.dropped_windows[(SYMBOLS[1], "4h")] == 1
    assert transport.dropped_windows[(SYMBOLS[0], "4h")] == 0


def test_series_must_start_with_a_present_bar_under_the_gap_policy() -> None:
    source = _source(
        HourlyTransport({SYMBOLS[0]: {0}}),
        policy=BinanceGapPolicy.NON_TRADABLE_ROWS,
    )

    with pytest.raises(ValueError, match="must start with a present"):
        source.load(SYMBOLS[0])


def test_gap_policy_rejects_unknown_values_and_duplicate_rows() -> None:
    with pytest.raises(ValueError):
        _source(HourlyTransport(), policy="skip")

    class Duplicating(HourlyTransport):
        def load_klines(self, **kwargs: Any) -> tuple[list[list[Any]], str]:
            rows, source = super().load_klines(**kwargs)
            return [*rows, rows[3]], source

    source = _source(Duplicating(), policy=BinanceGapPolicy.NON_TRADABLE_ROWS)
    with pytest.raises(ValueError, match="duplicate"):
        source.load(SYMBOLS[0])


def test_built_dataset_marks_gap_rows_non_tradable_and_masks_recursive_features() -> (
    None
):
    transport = BaseIntervalAggregatingTransport(
        HourlyTransport({SYMBOLS[1]: {100, 101, 102, 103}})
    )

    result = _build(transport)
    dataset = result.dataset
    tradable = np.asarray(dataset.tradable)
    available = np.asarray(dataset.feature_available)
    names = list(dataset.feature_names)

    assert dataset.n_bars == 480
    assert (~tradable).sum(axis=0).tolist() == [0, 4]
    assert not tradable[100:104, 1].any()
    assert result.gap_report["ETHUSDT:1h"]["missing_bars"] == 4
    assert result.gap_report["BTCUSDT:1h"]["missing_bars"] == 0
    # Recursive 4h features (26-bar EMA/MACD) never recover after the gap, the
    # complete symbol keeps them; 1h features recover once the window clears.
    for name in ("4h__ema_slope_26", "4h__macd_line_12_26"):
        column = names.index(name)
        assert available[:, 1, column].sum() <= 1
        assert not available[200:-1, 1, column].any()
        assert available[200:-1, 0, column].all()
    one_hour = names.index("1h__log_return_1bar")
    assert not available[100:105, 1, one_hour].all()
    assert available[300:, 1, one_hour].all()


# --- aggregation ------------------------------------------------------------


def test_aggregate_klines_matches_manual_ohlcv() -> None:
    rows = [
        [
            _ms(START) + index * HOUR_MS,
            str(open_),
            str(high),
            str(low),
            str(close),
            str(volume),
            _ms(START) + (index + 1) * HOUR_MS - 1,
            str(quote),
        ]
        for index, (open_, high, low, close, volume, quote) in enumerate(
            [
                (10.0, 12.0, 9.0, 11.0, 1.0, 100.0),
                (11.0, 15.0, 10.5, 14.0, 2.0, 200.0),
                (14.0, 14.5, 8.0, 9.0, 3.0, 300.0),
                (9.0, 10.0, 8.5, 9.5, 4.0, 400.0),
            ]
        )
    ]
    start_ms = _ms(START)

    aggregated, dropped = aggregate_klines(
        rows,
        base_interval="1h",
        interval="4h",
        start_ms=start_ms,
        end_ms=start_ms + 4 * HOUR_MS,
    )

    assert dropped == 0
    assert len(aggregated) == 1
    open_ms, open_, high, low, close, volume, close_ms, quote = aggregated[0]
    assert open_ms == start_ms
    assert open_ms + 4 * HOUR_MS - 1 == close_ms
    assert (float(open_), float(high), float(low), float(close)) == (
        10.0,
        15.0,
        8.0,
        9.5,
    )
    assert float(volume) == 10.0
    assert float(quote) == 1000.0


def test_aggregate_klines_drops_incomplete_windows_and_never_interpolates() -> None:
    start_ms = _ms(START)
    rows = [_row(start_ms + index * HOUR_MS, 100.0 + index) for index in range(8)]
    del rows[5]

    aggregated, dropped = aggregate_klines(
        rows,
        base_interval="1h",
        interval="4h",
        start_ms=start_ms,
        end_ms=start_ms + 8 * HOUR_MS,
    )

    assert dropped == 1
    assert [row[0] for row in aggregated] == [start_ms]


def test_aggregate_klines_accepts_microsecond_open_times() -> None:
    start_ms = _ms(START)
    rows = [_row(start_ms + index * HOUR_MS, 100.0 + index) for index in range(4)]
    micro = [[int(row[0]) * 1_000, *row[1:]] for row in rows]

    plain, _ = aggregate_klines(
        rows,
        base_interval="1h",
        interval="4h",
        start_ms=start_ms,
        end_ms=start_ms + 4 * HOUR_MS,
    )
    scaled, _ = aggregate_klines(
        micro,
        base_interval="1h",
        interval="4h",
        start_ms=start_ms,
        end_ms=start_ms + 4 * HOUR_MS,
    )

    assert plain == scaled


def test_aggregate_klines_daily_from_hourly_and_input_validation() -> None:
    start_ms = _ms(START)
    rows = [_row(start_ms + index * HOUR_MS, 100.0 + index) for index in range(48)]

    daily, dropped = aggregate_klines(
        rows,
        base_interval="1h",
        interval="1d",
        start_ms=start_ms,
        end_ms=start_ms + 48 * HOUR_MS,
    )

    assert dropped == 0
    assert [row[0] for row in daily] == [start_ms, start_ms + 24 * HOUR_MS]
    assert float(daily[0][5]) == 240.0
    with pytest.raises(ValueError, match="multiple of the base"):
        aggregate_klines(
            rows,
            base_interval="4h",
            interval="1h",
            start_ms=start_ms,
            end_ms=start_ms + 4 * HOUR_MS,
        )
    with pytest.raises(ValueError, match="align"):
        aggregate_klines(
            rows,
            base_interval="1h",
            interval="4h",
            start_ms=start_ms + HOUR_MS,
            end_ms=start_ms + 5 * HOUR_MS,
        )
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_klines(
            [rows[0], rows[0]],
            base_interval="1h",
            interval="4h",
            start_ms=start_ms,
            end_ms=start_ms + 4 * HOUR_MS,
        )
    with pytest.raises(ValueError, match="eight fields"):
        aggregate_klines(
            [[start_ms, "1"]],
            base_interval="1h",
            interval="4h",
            start_ms=start_ms,
            end_ms=start_ms + 4 * HOUR_MS,
        )


def test_aggregating_transport_delegates_base_funding_and_metadata() -> None:
    class Exchange:
        def load_exchange_information(self, **_: object) -> tuple[str, str]:
            return "info", "frozen"

    inner = HourlyTransport(funding_hours=(1,))
    transport = BaseIntervalAggregatingTransport(inner, exchange_information=Exchange())
    start_ms = _ms(START)

    rows, source = transport.load_klines(
        market="usds-m",
        symbol=SYMBOLS[0],
        interval="1h",
        start_ms=start_ms,
        end_ms=start_ms + 4 * HOUR_MS,
        mode="vision",
    )
    four, _ = transport.load_klines(
        market="usds-m",
        symbol=SYMBOLS[0],
        interval="4h",
        start_ms=start_ms,
        end_ms=start_ms + 4 * HOUR_MS,
        mode="vision",
    )

    assert source == "fixture"
    assert len(rows) == 4
    assert len(four) == 1
    assert inner.kline_intervals == ["1h", "1h"]
    assert transport.load_funding_rates(start_ms=start_ms)[0][0][1] == 0.0001
    assert transport.load_exchange_information() == ("info", "frozen")
    with pytest.raises(ValueError):
        BaseIntervalAggregatingTransport(inner, base_interval="7h")
