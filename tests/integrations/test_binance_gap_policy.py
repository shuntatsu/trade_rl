"""Gap policy and base-interval aggregation for offline development builds."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from inspect import signature
from typing import Any

import numpy as np
import pytest

from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
)
from trade_rl.data.features.cross_asset import CROSS_ASSET_FEATURE_KINDS
from trade_rl.data.features.multitimeframe import _native_events, align_native_feature
from trade_rl.data.source import RawMarketSeries
from trade_rl.integrations.binance.dataset import (
    BinanceGapPolicy,
    BinanceMarketDataSource,
    _gap_tolerant_series,
    build_binance_market_dataset,
)
from trade_rl.integrations.binance.derived_klines import (
    BaseIntervalAggregatingTransport,
    aggregate_klines,
)
from trade_rl.integrations.market_store.cli import build_parser
from trade_rl.integrations.market_store.export import export_market_dataset

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


@pytest.mark.parametrize(
    "missing_indices",
    ((0,), (47,), (0, 47)),
    ids=("missing-first", "missing-last", "missing-both"),
)
def test_strict_rejects_missing_requested_boundary_bars(
    missing_indices: tuple[int, ...],
) -> None:
    """A gap at either endpoint must not silently shorten a requested range."""
    source = _source(HourlyTransport({SYMBOLS[0]: set(missing_indices)}))
    with pytest.raises(ValueError, match="does not cover requested boundaries"):
        source.load(SYMBOLS[0])
    assert source.gap_report == {}


def test_direct_and_market_store_gap_policy_defaults_are_distinct() -> None:
    """The different defaults are deliberate; do not claim STRICT everywhere."""
    assert (
        signature(BinanceMarketDataSource).parameters["gap_policy"].default
        is BinanceGapPolicy.STRICT
    )
    assert (
        signature(build_binance_market_dataset).parameters["gap_policy"].default
        is BinanceGapPolicy.STRICT
    )
    assert (
        signature(export_market_dataset).parameters["gap_policy"].default
        is BinanceGapPolicy.NON_TRADABLE_ROWS
    )
    args = build_parser().parse_args(
        [
            "export",
            "--market",
            "spot",
            "--start",
            "2020-12-01",
            "--end",
            "2021-01-01",
            "--output-dir",
            "unused-output",
        ]
    )
    assert args.gap_policy == BinanceGapPolicy.NON_TRADABLE_ROWS.value


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


def test_higher_clock_placeholder_is_price_carried_non_tradable_and_unpublished() -> (
    None
):
    transport = BaseIntervalAggregatingTransport(HourlyTransport({SYMBOLS[1]: {5}}))
    source = _source(transport, hours=24, policy=BinanceGapPolicy.NON_TRADABLE_ROWS)

    four_hour = source.load_timeframe(SYMBOLS[1], "4h")
    clean = source.load_timeframe(SYMBOLS[0], "4h")

    assert len(four_hour.timestamps) == 6
    # Window 1 (hours 4..7) contains the missing hour 5: placeholder slot. It
    # is never published (not at the requested range end) and not tradable.
    assert four_hour.available_at[1] == NEVER_AVAILABLE
    assert four_hour.tradable.tolist() == [True, False, True, True, True, True]
    assert bool(clean.tradable.all())
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


def test_shared_missing_interior_hour_keeps_non_tradable_clock_row() -> None:
    """An interior hole in every symbol still has a shared clock position."""
    missing = {symbol: {100} for symbol in SYMBOLS}
    result = _build(BaseIntervalAggregatingTransport(HourlyTransport(missing)))
    dataset = result.dataset
    assert dataset.n_bars == 20 * 24
    assert not bool(dataset.tradable[100, :].any())
    assert bool(dataset.tradable[99, :].all())
    assert bool(dataset.tradable[101, :].all())
    for symbol in SYMBOLS:
        assert result.gap_report[f"{symbol}:1h"]["missing_bars"] == 1


def test_shared_missing_final_bar_cannot_shorten_requested_dataset() -> None:
    """No symbol can supply the final clock: refuse rather than omit it."""
    last_hour = 20 * 24 - 1
    missing = {symbol: {last_hour} for symbol in SYMBOLS}
    with pytest.raises(ValueError, match="does not cover requested boundaries"):
        _build(BaseIntervalAggregatingTransport(HourlyTransport(missing)))


def test_single_symbol_missing_final_bar_keeps_complete_common_clock() -> None:
    """The other symbol's final mark keeps the missing slot non-tradable."""
    last_hour = 20 * 24 - 1
    result = _build(
        BaseIntervalAggregatingTransport(HourlyTransport({SYMBOLS[0]: {last_hour}}))
    )
    dataset = result.dataset
    assert dataset.n_bars == 20 * 24
    assert dataset.timestamps[-1] == np.datetime64(
        (START + timedelta(days=20)).replace(tzinfo=None), "ns"
    )
    assert not bool(dataset.tradable[-1, 0])
    assert bool(dataset.tradable[-1, 1])
    assert result.gap_report[f"{SYMBOLS[0]}:1h"]["missing_bars"] == 1
    assert result.gap_report[f"{SYMBOLS[1]}:1h"]["missing_bars"] == 0


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
    # The 4h window of hours 100..103 (native bar 25) is a placeholder. Native
    # features restart after it exactly as after a base gap: no 4h value is
    # available between the gap and the end of the new warm-up, then they
    # recover; the complete symbol keeps them throughout.
    for name, warmup in (("4h__ema_slope_26", 26), ("4h__macd_line_12_26", 26)):
        column = names.index(name)
        # Base row r observes native bar floor((r + 1) / 4) - 1; the new
        # segment starts at native bar 26, first valid at 26 + warmup - 1.
        recovered = 4 * (26 + warmup) - 1
        assert not available[103:recovered, 1, column].any()
        assert available[recovered:, 1, column].all()
        assert available[200:, 0, column].all()
    one_hour = names.index("1h__log_return_1bar")
    assert not available[100:105, 1, one_hour].all()
    assert available[300:, 1, one_hour].all()


# --- higher-clock placeholder causality ----------------------------------------

FOUR_HOUR_MS = 4 * HOUR_MS
NEVER_AVAILABLE = np.datetime64(np.iinfo(np.int64).max, "ns")
_RECURSIVE_KINDS = (
    (FeatureKind.EMA_DISTANCE, 5),
    (FeatureKind.EMA_SLOPE, 5),
    (FeatureKind.RSI, 5),
    (FeatureKind.ATR_PCT, 5),
    (FeatureKind.ADX, 5),
    (FeatureKind.ADX_CAUSAL, 5),
    (FeatureKind.LOG_RETURN, 1),
    (FeatureKind.REALIZED_VOLATILITY, 5),
)


def _four_hour_rows(n_bars: int, missing: set[int]) -> list[list[Any]]:
    rng = np.random.default_rng(1)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n_bars)))
    rows: list[list[Any]] = []
    for index in range(n_bars):
        if index in missing:
            continue
        open_ms = _ms(START) + index * FOUR_HOUR_MS
        close = float(closes[index])
        rows.append(
            [open_ms, close, close * 1.01, close * 0.99, close, 1.0]
            + [open_ms + FOUR_HOUR_MS - 1, 1_000.0]
        )
    return rows


def _gap_native(n_bars: int, missing: set[int]) -> Any:
    end_ms = _ms(START) + n_bars * FOUR_HOUR_MS
    series, _ = _gap_tolerant_series(
        _four_hour_rows(n_bars, missing),
        symbol="X",
        interval_ms=FOUR_HOUR_MS,
        start_ms=_ms(START),
        end_ms=end_ms,
        is_base_clock=False,
        events=[],
    )
    return series


def _aligned_4h(
    n_bars: int, missing: set[int], kind: FeatureKind, lookback: int, stale: float
) -> tuple[np.ndarray, ...]:
    native = _gap_native(n_bars, missing)
    base = (
        np.datetime64(START.replace(tzinfo=None), "ns")
        + np.timedelta64(1, "h")
        + np.arange(n_bars * 4) * np.timedelta64(1, "h")
    )
    spec = FeatureSpec(
        kind.value,
        kind,
        lookback=lookback,
        timeframe="4h",
        max_staleness_hours=stale,
    )
    contract = InstrumentContract("X", listed_at=START - timedelta(days=100))
    return align_native_feature(
        spec, native, contract, base, np.ones(len(base), bool), timeframe="4h"
    )


@pytest.mark.parametrize(("kind", "lookback"), _RECURSIVE_KINDS)
@pytest.mark.parametrize("stale", (24.0, 1e6))
def test_higher_clock_gap_features_do_not_depend_on_requested_range_end(
    kind: FeatureKind, lookback: int, stale: float
) -> None:
    """A missing 4h bar is never published at the (caller-chosen) range end."""
    short = _aligned_4h(30, {20}, kind, lookback, stale)
    long = _aligned_4h(40, {20}, kind, lookback, stale)
    rows = len(short[0])
    for short_array, long_array in zip(short, long, strict=True):
        assert np.array_equal(short_array, long_array[:rows])


def test_higher_clock_placeholder_is_never_published_or_tradable() -> None:
    native = _gap_native(30, {20})
    origin = np.datetime64(START.replace(tzinfo=None), "ns")
    closes = origin + (np.arange(30) + 1) * np.timedelta64(4, "h")

    assert np.array_equal(native.timestamps, closes)
    assert native.tradable.tolist() == [index != 20 for index in range(30)]
    assert native.available_at[20] == NEVER_AVAILABLE
    present = np.arange(30) != 20
    assert np.array_equal(native.available_at[present], closes[present])


@pytest.mark.parametrize(("kind", "lookback"), _RECURSIVE_KINDS)
def test_no_native_feature_window_bridges_a_placeholder(
    kind: FeatureKind, lookback: int
) -> None:
    native = _gap_native(40, {20})
    spec = FeatureSpec(kind.value, kind, lookback=lookback, timeframe="4h")
    contract = InstrumentContract("X", listed_at=START - timedelta(days=100))
    values, valid, available_at = _native_events(spec, native, contract)

    events = np.flatnonzero(valid)
    assert events.size
    assert 20 not in events
    # Every published event depends only on present (published) bars ...
    assert bool((available_at[events] < NEVER_AVAILABLE).all())
    assert bool((available_at[events] == native.timestamps[events]).all())
    # ... and the post-gap segment restarts exactly as a fresh series would.
    after = native.timestamps > native.timestamps[20]
    fresh = _gap_tolerant_series(
        _four_hour_rows(40, set(range(21))),
        symbol="X",
        interval_ms=FOUR_HOUR_MS,
        start_ms=_ms(START) + 21 * FOUR_HOUR_MS,
        end_ms=_ms(START) + 40 * FOUR_HOUR_MS,
        is_base_clock=False,
        events=[],
    )[0]
    fresh_values, fresh_valid, _ = _native_events(spec, fresh, contract)
    assert np.array_equal(valid[after], fresh_valid)
    assert np.array_equal(values[after][fresh_valid], fresh_values[fresh_valid])


def _single_symbol_kind_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for kind in FeatureKind:
        if kind in CROSS_ASSET_FEATURE_KINDS:
            continue
        for lookback in (5, 14, 26, 52):
            try:
                specs.append(
                    FeatureSpec(f"{kind.value}_{lookback}", kind, lookback=lookback)
                )
            except ValueError:
                continue
    return specs


def test_every_native_kind_restarts_after_a_placeholder() -> None:
    """No single-symbol kind lets a window span the placeholder (full roster)."""
    specs = _single_symbol_kind_specs()
    assert {spec.kind for spec in specs} == set(FeatureKind) - set(
        CROSS_ASSET_FEATURE_KINDS
    )
    contract = InstrumentContract("X", listed_at=START - timedelta(days=100))
    native = _gap_native(120, {70})
    fresh = _gap_tolerant_series(
        _four_hour_rows(120, set(range(71))),
        symbol="X",
        interval_ms=FOUR_HOUR_MS,
        start_ms=_ms(START) + 71 * FOUR_HOUR_MS,
        end_ms=_ms(START) + 120 * FOUR_HOUR_MS,
        is_base_clock=False,
        events=[],
    )[0]
    after = native.timestamps > native.timestamps[70]
    for spec in specs:
        values, valid, available_at = _native_events(spec, native, contract)
        fresh_values, fresh_valid, _ = _native_events(spec, fresh, contract)
        assert not valid[70], spec.name
        assert bool((available_at[valid] < NEVER_AVAILABLE).all()), spec.name
        if spec.kind is FeatureKind.VOLUME_ZSCORE:
            # Gap-tolerant by contract (samples present rows only): covered by
            # its own oracle test below, not by the restart equality.
            continue
        assert np.array_equal(valid[after], fresh_valid), spec.name
        assert np.array_equal(values[after][fresh_valid], fresh_values[fresh_valid]), (
            spec.name
        )


@pytest.mark.parametrize("lookback", (5, 14))
def test_gap_tolerant_volume_zscore_uses_only_present_rows_and_their_availability(
    lookback: int,
) -> None:
    """The z-score samples present rows; the placeholder never gates it."""
    spec = FeatureSpec("vz", FeatureKind.VOLUME_ZSCORE, lookback=lookback)
    contract = InstrumentContract("X", listed_at=START - timedelta(days=100))
    rows = _four_hour_rows(60, {30})
    rng = np.random.default_rng(7)
    for row in rows:
        row[5] = float(rng.uniform(1.0, 10.0))
    kwargs: dict[str, Any] = {
        "symbol": "X",
        "interval_ms": FOUR_HOUR_MS,
        "start_ms": _ms(START),
        "is_base_clock": False,
        "events": [],
    }
    short = _gap_tolerant_series(rows, end_ms=_ms(START) + 40 * FOUR_HOUR_MS, **kwargs)[
        0
    ]
    full = _gap_tolerant_series(rows, end_ms=_ms(START) + 60 * FOUR_HOUR_MS, **kwargs)[
        0
    ]
    short_values, short_valid, short_at = _native_events(spec, short, contract)
    values, valid, available_at = _native_events(spec, full, contract)

    # Range-end independence at every shared row, including windows holding the gap.
    assert np.array_equal(valid[:40], short_valid)
    assert np.array_equal(values[:40][short_valid], short_values[short_valid])
    assert np.array_equal(available_at[:40][short_valid], short_at[short_valid])
    # Oracle: z-score over the present rows of the trailing window; published at
    # the event's own close, never at a range end or the sentinel.
    volume = np.asarray(full.volume)
    present = np.asarray(full.tradable)
    for index in np.flatnonzero(valid):
        window = slice(max(0, index - lookback + 1), index + 1)
        sample = volume[window][present[window]]
        assert sample.size >= spec.min_periods
        spread = float(sample.std())
        expected = 0.0 if spread <= 1e-12 else (volume[index] - sample.mean()) / spread
        assert values[index] == pytest.approx(expected, rel=1e-12)
        assert available_at[index] == full.timestamps[index]
    assert not valid[30]
    assert valid[31:].any()


def test_built_dataset_with_native_gap_is_identical_across_range_ends() -> None:
    missing = {SYMBOLS[1]: {100, 101, 102, 103}}
    short = _build(
        BaseIntervalAggregatingTransport(HourlyTransport(missing)), days=10
    ).dataset
    long = _build(
        BaseIntervalAggregatingTransport(HourlyTransport(missing)), days=14
    ).dataset
    rows = short.n_bars
    assert np.array_equal(short.timestamps, long.timestamps[:rows])
    for name in (
        "features",
        "feature_available",
        "feature_staleness",
        "feature_staleness_hours",
        "tradable",
    ):
        assert np.array_equal(
            np.asarray(getattr(short, name)),
            np.asarray(getattr(long, name))[:rows],
        ), name


def test_sub_base_native_gap_is_identical_across_range_ends() -> None:
    """A 1h placeholder under a 4h base clock (incl. cross-asset features)."""

    def build(days: int) -> Any:
        return build_binance_market_dataset(
            market="usds-m",
            symbols=SYMBOLS,
            interval="4h",
            start_time=START,
            end_time=START + timedelta(days=days),
            transport_mode="vision",
            transport=BaseIntervalAggregatingTransport(
                HourlyTransport({SYMBOLS[1]: {101, 230}})
            ),
            tick_sizes=(0.01, 0.01),
            lot_sizes=(0.001, 0.001),
            minimum_notionals=(5.0, 5.0),
            listed_ats=(START - timedelta(days=100),) * 2,
            feature_timeframes=("1h",),
            gap_policy=BinanceGapPolicy.NON_TRADABLE_ROWS,
        ).dataset

    short, long = build(10), build(14)
    rows = short.n_bars
    assert "1h__rolling_beta_to_btc_24bar" in short.feature_names
    for name in (
        "features",
        "feature_available",
        "feature_staleness",
        "feature_staleness_hours",
    ):
        assert np.array_equal(
            np.asarray(getattr(short, name)),
            np.asarray(getattr(long, name))[:rows],
        ), name


def test_native_event_refuses_a_tradable_never_published_row() -> None:
    native = _gap_native(30, set())
    available_at = native.available_at.copy()
    available_at[3] = NEVER_AVAILABLE
    forged = RawMarketSeries(
        timestamps=native.timestamps,
        available_at=available_at,
        open=native.open,
        high=native.high,
        low=native.low,
        close=native.close,
        volume=native.volume,
        funding_rate=native.funding_rate,
        funding_available=native.funding_available,
        funding_event_count=native.funding_event_count,
        tradable=native.tradable,
    )
    spec = FeatureSpec("ema", FeatureKind.EMA_DISTANCE, lookback=5, timeframe="4h")
    contract = InstrumentContract("X", listed_at=START - timedelta(days=100))
    with pytest.raises(ValueError, match="never-published"):
        _native_events(spec, forged, contract)


def test_complete_range_dataset_is_identical_under_strict_and_gap_policy() -> None:
    strict = _build(
        BaseIntervalAggregatingTransport(HourlyTransport()),
        days=10,
        policy=BinanceGapPolicy.STRICT,
    ).dataset
    tolerant = _build(
        BaseIntervalAggregatingTransport(HourlyTransport()),
        days=10,
        policy=BinanceGapPolicy.NON_TRADABLE_ROWS,
    ).dataset
    for name in (
        "timestamps",
        "features",
        "feature_available",
        "feature_staleness",
        "feature_staleness_hours",
        "tradable",
    ):
        assert np.array_equal(
            np.asarray(getattr(strict, name)), np.asarray(getattr(tolerant, name))
        ), name


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
