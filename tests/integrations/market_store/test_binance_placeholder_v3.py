"""Synthetic row-clock oracles for a new, opt-in spot Vision policy."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.integrations.market_store.support import (
    HOUR_MS,
    NOW,
    FakeFetcher,
    MemoryStore,
    kline_csv_rows,
    populate_vision_cache,
    stored_exchange_info,
    write_frozen_exchange_info,
    zip_csv,
)
from trade_rl.data.build.economics import ExecutionEconomicsProfile
from trade_rl.integrations.binance.dataset import (
    BinanceGapPolicy,
    BinanceMarketDataSource,
    build_binance_market_dataset,
)
from trade_rl.integrations.binance.derived_klines import (
    BaseIntervalAggregatingTransport,
    aggregate_klines,
)
from trade_rl.integrations.binance.metadata import FrozenBinanceExchangeInfoTransport
from trade_rl.integrations.binance.transport import BinancePublicTransport
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    PeriodKind,
    StoreMarket,
    Universe,
    ms_from_utc,
)
from trade_rl.integrations.market_store.embargo import EmbargoPolicy
from trade_rl.integrations.market_store.export import (
    ECONOMICS_PROFILE_NAME,
    export_market_dataset,
)
from trade_rl.integrations.market_store.ingest import run_ingestion
from trade_rl.integrations.market_store.parse import (
    StoreKlineAdmission,
    parse_kline_archive,
)
from trade_rl.integrations.market_store.plan import plan_ingestion
from trade_rl.integrations.market_store.rules import parse_symbol_rules

POLICY = "strict_spot_placeholder_v3"
START = datetime(2020, 12, 1, tzinfo=UTC)
END = datetime(2021, 1, 1, tzinfo=UTC)
TARGET = 1608559200000
OTHER_GAPS = frozenset(
    (
        TARGET + HOUR_MS,
        TARGET + 2 * HOUR_MS,
        TARGET + 3 * HOUR_MS,
        ms_from_utc(datetime(2020, 12, 25, 2, tzinfo=UTC)),
    )
)
SYMBOL = "BTCUSDT"


def rows(*, placeholder: bool = True, gaps: bool = True) -> list[list[str]]:
    result = kline_csv_rows(
        SYMBOL,
        ms_from_utc(START),
        ms_from_utc(END),
        skip=OTHER_GAPS if gaps else frozenset(),
    )
    if placeholder:
        i = next(i for i, r in enumerate(result) if int(r[0]) == TARGET)
        result[i] = [
            str(TARGET),
            "100",
            "100",
            "100",
            "100",
            "0",
            str(TARGET - 1234),
            "0",
            "0",
            "0",
            "0",
            "0",
        ]
    return result


class RawTransport:
    def __init__(self, raw: list[list[object]]) -> None:
        self.raw = raw
        self.calls = 0

    def load_klines(self, **kwargs: object) -> tuple[list[list[object]], str]:
        self.calls += 1
        assert kwargs["interval"] == "1h"
        return self.raw, "synthetic"

    def load_funding_rates(
        self, **kwargs: object
    ) -> tuple[list[tuple[int, float]], str]:
        return [], "synthetic-spot"


def source(
    raw: list[list[str]],
    *,
    policy: str = POLICY,
    gap: str = "non_tradable_rows",
    symbol_market: str = "spot",
    mode: str = "vision",
) -> BinanceMarketDataSource:
    return BinanceMarketDataSource(
        market=symbol_market,
        interval="1h",
        start_time=START,
        end_time=END,
        transport_mode=mode,
        transport=RawTransport([list(r) for r in raw]),
        source_close_policy=policy,
        gap_policy=gap,
    )


def spec(symbol: str = SYMBOL) -> ArchiveSpec:
    url = "https://example.invalid/" + symbol + "-1h-2020-12.zip"
    return ArchiveSpec(
        ArchiveKind.KLINES,
        StoreMarket.SPOT,
        symbol,
        "1h",
        PeriodKind.MONTHLY,
        START,
        END,
        url,
        url + ".CHECKSUM",
    )


def test_new_policy_matches_store_native_row_clock_and_counts() -> None:
    raw = rows()
    parsed = parse_kline_archive(
        zip_csv(raw),
        spec(),
        embargo=EmbargoPolicy.enforced(),
        admission=StoreKlineAdmission.STRICT_NATIVE_V2,
    )
    s = source(raw)
    series = s.load(SYMBOL)
    expected = np.array(
        sorted(
            set(range(ms_from_utc(START), ms_from_utc(END), HOUR_MS))
            - OTHER_GAPS
            - {TARGET}
        ),
        dtype=np.int64,
    )
    actual = series.timestamps.astype("datetime64[ms]").astype("int64") - HOUR_MS
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, [r.open_time_ms for r in parsed.rows])
    assert len(parsed.placeholders) == 1
    assert s.gap_report[SYMBOL + ":1h"] == {
        "expected_bars": 744,
        "present_bars": 739,
        "missing_bars": 5,
        "funding_events_in_missing_bars": 0,
        "source_placeholders_skipped": 1,
    }
    assert series.funding_event_count.sum() == 0
    assert series.tradable.all()
    np.testing.assert_array_equal(series.available_at, series.timestamps)


def test_strict_refuses_new_placeholder_as_gap_without_changing_legacy() -> None:
    raw = rows(gaps=False)
    assert (
        len(
            source(raw, policy="reconstructed_v1", gap="strict").load(SYMBOL).timestamps
        )
        == 744
    )
    with pytest.raises(ValueError, match="source close_time"):
        source(raw, policy="strict_source_close_v2", gap="strict").load(SYMBOL)
    with pytest.raises(ValueError, match="complete and exactly regular"):
        source(raw, gap="strict").load(SYMBOL)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        (1, "101"),
        (5, "0.01"),
        (7, "1"),
        (8, "1"),
        (9, "1"),
        (10, "1"),
        (0, "+1608559200000"),
        (6, str((TARGET - 1234) * 1000)),
        (6, str(TARGET - HOUR_MS)),
        (6, "NaN"),
        (5, "NaN"),
    ],
)
def test_malformed_near_miss_is_not_silently_skipped(column: int, value: str) -> None:
    raw = rows()
    target = next(r for r in raw if int(r[0]) == TARGET)
    target[column] = value
    with pytest.raises(ValueError):
        source(raw).load(SYMBOL)


def test_unknown_symbol_and_duplicate_placeholder_fail_closed() -> None:
    raw = rows()
    with pytest.raises(ValueError):
        source(raw).load("UNKNOWNUSDT")
    i = next(i for i, row in enumerate(raw) if int(row[0]) == TARGET)
    raw.insert(i, list(raw[i]))
    with pytest.raises(ValueError, match="strictly increasing"):
        source(raw).load(SYMBOL)


@pytest.mark.parametrize(
    ("market", "mode"), [("usds-m", "vision"), ("spot", "auto"), ("spot", "rest")]
)
def test_v3_is_scoped_to_spot_vision(market: str, mode: str) -> None:
    with pytest.raises(ValueError, match="spot.*Vision"):
        source(rows(), symbol_market=market, mode=mode)


def test_new_dataset_identity_distinguishes_policy_and_same_admitted_rows() -> None:
    raw = rows()
    parsed = parse_kline_archive(
        zip_csv(raw),
        spec(),
        embargo=EmbargoPolicy.enforced(),
        admission=StoreKlineAdmission.STRICT_NATIVE_V2,
    )
    common = dict(
        market="spot",
        symbols=(SYMBOL,),
        interval="1h",
        start_time=START,
        end_time=END,
        transport_mode="vision",
        tick_sizes=(0.01,),
        lot_sizes=(0.001,),
        minimum_notionals=(5.0,),
        listed_ats=(START - timedelta(days=100),),
        gap_policy=BinanceGapPolicy.NON_TRADABLE_ROWS,
    )
    new = build_binance_market_dataset(
        **common,
        transport=RawTransport([list(r) for r in raw]),
        source_close_policy=POLICY,
    ).dataset
    store_rows = build_binance_market_dataset(
        **common, transport=RawTransport([r.vision_row() for r in parsed.rows])
    ).dataset
    assert new.dataset_id != store_rows.dataset_id
    assert new.recomputed_dataset_id() == new.dataset_id
    assert (
        json.loads(new.identity_payload_json)["metadata_evidence"][
            "binance_source_close_policy"
        ]
        == POLICY
    )
    for key, value in new.identity_arrays().items():
        np.testing.assert_array_equal(value, store_rows.identity_arrays()[key])
    missing = np.isin(
        new.timestamps,
        np.array(
            [np.datetime64(x + HOUR_MS, "ms") for x in sorted(OTHER_GAPS | {TARGET})]
        ),
    )
    assert missing.sum() == 5
    assert not new.tradable[missing].any()
    assert not new.resolved_array("information_available")[missing].any()
    assert not new.resolved_array("funding_event_count")[missing].any()
    # Default OHLCV features may carry the last observed value for up to
    # two hours, including age=2; a gap is never a fresh feature event.
    expected_ages = [1, 2, 3, 4, 1]
    assert new.feature_names == (
        "log_return_1bar",
        "log_return_1d",
        "realized_volatility_1d",
        "volume_zscore_1d",
        "funding_bps",
    )
    expected = np.array([[age <= 2] * 4 + [False] for age in expected_ages])
    np.testing.assert_array_equal(new.feature_available[missing, 0, :], expected)
    for row_index, age in zip(np.flatnonzero(missing), expected_ages, strict=True):
        if age <= 2:
            np.testing.assert_array_equal(
                new.features[row_index, 0, :4],
                new.features[row_index - age, 0, :4],
            )
    assert not new.resolved_array("funding_event_count").any()


def test_v3_refuses_known_aggregation_and_higher_timeframe() -> None:
    raw = rows()
    with pytest.raises(ValueError, match="not aggregation"):
        aggregate_klines(
            raw,
            base_interval="1h",
            interval="4h",
            start_ms=ms_from_utc(START),
            end_ms=ms_from_utc(END),
            source_close_policy=POLICY,
        )
    with pytest.raises(ValueError, match="not aggregation"):
        BaseIntervalAggregatingTransport(RawTransport([]), source_close_policy=POLICY)
    s = source(raw)
    with pytest.raises(ValueError, match="direct spot 1h"):
        s.load_timeframe(SYMBOL, "4h")
    assert s.transport.calls == 0
    s.load(SYMBOL)
    s.transport = BaseIntervalAggregatingTransport(RawTransport([]))
    with pytest.raises(ValueError, match="direct spot 1h"):
        s.load(SYMBOL)


@pytest.mark.parametrize("bad_header", [False, True])
def test_v3_native_vision_header_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    bad_header: bool,
) -> None:
    raw = rows()
    header = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    if bad_header:
        header[0] = "unknown_column"
    archive = zip_csv([header, *raw])
    transport = BinancePublicTransport(allow_network=False)
    monkeypatch.setattr(transport, "_request_bytes", lambda url: archive)
    s = BinanceMarketDataSource(
        market="spot",
        interval="1h",
        start_time=START,
        end_time=END,
        transport_mode="vision",
        transport=transport,
        source_close_policy=POLICY,
        gap_policy="non_tradable_rows",
    )
    if bad_header:
        with pytest.raises(ValueError):
            s.load(SYMBOL)
    else:
        assert len(s.load(SYMBOL).timestamps) == 739


def test_v3_complete_ordinary_grid_still_passes_strict() -> None:
    raw = rows(placeholder=False, gaps=False)
    v3 = source(raw, gap="strict").load(SYMBOL)
    v2 = source(raw, policy="strict_source_close_v2", gap="strict").load(SYMBOL)
    np.testing.assert_array_equal(v3.timestamps, v2.timestamps)
    np.testing.assert_array_equal(v3.close, v2.close)
    assert len(v3.timestamps) == 744


def test_v3_refuses_spot_funding_events(monkeypatch: pytest.MonkeyPatch) -> None:
    s = source(rows())
    monkeypatch.setattr(
        s.transport,
        "load_funding_rates",
        lambda **kwargs: ([(TARGET, 0.001)], "invalid-synthetic-spot"),
    )
    with pytest.raises(ValueError, match="requires no funding events"):
        s.load(SYMBOL)


@pytest.fixture(params=["memory", "postgres"])
def parity_store(request: pytest.FixtureRequest) -> Any:
    if request.param == "memory":
        return MemoryStore()
    return request.getfixturevalue("pg_store")


def test_checksum_ingest_export_and_file_v3_parity(
    parity_store: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise actual ingestion/export interfaces, not a reconstructed store row list."""
    store = parity_store
    market = StoreMarket.SPOT
    plan = plan_ingestion(
        Universe(symbols=(SYMBOL,), markets=(market,)), start=START, end=END, now=NOW
    )
    assert len(plan.specs) == 1
    archive = plan.specs[0]
    fetcher = FakeFetcher()
    fetcher.add(archive.url, zip_csv(rows()))
    report = run_ingestion(
        store,
        fetcher,
        plan,
        code_commit="synthetic-test",
        command_line="pytest",
        clock=lambda: NOW,
    )
    assert report.status == "succeeded"
    assert report.counts.archives_ingested == 1
    assert report.counts.rows_inserted == 739
    assert report.counts.placeholder_rows_skipped == 1
    assert len(report.skipped_placeholders) == 1
    assert report.skipped_placeholders[0].open_time_ms == TARGET
    assert store.source_record(archive.url) is not None
    persisted = store.read_klines(
        market, SYMBOL, "1h", ms_from_utc(START), ms_from_utc(END)
    )
    expected_opens = sorted(
        set(range(ms_from_utc(START), ms_from_utc(END), HOUR_MS))
        - OTHER_GAPS
        - {TARGET}
    )
    assert [r.open_time_ms for r in persisted] == expected_opens
    info = stored_exchange_info(market)
    if isinstance(store, MemoryStore):
        store.exchange_info.append(info)
    else:
        rules = parse_symbol_rules(info.raw_payload, market=market, symbols=(SYMBOL,))
        assert store.store_exchange_info(info, rules, run_id=None)
    exported = export_market_dataset(
        store,
        market=market,
        symbols=(SYMBOL,),
        interval="1h",
        start_time=START,
        end_time=END,
        feature_timeframes=(),
    )
    assert exported.evidence["klines"][SYMBOL]["rows"] == 739
    assert exported.evidence["funding_rates"][SYMBOL]["rows"] == 0
    cache = tmp_path / "vision-cache"
    populate_vision_cache(cache, fetcher, plan)
    write_frozen_exchange_info(tmp_path / "exchange-info", market)
    frozen = FrozenBinanceExchangeInfoTransport(tmp_path / "exchange-info")
    transport = BinancePublicTransport(cache_root=cache, allow_network=False)
    monkeypatch.setattr(
        transport, "load_exchange_information", frozen.load_exchange_information
    )
    file_result = build_binance_market_dataset(
        market=market.binance,
        symbols=(SYMBOL,),
        interval="1h",
        start_time=START,
        end_time=END,
        transport_mode="vision",
        transport=transport,
        feature_timeframes=(),
        source_close_policy=POLICY,
        gap_policy=BinanceGapPolicy.NON_TRADABLE_ROWS,
        metadata_evidence=exported.evidence,
        execution_economics=ExecutionEconomicsProfile(name=ECONOMICS_PROFILE_NAME),
    )
    assert file_result.gap_report[SYMBOL + ":1h"]["source_placeholders_skipped"] == 1
    assert file_result.dataset.dataset_id != exported.dataset.dataset_id
    assert file_result.dataset.recomputed_dataset_id() == file_result.dataset.dataset_id
    assert exported.dataset.recomputed_dataset_id() == exported.dataset.dataset_id
    for key, value in exported.dataset.identity_arrays().items():
        np.testing.assert_array_equal(file_result.dataset.identity_arrays()[key], value)
    gap = np.isin(
        exported.dataset.timestamps,
        [np.datetime64(x + HOUR_MS, "ms") for x in sorted(OTHER_GAPS | {TARGET})],
    )
    assert gap.sum() == 5
    assert not exported.dataset.tradable[gap].any()
    assert not exported.dataset.resolved_array("information_available")[gap].any()
    assert not exported.dataset.resolved_array("funding_event_count").any()
    assert not exported.dataset.resolved_array("funding_rate").any()
