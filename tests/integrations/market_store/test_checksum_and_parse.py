from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

from tests.integrations.market_store.support import (
    HOUR_MS,
    PLACEHOLDER_OPEN_MS,
    REAL_BTC_PLACEHOLDER,
    funding_csv_rows,
    kline_csv_rows,
    zip_csv,
)
from trade_rl.integrations.binance.dataset import (
    BinanceGapPolicy,
    BinanceMarketDataSource,
    _parse_kline_rows,
)
from trade_rl.integrations.market_store.checksum import (
    ChecksumError,
    parse_checksum,
    verify_archive,
)
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    FundingRow,
    KlineRow,
    PeriodKind,
    StoreMarket,
    funding_digest,
    klines_digest,
    ms_from_utc,
    utc_from_ms,
)
from trade_rl.integrations.market_store.embargo import (
    FS1_WARMUP_START,
    FS1_WARMUP_START_MS,
    SEALED_WINDOW_START,
    SEALED_WINDOW_START_MS,
    EmbargoPolicy,
    Fs1WarmupSpanError,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.parse import (
    ArchiveContentError,
    parse_funding_archive,
    parse_kline_archive,
)

JAN = datetime(2024, 1, 1, tzinfo=UTC)
FEB = datetime(2024, 2, 1, tzinfo=UTC)
JUL = datetime(2026, 7, 1, tzinfo=UTC)
OCT = datetime(2026, 10, 1, tzinfo=UTC)
ENFORCED = EmbargoPolicy.enforced()


def spec(
    kind: ArchiveKind = ArchiveKind.KLINES,
    *,
    start: datetime = JAN,
    end: datetime = FEB,
    market: StoreMarket = StoreMarket.SPOT,
    symbol: str = "BTCUSDT",
) -> ArchiveSpec:
    name = f"{symbol}-1h" if kind is ArchiveKind.KLINES else f"{symbol}-fundingRate"
    url = f"https://data.binance.vision/data/x/{name}-{start:%Y-%m}.zip"
    return ArchiveSpec(
        kind=kind,
        market=market,
        symbol=symbol,
        interval="1h" if kind is ArchiveKind.KLINES else None,
        period_kind=PeriodKind.MONTHLY,
        period_start=start,
        period_end=end,
        url=url,
        checksum_url=f"{url}.CHECKSUM",
    )


def klines(start: datetime = JAN, end: datetime = FEB, **kwargs: object):  # type: ignore[no-untyped-def]
    return kline_csv_rows(
        "BTCUSDT",
        ms_from_utc(start),
        ms_from_utc(end),
        **kwargs,  # type: ignore[arg-type]
    )


# -- checksum -------------------------------------------------------------------


def test_checksum_accepts_the_official_single_line_format() -> None:
    payload = b"archive-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    evidence = verify_archive(
        payload, f"{digest}  data.zip\n".encode(), filename="data.zip"
    )
    assert evidence.archive_sha256 == digest == evidence.checksum_line_sha256
    assert evidence.size_bytes == len(payload)
    assert (
        evidence.checksum_file_sha256
        == hashlib.sha256(f"{digest}  data.zip\n".encode()).hexdigest()
    )
    assert parse_checksum(f"{digest} *data.zip", filename="data.zip") == digest


@pytest.mark.parametrize(
    "checksum",
    [
        b"",
        b"not a checksum\n",
        b"0" * 64 + b"\n",
        b"0" * 64 + b"  other.zip\n",
        b"0" * 63 + b"  data.zip\n",
        ("A" * 64).encode() + b"  data.zip\n",
        b"0" * 64 + b"  data.zip\n" + b"1" * 64 + b"  data.zip\n",
    ],
)
def test_checksum_fails_closed_on_mismatch_or_malformed_file(checksum: bytes) -> None:
    with pytest.raises(ChecksumError):
        verify_archive(b"archive-bytes", checksum, filename="data.zip")


def test_checksum_rejects_an_empty_archive() -> None:
    with pytest.raises(ChecksumError, match="empty"):
        verify_archive(b"", b"0" * 64 + b"  data.zip\n", filename="data.zip")


# -- klines ---------------------------------------------------------------------


def test_klines_parse_to_exact_decimals_in_order() -> None:
    rows = parse_kline_archive(zip_csv(klines()), spec(), embargo=ENFORCED).rows
    assert len(rows) == 31 * 24
    assert rows[0].open_time_ms == ms_from_utc(JAN)
    assert rows[1].open_time_ms - rows[0].open_time_ms == HOUR_MS
    assert all(isinstance(value, Decimal) for value in (rows[0].open, rows[0].volume))
    text = klines()[0]
    assert rows[0].open == Decimal(text[1])
    assert format(rows[0].open, "f") == text[1]  # no float rounding, scale preserved
    assert rows[0].vision_row()[1] == text[1]


def test_decimal_values_a_float_would_corrupt_survive_exactly() -> None:
    assert float("12345678.12345678") != Decimal("12345678.12345678")
    assert 0.1 + 0.2 != 0.3
    row = KlineRow(
        open_time_ms=0,
        open=Decimal("12345678.12345678"),
        high=Decimal("99999999.99999999"),
        low=Decimal("0.00000001"),
        close=Decimal("0.1"),
        volume=Decimal("0.00000001"),
        close_time_ms=1,
        quote_volume=Decimal("0.3"),
        trades=1,
        taker_buy_base_volume=Decimal("0"),
        taker_buy_quote_volume=Decimal("0"),
    )
    vision = row.vision_row()
    assert vision[1] == "12345678.12345678"
    assert vision[2] == "99999999.99999999"
    assert vision[3] == "0.00000001"
    assert Decimal("0.1") + Decimal("0.2") == Decimal("0.3")  # exact, unlike float


def test_trailing_zeros_do_not_change_value_or_digest() -> None:
    a = KlineRow(
        0,
        *(Decimal("1.50"),) * 4,
        Decimal("2.0"),
        1,
        Decimal("3.00"),
        1,
        *(Decimal("0"),) * 2,
    )
    b = KlineRow(
        0,
        *(Decimal("1.5"),) * 4,
        Decimal("2"),
        1,
        Decimal("3"),
        1,
        *(Decimal("0.000"),) * 2,
    )
    assert a.comparable() == b.comparable()
    assert klines_digest([a]) == klines_digest([b])
    assert funding_digest([FundingRow(1, Decimal("0.0001000"))]) == funding_digest(
        [FundingRow(1, Decimal("0.0001"))]
    )


def test_header_row_and_microsecond_spot_timestamps_are_normalised() -> None:
    plain = parse_kline_archive(zip_csv(klines()), spec(), embargo=ENFORCED).rows
    headered = parse_kline_archive(
        zip_csv(klines(header=True, micro=True)), spec(), embargo=ENFORCED
    ).rows
    assert headered == plain


def test_rows_outside_the_archive_period_are_refused() -> None:
    wide = klines(JAN, datetime(2024, 2, 2, tzinfo=UTC))
    with pytest.raises(ArchiveContentError, match="outside the archive period"):
        parse_kline_archive(zip_csv(wide), spec(), embargo=ENFORCED)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.__setitem__(2, "0.00000001"),  # high below open/close
        lambda row: row.__setitem__(3, "999999999.0"),  # low above open/close
        lambda row: row.__setitem__(5, "-1"),  # negative volume
        lambda row: row.__setitem__(1, "0"),  # non-positive price
        lambda row: row.__setitem__(1, "nan"),
        lambda row: row.__setitem__(1, "abc"),
    ],
)
def test_invalid_klines_are_refused(mutate) -> None:  # type: ignore[no-untyped-def]
    rows = klines()
    mutate(rows[3])
    with pytest.raises(ArchiveContentError):
        parse_kline_archive(zip_csv(rows), spec(), embargo=ENFORCED)


def test_duplicate_or_unordered_rows_are_refused() -> None:
    rows = klines()
    rows[5] = list(rows[4])
    with pytest.raises(ArchiveContentError, match="increasing"):
        parse_kline_archive(zip_csv(rows), spec(), embargo=ENFORCED)


def test_short_and_corrupt_archives_are_refused() -> None:
    with pytest.raises(ArchiveContentError):
        parse_kline_archive(b"not a zip", spec(), embargo=ENFORCED)
    with pytest.raises(ArchiveContentError, match="short"):
        parse_kline_archive(
            zip_csv([row[:5] for row in klines()]), spec(), embargo=ENFORCED
        )


def test_gaps_are_preserved_not_filled() -> None:
    gap = ms_from_utc(JAN) + 10 * HOUR_MS
    rows = parse_kline_archive(
        zip_csv(klines(skip=frozenset({gap}))), spec(), embargo=ENFORCED
    ).rows
    assert len(rows) == 31 * 24 - 1
    assert gap not in {row.open_time_ms for row in rows}


# -- funding --------------------------------------------------------------------


@pytest.mark.parametrize("header", ["calc_time", "fundingTime"])
def test_funding_headers_parse_exactly(header: str) -> None:
    funding_spec = spec(ArchiveKind.FUNDING_RATE, market=StoreMarket.USDS_M)
    payload = zip_csv(
        funding_csv_rows(ms_from_utc(JAN), ms_from_utc(FEB), header=header)
    )
    rows = parse_funding_archive(payload, funding_spec, embargo=ENFORCED)
    assert len(rows) == 93
    assert rows[0].funding_time_ms == ms_from_utc(JAN) + 5
    assert isinstance(rows[0].rate, Decimal)
    assert (rows[0].mark_price is not None) == (header == "fundingTime")
    assert (rows[0].interval_hours == 8) == (header == "calc_time")


def test_funding_refuses_missing_header_and_implausible_rates() -> None:
    funding_spec = spec(ArchiveKind.FUNDING_RATE, market=StoreMarket.USDS_M)
    with pytest.raises(ArchiveContentError, match="header"):
        parse_funding_archive(
            zip_csv([["1704067200005", "8", "0.0001"]]), funding_spec, embargo=ENFORCED
        )
    rows = funding_csv_rows(ms_from_utc(JAN), ms_from_utc(FEB))
    rows[1][2] = "1.5"
    with pytest.raises(ArchiveContentError, match="implausible"):
        parse_funding_archive(zip_csv(rows), funding_spec, embargo=ENFORCED)


# -- embargo inside the parser ---------------------------------------------------


WARMUP = EmbargoPolicy.with_warmup(
    "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7"
)


def test_a_kline_in_the_sealed_window_is_refused_under_every_policy() -> None:
    november = spec(start=SEALED_WINDOW_START, end=datetime(2026, 12, 1, tzinfo=UTC))
    payload = zip_csv(klines(SEALED_WINDOW_START, datetime(2026, 11, 2, tzinfo=UTC)))
    for policy in (ENFORCED, WARMUP):
        with pytest.raises(SealedWindowEmbargoError, match="sealed window"):
            parse_kline_archive(payload, november, embargo=policy)


def test_a_warmup_kline_needs_the_authorization() -> None:
    august = spec(start=FS1_WARMUP_START, end=datetime(2026, 9, 1, tzinfo=UTC))
    payload = zip_csv(klines(FS1_WARMUP_START, datetime(2026, 9, 1, tzinfo=UTC)))
    with pytest.raises(Fs1WarmupSpanError, match="warm-up"):
        parse_kline_archive(payload, august, embargo=ENFORCED)
    assert parse_kline_archive(payload, august, embargo=WARMUP).rows


def test_one_closed_span_row_poisons_an_otherwise_valid_archive() -> None:
    july = spec(start=JUL, end=datetime(2026, 8, 2, tzinfo=UTC))
    payload = zip_csv(klines(JUL, datetime(2026, 8, 1, 1, tzinfo=UTC)))
    with pytest.raises(Fs1WarmupSpanError):
        parse_kline_archive(payload, july, embargo=ENFORCED)
    just_before = zip_csv(klines(JUL, FS1_WARMUP_START))
    rows = parse_kline_archive(
        just_before, spec(start=JUL, end=FS1_WARMUP_START), embargo=ENFORCED
    ).rows
    assert rows[-1].open_time_ms == FS1_WARMUP_START_MS - HOUR_MS
    assert rows[-1].close_time_ms == FS1_WARMUP_START_MS - 1
    october = spec(start=OCT, end=datetime(2026, 11, 2, tzinfo=UTC))
    late = zip_csv(klines(OCT, datetime(2026, 11, 1, 1, tzinfo=UTC)))
    with pytest.raises(SealedWindowEmbargoError):
        parse_kline_archive(late, october, embargo=WARMUP)


def test_a_funding_row_at_the_boundary_is_refused() -> None:
    november = spec(
        ArchiveKind.FUNDING_RATE,
        start=SEALED_WINDOW_START,
        end=datetime(2026, 12, 1, tzinfo=UTC),
        market=StoreMarket.USDS_M,
    )
    payload = zip_csv(
        funding_csv_rows(
            SEALED_WINDOW_START_MS - 8 * HOUR_MS, SEALED_WINDOW_START_MS + 8 * HOUR_MS
        )
    )
    # the pre-boundary row also lies outside the period; the embargo check runs first
    # on whichever row it meets, and the sealed row can never be returned
    with pytest.raises((SealedWindowEmbargoError, ArchiveContentError)):
        parse_funding_archive(payload, november, embargo=ENFORCED)
    sealed_only = zip_csv(
        [
            ["calc_time", "funding_interval_hours", "last_funding_rate"],
            [str(SEALED_WINDOW_START_MS), "8", "0.0001"],
        ]
    )
    with pytest.raises(SealedWindowEmbargoError):
        parse_funding_archive(sealed_only, november, embargo=ENFORCED)


def test_time_helpers_round_trip_exactly() -> None:
    for value in (0, 1, 999, 1_704_067_200_123, SEALED_WINDOW_START_MS - 1):
        assert ms_from_utc(utc_from_ms(value)) == value


# -- the documented 2020-12-21 spot placeholder candle ---------------------------

DEC_2020 = datetime(2020, 12, 1, tzinfo=UTC)
JAN_2021 = datetime(2021, 1, 1, tzinfo=UTC)


def dec_2020_rows(replacement: list[str] | None = None) -> list[list[str]]:
    rows = klines(DEC_2020, JAN_2021)
    index = next(i for i, row in enumerate(rows) if int(row[0]) == PLACEHOLDER_OPEN_MS)
    rows[index] = REAL_BTC_PLACEHOLDER if replacement is None else replacement
    return rows


def test_the_documented_spot_placeholder_is_skipped_and_reported() -> None:
    parsed = parse_kline_archive(
        zip_csv(dec_2020_rows()), spec(start=DEC_2020, end=JAN_2021), embargo=ENFORCED
    )
    assert len(parsed.rows) == 31 * 24 - 1
    assert PLACEHOLDER_OPEN_MS not in {row.open_time_ms for row in parsed.rows}
    (placeholder,) = parsed.placeholders
    assert placeholder.open_time_ms == PLACEHOLDER_OPEN_MS
    assert placeholder.close_time_ms == 1608558440521
    assert placeholder.price == Decimal("22646.53")
    assert placeholder.to_dict()["price"] == "22646.53000000"
    # the neighbours are untouched, so the gap is visible rather than filled
    assert {row.open_time_ms for row in parsed.rows} >= {
        PLACEHOLDER_OPEN_MS - HOUR_MS,
        PLACEHOLDER_OPEN_MS + HOUR_MS,
    }


@pytest.mark.parametrize(
    "symbol",
    ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT"),
)
def test_placeholder_scope_is_limited_to_verified_symbol_roster(symbol: str) -> None:
    parsed = parse_kline_archive(
        zip_csv(dec_2020_rows()),
        spec(start=DEC_2020, end=JAN_2021, symbol=symbol),
        embargo=ENFORCED,
    )
    assert len(parsed.placeholders) == 1


def test_file_backed_reader_currently_retains_the_store_skipped_placeholder() -> None:
    """Characterize the existing difference, without changing source data or readers."""
    raw_rows = dec_2020_rows()
    assert int(REAL_BTC_PLACEHOLDER[6]) < PLACEHOLDER_OPEN_MS

    timestamps, _opens, _highs, _lows, closes, quote_volumes = _parse_kline_rows(
        raw_rows,
        interval_ms=HOUR_MS,
        start_ms=ms_from_utc(DEC_2020),
        end_ms=ms_from_utc(JAN_2021),
    )
    # This separate reader infers each close time from open+interval. It does
    # not validate the archive's anomalous source close_time field.
    close_times_ms = timestamps.astype("datetime64[ms]").astype("int64")
    matches = [
        i
        for i, close_ms in enumerate(close_times_ms)
        if int(close_ms) == PLACEHOLDER_OPEN_MS + HOUR_MS
    ]
    assert len(matches) == 1
    assert len(timestamps) == 31 * 24
    assert closes[matches[0]] == float(REAL_BTC_PLACEHOLDER[4])
    assert quote_volumes[matches[0]] == 0.0

    store = parse_kline_archive(
        zip_csv(raw_rows), spec(start=DEC_2020, end=JAN_2021), embargo=ENFORCED
    )
    assert len(store.rows) == len(timestamps) - 1
    assert PLACEHOLDER_OPEN_MS not in {row.open_time_ms for row in store.rows}
    assert len(store.placeholders) == 1


@pytest.mark.parametrize(
    "policy",
    (BinanceGapPolicy.STRICT, BinanceGapPolicy.NON_TRADABLE_ROWS),
)
def test_file_backed_gap_policies_do_not_mask_store_skipped_placeholder(
    policy: BinanceGapPolicy,
) -> None:
    """Pin existing source-clock divergence without relabeling historical Datasets."""
    raw_rows = dec_2020_rows()
    start_ms = ms_from_utc(DEC_2020)
    end_ms = ms_from_utc(JAN_2021)

    class OfflineTransport:
        def load_klines(self, **kwargs: object) -> tuple[list[list[str]], str]:
            assert kwargs["symbol"] == "BTCUSDT"
            assert kwargs["interval"] == "1h"
            assert kwargs["start_ms"] == start_ms
            assert kwargs["end_ms"] == end_ms
            return raw_rows, "synthetic-neighbours-real-placeholder"

        def load_funding_rates(self, **kwargs: object) -> tuple[list[tuple[int, float]], str]:
            assert kwargs["symbol"] == "BTCUSDT"
            return [], "no-funding-spot-fixture"

    source = BinanceMarketDataSource(
        market="spot",
        interval="1h",
        start_time=DEC_2020,
        end_time=JAN_2021,
        transport_mode="vision",
        transport=OfflineTransport(),
        gap_policy=policy,
    )
    series = source.load("BTCUSDT")
    index = (PLACEHOLDER_OPEN_MS - start_ms) // HOUR_MS
    expected_close = np.datetime64(PLACEHOLDER_OPEN_MS + HOUR_MS, "ms")

    assert len(series.timestamps) == 31 * 24
    assert series.timestamps[index] == expected_close
    assert series.available_at[index] == expected_close
    assert bool(series.tradable[index])
    assert series.volume[index] == 0.0
    assert series.funding_rate[index] == 0.0
    if policy is BinanceGapPolicy.NON_TRADABLE_ROWS:
        assert source.gap_report == {
            "BTCUSDT:1h": {
                "expected_bars": 31 * 24,
                "present_bars": 31 * 24,
                "missing_bars": 0,
                "funding_events_in_missing_bars": 0,
            }
        }
    else:
        assert source.gap_report == {}

    # Reuse the exact same constructed rows (including the authentic
    # checksum-backed anomalous candle) through the separate store parser.
    stored = parse_kline_archive(
        zip_csv(raw_rows),
        spec(start=DEC_2020, end=JAN_2021),
        embargo=ENFORCED,
    )
    assert len(stored.rows) == len(series.timestamps) - 1
    assert len(stored.placeholders) == 1
    assert stored.placeholders[0].open_time_ms == PLACEHOLDER_OPEN_MS
    assert PLACEHOLDER_OPEN_MS not in {row.open_time_ms for row in stored.rows}

def test_an_ordinary_archive_has_no_placeholders() -> None:
    assert (
        parse_kline_archive(zip_csv(klines()), spec(), embargo=ENFORCED).placeholders
        == ()
    )


def with_columns(**changes: str) -> list[str]:
    names = {
        "open_ms": 0,
        "open": 1,
        "high": 2,
        "low": 3,
        "close": 4,
        "volume": 5,
        "close_ms": 6,
        "quote": 7,
        "trades": 8,
        "taker_base": 9,
        "taker_quote": 10,
    }
    row = list(REAL_BTC_PLACEHOLDER)
    for name, value in changes.items():
        row[names[name]] = value
    return row


@pytest.mark.parametrize(
    "row",
    [
        with_columns(volume="0.00000001"),  # any nonzero volume
        with_columns(quote="0.00000001"),
        with_columns(taker_base="0.00000001"),
        with_columns(taker_quote="0.00000001"),
        with_columns(trades="1"),  # any trade
        with_columns(high="22646.54000000"),  # not flat
        with_columns(low="22646.52000000"),
        with_columns(close="22646.52000000"),
        with_columns(
            open="0.00000000",
            high="0.00000000",
            low="0.00000000",
            close="0.00000000",
        ),  # flat but non-positive price
        with_columns(close_ms=str(PLACEHOLDER_OPEN_MS)),  # close == open
        with_columns(close_ms=str(PLACEHOLDER_OPEN_MS - HOUR_MS)),  # a whole bar early
        with_columns(close_ms=str(PLACEHOLDER_OPEN_MS - 2 * HOUR_MS)),
    ],
)
def test_near_miss_placeholder_shapes_are_still_refused(row: list[str]) -> None:
    with pytest.raises(ArchiveContentError):
        parse_kline_archive(
            zip_csv(dec_2020_rows(row)),
            spec(start=DEC_2020, end=JAN_2021),
            embargo=ENFORCED,
        )


def test_the_placeholder_shape_at_any_other_bar_is_refused() -> None:
    next_hour = PLACEHOLDER_OPEN_MS + HOUR_MS
    shifted = with_columns(
        open_ms=str(next_hour), close_ms=str(next_hour - 12 * 60 * 1000)
    )
    rows = dec_2020_rows()
    index = next(i for i, row in enumerate(rows) if int(row[0]) == next_hour)
    rows[index] = shifted
    with pytest.raises(ArchiveContentError, match="close_time not after open"):
        parse_kline_archive(
            zip_csv(rows), spec(start=DEC_2020, end=JAN_2021), embargo=ENFORCED
        )


def test_the_placeholder_is_only_recognised_in_the_spot_market() -> None:
    futures = spec(start=DEC_2020, end=JAN_2021, market=StoreMarket.USDS_M)
    with pytest.raises(ArchiveContentError, match="close_time not after open"):
        parse_kline_archive(zip_csv(dec_2020_rows()), futures, embargo=ENFORCED)


def test_the_placeholder_is_only_recognised_for_verified_symbols() -> None:
    unverified = spec(start=DEC_2020, end=JAN_2021, symbol="DOGEUSDT")
    with pytest.raises(ArchiveContentError, match="close_time not after open"):
        parse_kline_archive(zip_csv(dec_2020_rows()), unverified, embargo=ENFORCED)


def test_a_zero_volume_row_with_a_valid_close_time_is_stored_not_skipped() -> None:
    # e.g. the 2021-02-11 03:00 UTC spot rows: zero volume but close after open. These
    # pass the unchanged validation and are NOT placeholders in the skip sense.
    row = with_columns(close_ms=str(PLACEHOLDER_OPEN_MS + 40 * 60 * 1000))
    parsed = parse_kline_archive(
        zip_csv(dec_2020_rows(row)),
        spec(start=DEC_2020, end=JAN_2021),
        embargo=ENFORCED,
    )
    assert parsed.placeholders == ()
    assert len(parsed.rows) == 31 * 24
