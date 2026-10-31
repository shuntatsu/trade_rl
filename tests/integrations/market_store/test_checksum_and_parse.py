from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tests.integrations.market_store.support import (
    HOUR_MS,
    funding_csv_rows,
    kline_csv_rows,
    zip_csv,
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
) -> ArchiveSpec:
    name = "BTCUSDT-1h" if kind is ArchiveKind.KLINES else "BTCUSDT-fundingRate"
    url = f"https://data.binance.vision/data/x/{name}-{start:%Y-%m}.zip"
    return ArchiveSpec(
        kind=kind,
        market=market,
        symbol="BTCUSDT",
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
    rows = parse_kline_archive(zip_csv(klines()), spec(), embargo=ENFORCED)
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
    plain = parse_kline_archive(zip_csv(klines()), spec(), embargo=ENFORCED)
    headered = parse_kline_archive(
        zip_csv(klines(header=True, micro=True)), spec(), embargo=ENFORCED
    )
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
    )
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
    assert parse_kline_archive(payload, august, embargo=WARMUP)


def test_one_closed_span_row_poisons_an_otherwise_valid_archive() -> None:
    july = spec(start=JUL, end=datetime(2026, 8, 2, tzinfo=UTC))
    payload = zip_csv(klines(JUL, datetime(2026, 8, 1, 1, tzinfo=UTC)))
    with pytest.raises(Fs1WarmupSpanError):
        parse_kline_archive(payload, july, embargo=ENFORCED)
    just_before = zip_csv(klines(JUL, FS1_WARMUP_START))
    rows = parse_kline_archive(
        just_before, spec(start=JUL, end=FS1_WARMUP_START), embargo=ENFORCED
    )
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
