"""Opt-in USD-M daily ``metrics`` and ``bookDepth`` archives: no network, no database.

The literal fixtures below are excerpts of the real public archives
``BTCUSDT-metrics-2024-03-15.zip`` and ``BTCUSDT-bookDepth-2024-03-15.zip`` (observed
layout: a header row, ``YYYY-MM-DD HH:MM:SS`` UTC timestamps, plain decimals, rows not
time-ordered in the metrics file). Everything else is synthetic.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from tests.integrations.market_store.support import (
    NOW,
    FakeFetcher,
    MemoryStore,
    archive_bytes,
    zip_csv,
)
from trade_rl.integrations.market_store.archive_url import (
    UnrecognizedArchiveUrlError,
    check_url_fetchable,
    parse_archive_url,
    verify_spec_matches_url,
    vision_daily_derivatives_url,
)
from trade_rl.integrations.market_store.cli import (
    EXIT_EMBARGO,
    EXIT_FAILURE,
    EXIT_OK,
    run,
)
from trade_rl.integrations.market_store.contracts import (
    DEFAULT_ARCHIVE_KINDS,
    ArchiveKind,
    ArchiveSpec,
    BookDepthRow,
    OpenInterestMetricRow,
    PeriodKind,
    StoreMarket,
    Universe,
    ms_from_utc,
)
from trade_rl.integrations.market_store.embargo import (
    EmbargoPolicy,
    Fs1WarmupSpanError,
    SealedWindowEmbargoError,
    checked_read_range,
)
from trade_rl.integrations.market_store.ingest import (
    ConflictError,
    IngestReport,
    run_ingestion,
)
from trade_rl.integrations.market_store.parse import (
    ArchiveContentError,
    parse_book_depth_archive,
    parse_open_interest_metrics_archive,
)
from trade_rl.integrations.market_store.plan import IngestionPlan, plan_ingestion

AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-77"
ENFORCED = EmbargoPolicy.enforced()
WARMUP = EmbargoPolicy.with_warmup(AUTH)
ROOT = "https://data.binance.vision/data"
DAY = datetime(2024, 3, 15, tzinfo=UTC)
NEXT_DAY = DAY + timedelta(days=1)
METRICS_URL = f"{ROOT}/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-03-15.zip"
DEPTH_URL = (
    f"{ROOT}/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-03-15.zip"
)
BOTH = (
    ArchiveKind.KLINES,
    ArchiveKind.FUNDING_RATE,
    ArchiveKind.METRICS,
    ArchiveKind.BOOK_DEPTH,
)

METRICS_HEADER = (
    "create_time,symbol,sum_open_interest,sum_open_interest_value,"
    "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
    "count_long_short_ratio,sum_taker_long_short_vol_ratio"
)
# the real file is not time-ordered: 00:05, 00:10 and then 00:00 appear in this order
REAL_METRICS = [
    "2024-03-15 00:05:00,BTCUSDT,84043.0430000000000000,6016044742.7647000000000000,1.76479256,1.53693400,1.61288163,2.17419200",
    "2024-03-15 00:10:00,BTCUSDT,84042.4920000000000000,6009744134.9328000000000000,1.76282825,1.53769600,1.61028785,0.80700400",
    "2024-03-15 00:00:00,BTCUSDT,83981.9070000000000000,6000778051.8311230000000000,1.76875308,1.53766300,1.61358306,1.09685300",
]
DEPTH_HEADER = "timestamp,percentage,depth,notional"
REAL_DEPTH = [
    "2024-03-15 00:00:09,-5,5828.70100000,406718563.05162000",
    "2024-03-15 00:00:09,-4,4374.75000000,307549180.86323000",
    "2024-03-15 00:00:09,-3,3716.98700000,262208834.19452000",
    "2024-03-15 00:00:09,-2,2980.30000000,210866529.10349000",
    "2024-03-15 00:00:09,-1,1419.62000000,100965678.22336000",
    "2024-03-15 00:00:09,1,2008.09700000,144155174.84886000",
    "2024-03-15 00:00:09,2,4291.54400000,309733090.46150000",
    "2024-03-15 00:00:09,3,5455.04500000,394952641.82888000",
    "2024-03-15 00:00:09,4,6843.25300000,497710155.71522000",
    "2024-03-15 00:00:09,5,7547.90000000,550345082.89471000",
    "2024-03-15 00:00:38,-5,5826.48800000,406922894.58542000",
    "2024-03-15 00:00:38,5,7602.55500000,554324613.59821000",
]


def derivatives_spec(
    kind: ArchiveKind,
    day: datetime = DAY,
    *,
    symbol: str = "BTCUSDT",
    market: StoreMarket = StoreMarket.USDS_M,
) -> ArchiveSpec:
    url = vision_daily_derivatives_url(kind, StoreMarket.USDS_M, symbol, day)
    return ArchiveSpec(
        kind=kind,
        market=market,
        symbol=symbol,
        interval=None,
        period_kind=PeriodKind.DAILY,
        period_start=day,
        period_end=day + timedelta(days=1),
        url=url,
        checksum_url=f"{url}.CHECKSUM",
    )


def text_zip(lines: Sequence[str], *, member: str = "data.csv") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, "\n".join(lines) + "\n")
    return buffer.getvalue()


def metrics_payload(*rows: str, header: str = METRICS_HEADER) -> bytes:
    return text_zip([header, *rows])


def depth_payload(*rows: str, header: str = DEPTH_HEADER) -> bytes:
    return text_zip([header, *rows])


# -- URLs -------------------------------------------------------------------------


def test_builder_and_parser_agree_for_both_kinds() -> None:
    assert (
        vision_daily_derivatives_url(
            ArchiveKind.METRICS, StoreMarket.USDS_M, "BTCUSDT", DAY
        )
        == METRICS_URL
    )
    assert (
        vision_daily_derivatives_url(
            ArchiveKind.BOOK_DEPTH, StoreMarket.USDS_M, "BTCUSDT", DAY
        )
        == DEPTH_URL
    )
    for url, kind in (
        (METRICS_URL, ArchiveKind.METRICS),
        (DEPTH_URL, ArchiveKind.BOOK_DEPTH),
    ):
        for candidate, checksum in ((url, False), (f"{url}.CHECKSUM", True)):
            parsed = parse_archive_url(candidate)
            assert parsed.kind is kind and parsed.is_checksum is checksum
            assert parsed.market is StoreMarket.USDS_M and parsed.interval is None
            assert parsed.period_kind is PeriodKind.DAILY
            assert (parsed.period_start, parsed.period_end) == (DAY, NEXT_DAY)
            assert parsed.symbol == "BTCUSDT"
        verify_spec_matches_url(derivatives_spec(kind))


def test_the_builder_refuses_spot_monthly_and_foreign_kinds() -> None:
    for kind in (ArchiveKind.METRICS, ArchiveKind.BOOK_DEPTH):
        with pytest.raises(ValueError, match="only for USD-M"):
            vision_daily_derivatives_url(kind, StoreMarket.SPOT, "BTCUSDT", DAY)
        with pytest.raises(ValueError, match="only as daily"):
            vision_daily_derivatives_url(
                kind,
                StoreMarket.USDS_M,
                "BTCUSDT",
                DAY,
                period_kind=PeriodKind.MONTHLY,
            )
        with pytest.raises(ValueError, match="invalid symbol"):
            vision_daily_derivatives_url(kind, StoreMarket.USDS_M, "btc/usdt", DAY)
        with pytest.raises(ValueError, match="timezone-aware"):
            vision_daily_derivatives_url(
                kind, StoreMarket.USDS_M, "BTCUSDT", datetime(2024, 3, 15)
            )
    with pytest.raises(ValueError, match="not a daily USD-M"):
        vision_daily_derivatives_url(
            ArchiveKind.KLINES, StoreMarket.USDS_M, "BTCUSDT", DAY
        )


@pytest.mark.parametrize(
    ("url", "message"),
    [
        (
            f"{ROOT}/spot/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-03-15.zip",
            "USD-M daily",
        ),
        (
            f"{ROOT}/spot/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-03-15.zip",
            "USD-M daily",
        ),
        (
            f"{ROOT}/futures/um/monthly/metrics/BTCUSDT/BTCUSDT-metrics-2024-03.zip",
            "futures/um/monthly",
        ),
        (
            f"{ROOT}/futures/um/monthly/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-03.zip",
            "futures/um/monthly",
        ),
        (
            f"{ROOT}/futures/cm/daily/metrics/BTCUSD_PERP/BTCUSD_PERP-metrics-2024-03-15.zip",
            "futures/cm/daily",
        ),
        (
            f"{ROOT}/futures/um/daily/metrics/BTCUSDT/ETHUSDT-metrics-2024-03-15.zip",
            "disagree",
        ),
        (
            f"{ROOT}/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-02-30.zip",
            "invalid archive period",
        ),
        (
            f"{ROOT}/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-03.zip",
            "unrecognised",
        ),
        (
            f"{ROOT}/futures/um/daily/metrics/BTCUSDT/BTCUSDT-bookDepth-2024-03-15.zip",
            "unrecognised",
        ),
        (
            f"{ROOT}/futures/um/daily/metrics/btcusdt/btcusdt-metrics-2024-03-15.zip",
            "unrecognised",
        ),
    ],
)
def test_unrecognised_or_mislabelled_urls_are_refused_with_a_clear_error(
    url: str, message: str
) -> None:
    with pytest.raises(UnrecognizedArchiveUrlError, match=message):
        parse_archive_url(url)


@pytest.mark.parametrize("kind", [ArchiveKind.METRICS, ArchiveKind.BOOK_DEPTH])
def test_archive_spec_refuses_spot_monthly_and_an_interval(kind: ArchiveKind) -> None:
    good = derivatives_spec(kind)
    fields: dict[str, Any] = {
        "kind": kind,
        "market": StoreMarket.USDS_M,
        "symbol": "BTCUSDT",
        "interval": None,
        "period_kind": PeriodKind.DAILY,
        "period_start": good.period_start,
        "period_end": good.period_end,
        "url": good.url,
        "checksum_url": good.checksum_url,
    }
    with pytest.raises(ValueError, match="only for USD-M"):
        ArchiveSpec(**{**fields, "market": StoreMarket.SPOT})
    with pytest.raises(ValueError, match="only as daily"):
        ArchiveSpec(
            **{
                **fields,
                "period_kind": PeriodKind.MONTHLY,
                "period_end": datetime(2024, 4, 1, tzinfo=UTC),
            }
        )
    with pytest.raises(ValueError, match="interval"):
        ArchiveSpec(**{**fields, "interval": "1h"})


# -- planning -----------------------------------------------------------------------


def plan_for(
    kinds: Sequence[ArchiveKind] | None = None,
    *,
    start: datetime = DAY,
    end: datetime = DAY + timedelta(days=3),
    markets: tuple[StoreMarket, ...] = (StoreMarket.USDS_M, StoreMarket.SPOT),
    embargo: EmbargoPolicy | None = None,
) -> IngestionPlan:
    extra: dict[str, Any] = {} if kinds is None else {"kinds": kinds}
    return plan_ingestion(
        Universe(symbols=("BTCUSDT", "ETHUSDT"), markets=markets),
        start=start,
        end=end,
        now=NOW,
        embargo=embargo,
        **extra,
    )


def test_the_default_plan_is_unchanged_by_the_opt_in_kinds() -> None:
    default = plan_for()
    explicit = plan_for(DEFAULT_ARCHIVE_KINDS)
    assert default.specs == explicit.specs
    assert default.to_dict() == explicit.to_dict()
    assert "kinds" not in default.to_dict()
    assert {spec.kind for spec in default.specs} == {ArchiveKind.KLINES}


def test_the_opt_in_adds_daily_usds_m_archives_after_the_default_ones() -> None:
    default = plan_for()
    plan = plan_for(BOTH)
    old_kinds = (ArchiveKind.KLINES, ArchiveKind.FUNDING_RATE)
    # the default archives keep their relative order inside the larger plan
    assert [s for s in plan.specs if s.kind in old_kinds] == list(default.specs)
    extra = [
        s for s in plan.specs if s.kind in (ArchiveKind.METRICS, ArchiveKind.BOOK_DEPTH)
    ]
    # 2 symbols x 3 days x 2 kinds, none for spot
    assert len(extra) == 12
    assert all(
        spec.market is StoreMarket.USDS_M
        and spec.period_kind is PeriodKind.DAILY
        and spec.interval is None
        and spec.period_end - spec.period_start == timedelta(days=1)
        for spec in extra
    )
    assert [
        s.period_start
        for s in extra
        if s.symbol == "BTCUSDT" and s.kind is ArchiveKind.METRICS
    ] == [
        DAY,
        DAY + timedelta(days=1),
        DAY + timedelta(days=2),
    ]
    assert plan.to_dict()["kinds"] == [
        "klines",
        "funding_rate",
        "metrics",
        "book_depth",
    ]
    for spec in extra:
        verify_spec_matches_url(spec)


def test_only_the_requested_kinds_are_planned() -> None:
    plan = plan_for((ArchiveKind.METRICS,), markets=(StoreMarket.USDS_M,))
    assert {s.kind for s in plan.specs} == {ArchiveKind.METRICS} and len(
        plan.specs
    ) == 6
    assert plan.skipped == ()
    assert plan.to_dict()["kinds"] == ["metrics"]


def test_opt_in_kinds_need_a_usds_m_market_and_a_clean_kind_list() -> None:
    with pytest.raises(ValueError, match="only for USD-M"):
        plan_for((ArchiveKind.BOOK_DEPTH,), markets=(StoreMarket.SPOT,))
    with pytest.raises(ValueError, match="without duplicates"):
        plan_for((ArchiveKind.METRICS, ArchiveKind.METRICS))
    with pytest.raises(ValueError, match="without duplicates"):
        plan_for(())


@pytest.mark.parametrize("kind", [ArchiveKind.METRICS, ArchiveKind.BOOK_DEPTH])
def test_planning_refuses_the_embargoed_spans_for_the_new_kinds(
    kind: ArchiveKind,
) -> None:
    last_allowed = datetime(2026, 8, 1, tzinfo=UTC)
    plan = plan_for((kind,), start=last_allowed - timedelta(days=2), end=last_allowed)
    assert max(spec.period_end for spec in plan.specs) == last_allowed
    with pytest.raises(Fs1WarmupSpanError):
        plan_for(
            (kind,),
            start=last_allowed - timedelta(days=1),
            end=last_allowed + timedelta(days=1),
        )
    with pytest.raises(Fs1WarmupSpanError):
        plan_for((kind,), start=last_allowed, end=datetime(2026, 10, 30, tzinfo=UTC))
    for policy in (ENFORCED, WARMUP):
        with pytest.raises(SealedWindowEmbargoError):
            plan_for(
                (kind,),
                start=datetime(2026, 10, 31, tzinfo=UTC),
                end=datetime(2026, 11, 2, tzinfo=UTC),
                embargo=policy,
            )
    admitted = plan_for(
        (kind,),
        start=datetime(2026, 7, 31, tzinfo=UTC),
        end=datetime(2026, 8, 3, tzinfo=UTC),
        embargo=WARMUP,
    )
    assert [s.period_start.day for s in admitted.specs if s.symbol == "BTCUSDT"] == [
        31,
        1,
        2,
    ]


# -- parsing: the real layout -----------------------------------------------------------


def test_real_layout_metrics_parse_exactly_and_sort_ascending() -> None:
    rows = parse_open_interest_metrics_archive(
        metrics_payload(*REAL_METRICS),
        derivatives_spec(ArchiveKind.METRICS),
        embargo=ENFORCED,
    )
    assert [row.create_time for row in rows] == [
        datetime(2024, 3, 15, 0, 0, tzinfo=UTC),
        datetime(2024, 3, 15, 0, 5, tzinfo=UTC),
        datetime(2024, 3, 15, 0, 10, tzinfo=UTC),
    ]
    first = rows[0]
    assert isinstance(first, OpenInterestMetricRow)
    assert first.create_time_ms == ms_from_utc(datetime(2024, 3, 15, tzinfo=UTC))
    assert first.sum_open_interest == Decimal("83981.907")
    assert format(first.sum_open_interest, "f") == "83981.9070000000000000"
    assert format(first.sum_open_interest_value, "f") == "6000778051.8311230000000000"
    assert format(first.count_toptrader_long_short_ratio, "f") == "1.76875308"
    assert format(first.sum_taker_long_short_vol_ratio, "f") == "1.09685300"
    with pytest.raises(AttributeError):
        first.sum_open_interest = Decimal(1)  # type: ignore[misc]


def test_real_layout_book_depth_parses_exactly_and_sorts() -> None:
    rows = parse_book_depth_archive(
        depth_payload(*reversed(REAL_DEPTH)),
        derivatives_spec(ArchiveKind.BOOK_DEPTH),
        embargo=ENFORCED,
    )
    assert len(rows) == 12
    assert [(r.timestamp_ms, r.percentage) for r in rows] == sorted(
        (r.timestamp_ms, r.percentage) for r in rows
    )
    first = rows[0]
    assert isinstance(first, BookDepthRow)
    assert (first.timestamp, first.percentage) == (
        datetime(2024, 3, 15, 0, 0, 9, tzinfo=UTC),
        -5,
    )
    assert format(first.depth, "f") == "5828.70100000"
    assert format(first.notional, "f") == "406718563.05162000"
    assert first.key == (first.timestamp_ms, -5)
    assert (
        rows[-1].percentage == 5
        and rows[-1].timestamp_ms - first.timestamp_ms == 29_000
    )


def test_empty_and_header_only_archives_are_valid() -> None:
    spec = derivatives_spec(ArchiveKind.METRICS)
    assert (
        parse_open_interest_metrics_archive(zip_csv([]), spec, embargo=ENFORCED) == []
    )
    assert (
        parse_open_interest_metrics_archive(metrics_payload(), spec, embargo=ENFORCED)
        == []
    )
    depth = derivatives_spec(ArchiveKind.BOOK_DEPTH)
    assert parse_book_depth_archive(depth_payload(), depth, embargo=ENFORCED) == []


def test_synthetic_full_day_archives_round_trip() -> None:
    spec = derivatives_spec(ArchiveKind.METRICS)
    rows = parse_open_interest_metrics_archive(
        archive_bytes(spec), spec, embargo=ENFORCED
    )
    assert len(rows) == 288
    assert [r.create_time_ms for r in rows] == sorted(r.create_time_ms for r in rows)
    depth = derivatives_spec(ArchiveKind.BOOK_DEPTH)
    parsed = parse_book_depth_archive(archive_bytes(depth), depth, embargo=ENFORCED)
    assert len(parsed) == 144 * 10 and len({r.key for r in parsed}) == len(parsed)


# -- parsing: refusals ----------------------------------------------------------------------

GOOD_M = REAL_METRICS[0]
GOOD_D = REAL_DEPTH[0]


def mutate(line: str, index: int, value: str) -> str:
    cells = line.split(",")
    cells[index] = value
    return ",".join(cells)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([GOOD_M.rsplit(",", 1)[0]], "7 columns"),
        ([GOOD_M + ",1.0"], "9 columns"),
        ([mutate(GOOD_M, 1, "ETHUSDT")], "does not match"),
        ([mutate(GOOD_M, 2, "NaN")], "invalid sum_open_interest"),
        ([mutate(GOOD_M, 3, "Infinity")], "invalid sum_open_interest_value"),
        ([mutate(GOOD_M, 4, "-inf")], "invalid count_toptrader"),
        ([mutate(GOOD_M, 5, "-0.5")], "negative sum_toptrader_long_short_ratio"),
        ([mutate(GOOD_M, 2, "-1.0")], "negative sum_open_interest"),
        ([mutate(GOOD_M, 6, "1e3")], "invalid count_long_short_ratio"),
        ([mutate(GOOD_M, 7, "+1.5")], "invalid sum_taker"),
        ([mutate(GOOD_M, 2, "1_000.5")], "invalid sum_open_interest"),
        ([mutate(GOOD_M, 2, "")], "invalid sum_open_interest"),
        ([mutate(GOOD_M, 2, " 1.5")], "invalid sum_open_interest"),
        ([mutate(GOOD_M, 0, "2024-03-15T00:05:00")], "invalid create_time"),
        ([mutate(GOOD_M, 0, "2024-03-15 00:05:00+00:00")], "invalid create_time"),
        ([mutate(GOOD_M, 0, "1710460800000")], "invalid create_time"),
        ([mutate(GOOD_M, 0, "2024-03-15 25:05:00")], "invalid create_time"),
        ([mutate(GOOD_M, 0, "2024-03-16 00:00:00")], "outside the archive period"),
        ([mutate(GOOD_M, 0, "2024-03-14 23:55:00")], "outside the archive period"),
        ([GOOD_M, GOOD_M], "duplicate create_time"),
        ([GOOD_M, mutate(GOOD_M, 2, "1.0")], "duplicate create_time"),
    ],
)
def test_invalid_metrics_rows_refuse_the_whole_archive(
    lines: list[str], message: str
) -> None:
    with pytest.raises(ArchiveContentError, match=message):
        parse_open_interest_metrics_archive(
            metrics_payload(*lines),
            derivatives_spec(ArchiveKind.METRICS),
            embargo=ENFORCED,
        )


@pytest.mark.parametrize(
    "header",
    [
        "time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio",
        METRICS_HEADER + ",extra",
        METRICS_HEADER.rsplit(",", 1)[0],
        GOOD_M,  # no header at all
    ],
)
def test_metrics_need_the_exact_observed_header(header: str) -> None:
    with pytest.raises(ArchiveContentError, match="header"):
        parse_open_interest_metrics_archive(
            metrics_payload(GOOD_M, header=header),
            derivatives_spec(ArchiveKind.METRICS),
            embargo=ENFORCED,
        )


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([GOOD_D.rsplit(",", 1)[0]], "3 columns"),
        ([GOOD_D + ",9"], "5 columns"),
        ([mutate(GOOD_D, 1, "0")], "not a signed band"),
        ([mutate(GOOD_D, 1, "101")], "not a signed band"),
        ([mutate(GOOD_D, 1, "-101")], "not a signed band"),
        ([mutate(GOOD_D, 1, "1.5")], "invalid percentage"),
        ([mutate(GOOD_D, 1, "")], "invalid percentage"),
        ([mutate(GOOD_D, 1, "+5")], "invalid percentage"),
        ([mutate(GOOD_D, 2, "NaN")], "invalid depth"),
        ([mutate(GOOD_D, 2, "-1.00000000")], "negative depth"),
        ([mutate(GOOD_D, 3, "inf")], "invalid notional"),
        ([mutate(GOOD_D, 3, "-3")], "negative notional"),
        ([mutate(GOOD_D, 3, "1E+9")], "invalid notional"),
        ([mutate(GOOD_D, 0, "2024-03-15 00:00")], "invalid timestamp"),
        ([mutate(GOOD_D, 0, "2024-03-15T00:00:09Z")], "invalid timestamp"),
        ([mutate(GOOD_D, 0, "2024-03-16 00:00:00")], "outside the archive period"),
        ([GOOD_D, GOOD_D], "duplicate"),
        ([GOOD_D, mutate(GOOD_D, 2, "1.0")], "duplicate"),
    ],
)
def test_invalid_book_depth_rows_refuse_the_whole_archive(
    lines: list[str], message: str
) -> None:
    with pytest.raises(ArchiveContentError, match=message):
        parse_book_depth_archive(
            depth_payload(*lines),
            derivatives_spec(ArchiveKind.BOOK_DEPTH),
            embargo=ENFORCED,
        )


@pytest.mark.parametrize(
    "header", ["time,percentage,depth,notional", DEPTH_HEADER + ",x", GOOD_D]
)
def test_book_depth_needs_the_exact_observed_header(header: str) -> None:
    with pytest.raises(ArchiveContentError, match="header"):
        parse_book_depth_archive(
            depth_payload(GOOD_D, header=header),
            derivatives_spec(ArchiveKind.BOOK_DEPTH),
            embargo=ENFORCED,
        )


def test_parsers_refuse_a_spec_of_another_kind_or_a_spot_market() -> None:
    metrics = derivatives_spec(ArchiveKind.METRICS)
    with pytest.raises(ValueError, match="not a book_depth"):
        parse_book_depth_archive(depth_payload(), metrics, embargo=ENFORCED)
    with pytest.raises(ValueError, match="not a metrics"):
        parse_open_interest_metrics_archive(
            metrics_payload(),
            derivatives_spec(ArchiveKind.BOOK_DEPTH),
            embargo=ENFORCED,
        )


# -- embargo: rows, at parse time ------------------------------------------------------------


def test_rows_in_the_warmup_span_are_refused_unless_authorized_and_sealed_never() -> (
    None
):
    warm_day = datetime(2026, 8, 1, tzinfo=UTC)
    sealed_day = datetime(2026, 11, 1, tzinfo=UTC)
    for kind, parse, payload in (
        (
            ArchiveKind.METRICS,
            parse_open_interest_metrics_archive,
            lambda day: metrics_payload(mutate(GOOD_M, 0, f"{day:%Y-%m-%d} 00:05:00")),
        ),
        (
            ArchiveKind.BOOK_DEPTH,
            parse_book_depth_archive,
            lambda day: depth_payload(mutate(GOOD_D, 0, f"{day:%Y-%m-%d} 00:00:09")),
        ),
    ):
        warm_spec = derivatives_spec(kind, warm_day)
        with pytest.raises(Fs1WarmupSpanError):
            parse(payload(warm_day), warm_spec, embargo=ENFORCED)
        assert len(parse(payload(warm_day), warm_spec, embargo=WARMUP)) == 1
        sealed_spec = derivatives_spec(kind, sealed_day)
        for policy in (ENFORCED, WARMUP):
            with pytest.raises(SealedWindowEmbargoError):
                parse(payload(sealed_day), sealed_spec, embargo=policy)


# -- fetching: URLs are gated before any request ----------------------------------------------


@pytest.mark.parametrize("kind", [ArchiveKind.METRICS, ArchiveKind.BOOK_DEPTH])
def test_the_url_gate_refuses_embargoed_days_for_the_new_kinds(
    kind: ArchiveKind,
) -> None:
    ok = vision_daily_derivatives_url(
        kind, StoreMarket.USDS_M, "BTCUSDT", datetime(2026, 7, 31, tzinfo=UTC)
    )
    assert check_url_fetchable(ok, ENFORCED).kind is kind
    warm = vision_daily_derivatives_url(
        kind, StoreMarket.USDS_M, "BTCUSDT", datetime(2026, 8, 1, tzinfo=UTC)
    )
    with pytest.raises(Fs1WarmupSpanError):
        check_url_fetchable(warm, ENFORCED)
    assert check_url_fetchable(warm, WARMUP).kind is kind
    sealed = vision_daily_derivatives_url(
        kind, StoreMarket.USDS_M, "BTCUSDT", datetime(2026, 11, 1, tzinfo=UTC)
    )
    for policy in (ENFORCED, WARMUP):
        with pytest.raises(SealedWindowEmbargoError):
            check_url_fetchable(sealed, policy)
        with pytest.raises(SealedWindowEmbargoError):
            check_url_fetchable(f"{sealed}.CHECKSUM", policy)


def test_the_vision_fetcher_never_opens_a_connection_for_an_embargoed_new_kind() -> (
    None
):
    from trade_rl.integrations.market_store.fetch import VisionFetcher

    def opener(*_a: object, **_k: object) -> object:
        raise AssertionError("the network must not be touched")

    fetcher = VisionFetcher(_opener=opener)
    for kind in (ArchiveKind.METRICS, ArchiveKind.BOOK_DEPTH):
        url = vision_daily_derivatives_url(
            kind, StoreMarket.USDS_M, "BTCUSDT", datetime(2026, 8, 1, tzinfo=UTC)
        )
        with pytest.raises(Fs1WarmupSpanError):
            fetcher.fetch(url)
        sealed = vision_daily_derivatives_url(
            kind, StoreMarket.USDS_M, "BTCUSDT", datetime(2026, 11, 3, tzinfo=UTC)
        )
        with pytest.raises(SealedWindowEmbargoError):
            VisionFetcher(warmup_authorization_url=AUTH, _opener=opener).fetch(sealed)


# -- ingest orchestration (in-memory store) ---------------------------------------------------


def ingest(
    store: MemoryStore, fetcher: FakeFetcher, plan: IngestionPlan, **kwargs: Any
) -> IngestReport:
    return run_ingestion(
        store,
        fetcher,
        plan,
        code_commit="t",
        command_line="t",
        clock=lambda: NOW,
        **kwargs,
    )


def new_kind_plan(day: datetime = DAY, days: int = 1) -> IngestionPlan:
    return plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.USDS_M,)),
        start=day,
        end=day + timedelta(days=days),
        now=NOW,
        kinds=(ArchiveKind.METRICS, ArchiveKind.BOOK_DEPTH),
    )


def fetcher_for_plan(plan: IngestionPlan) -> FakeFetcher:
    fetcher = FakeFetcher()
    for spec in plan.specs:
        fetcher.add(spec.url, archive_bytes(spec))
    return fetcher


def test_ingest_stores_both_kinds_and_is_idempotent() -> None:
    plan = new_kind_plan(days=2)
    store, fetcher = MemoryStore(), fetcher_for_plan(plan)
    first = ingest(store, fetcher, plan)
    assert first.counts.archives_ingested == 4
    assert first.counts.rows_inserted == 2 * 288 + 2 * 1440
    assert len(store.metrics) == 576 and len(store.depth) == 2880
    assert store.klines == {} and store.funding == {}
    assert {row["period_end"] for row in store.sources.values()} == {
        DAY + timedelta(days=1),
        DAY + timedelta(days=2),
    }

    fetcher.requested.clear()
    second = ingest(store, fetcher, plan)
    assert fetcher.requested == []
    assert second.counts.archives_skipped == 4 and second.counts.rows_inserted == 0

    refetched = ingest(store, fetcher, plan, refetch=True)
    assert (
        refetched.counts.archives_skipped == 4 and refetched.counts.rows_inserted == 0
    )
    assert len(store.metrics) == 576 and len(store.depth) == 2880


def test_a_partially_known_day_inserts_only_the_unseen_keys() -> None:
    plan = new_kind_plan()
    store = MemoryStore()
    instrument = store.ensure_instrument(StoreMarket.USDS_M, "BTCUSDT")
    spec = next(s for s in plan.specs if s.kind is ArchiveKind.METRICS)
    known = parse_open_interest_metrics_archive(
        archive_bytes(spec), spec, embargo=ENFORCED
    )[:10]
    store.insert_open_interest_metrics(instrument, 99, known)
    report = ingest(store, fetcher_for_plan(plan), plan)
    assert report.counts.rows_identical == 10
    assert len(store.metrics) == 288


def test_a_changed_metrics_value_is_a_conflict_and_nothing_is_overwritten() -> None:
    plan = new_kind_plan()
    store = MemoryStore()
    instrument = store.ensure_instrument(StoreMarket.USDS_M, "BTCUSDT")
    spec = next(s for s in plan.specs if s.kind is ArchiveKind.METRICS)
    rows = parse_open_interest_metrics_archive(
        archive_bytes(spec), spec, embargo=ENFORCED
    )
    changed = replace(rows[3], sum_open_interest=rows[3].sum_open_interest + 1)
    store.insert_open_interest_metrics(instrument, 99, [changed])
    with pytest.raises(ConflictError, match="nothing was overwritten") as caught:
        ingest(store, fetcher_for_plan(plan), plan)
    [record] = caught.value.conflicts
    assert (record.kind, record.column, record.key_time_ms) == (
        "open_interest_metric",
        "sum_open_interest",
        rows[3].create_time_ms,
    )
    assert store.metrics[(instrument, rows[3].create_time_ms)] == changed
    assert len(store.metrics) == 1 and spec.url not in store.sources
    assert store.runs[1]["status"] == "failed" and store.conflicts == [record]


def test_a_changed_book_depth_value_names_the_band_in_the_conflict() -> None:
    plan = new_kind_plan()
    store = MemoryStore()
    instrument = store.ensure_instrument(StoreMarket.USDS_M, "BTCUSDT")
    spec = next(s for s in plan.specs if s.kind is ArchiveKind.BOOK_DEPTH)
    rows = parse_book_depth_archive(archive_bytes(spec), spec, embargo=ENFORCED)
    target = rows[7]
    changed = BookDepthRow(
        target.timestamp_ms, target.percentage, target.depth, target.notional + 1
    )
    store.insert_book_depth(instrument, 99, [changed])
    with pytest.raises(ConflictError) as caught:
        ingest(store, fetcher_for_plan(plan), plan)
    [record] = caught.value.conflicts
    assert record.kind == "book_depth"
    assert record.column == f"notional@percentage={target.percentage}"
    assert record.key_time_ms == target.timestamp_ms
    assert Decimal(record.incoming or "") + 1 == Decimal(record.existing or "")
    assert len(store.depth) == 1 and spec.url not in store.sources


def test_a_changed_archive_hash_is_an_archive_conflict() -> None:
    plan = new_kind_plan()
    store, fetcher = MemoryStore(), fetcher_for_plan(plan)
    ingest(store, fetcher, plan)
    spec = plan.specs[0]
    fetcher.add(spec.url, archive_bytes(spec, value_bump=1))
    with pytest.raises(ConflictError) as caught:
        ingest(store, fetcher, plan, refetch=True)
    assert caught.value.conflicts[0].kind == "archive_sha256"


def test_a_404_day_is_recorded_missing_and_retried() -> None:
    plan = new_kind_plan()
    store, fetcher = MemoryStore(), fetcher_for_plan(plan)
    gone = plan.specs[0]
    published = fetcher.files.pop(gone.url)
    report = ingest(store, fetcher, plan)
    assert report.counts.archives_missing == 1 and report.missing_urls == (gone.url,)
    fetcher.files[gone.url] = published
    again = ingest(store, fetcher, plan)
    assert (
        again.counts.archives_ingested == 1
        and store.sources[gone.url]["status"] == "ingested"
    )


@pytest.mark.parametrize("kind", [ArchiveKind.METRICS, ArchiveKind.BOOK_DEPTH])
def test_ingest_refuses_embargoed_days_before_any_run_or_request(
    kind: ArchiveKind,
) -> None:
    universe = Universe(symbols=("BTCUSDT",), markets=(StoreMarket.USDS_M,))
    cases = (
        (datetime(2026, 8, 1, tzinfo=UTC), ENFORCED, Fs1WarmupSpanError),
        (datetime(2026, 10, 31, tzinfo=UTC), ENFORCED, Fs1WarmupSpanError),
        (datetime(2026, 11, 1, tzinfo=UTC), ENFORCED, SealedWindowEmbargoError),
        (datetime(2026, 11, 1, tzinfo=UTC), WARMUP, SealedWindowEmbargoError),
        (datetime(2026, 11, 20, tzinfo=UTC), WARMUP, SealedWindowEmbargoError),
    )
    for day, policy, error in cases:
        spec = derivatives_spec(kind, day)
        plan = IngestionPlan(
            universe, day, day + timedelta(days=1), policy, (spec,), (), (kind,)
        )
        fetcher, store = FakeFetcher(), MemoryStore()
        for url in (spec.url, spec.checksum_url):
            fetcher.files[url] = b"must never be downloaded"
        with pytest.raises(error):
            ingest(store, fetcher, plan)
        assert fetcher.requested == [] and store.runs == {} and store.sources == {}


def test_a_warmup_day_with_the_authorization_is_admitted_for_the_new_kinds() -> None:
    day = datetime(2026, 8, 1, tzinfo=UTC)
    plan = plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.USDS_M,)),
        start=day,
        end=day + timedelta(days=1),
        now=datetime(2026, 11, 1, 12, tzinfo=UTC),
        embargo=WARMUP,
        kinds=(ArchiveKind.METRICS, ArchiveKind.BOOK_DEPTH),
    )
    store = MemoryStore()
    report = ingest(store, fetcher_for_plan(plan), plan)
    assert report.counts.archives_ingested == 2
    assert store.runs[1]["fs1_warmup_allowed"] is True


# -- command line ------------------------------------------------------------------------------

CLI_NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


def invoke(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = run(list(argv), out=out, err=err, clock=lambda: CLI_NOW)
    return code, out.getvalue(), err.getvalue()


def test_cli_default_kinds_are_exactly_the_current_behaviour() -> None:
    base = ("plan", "--bases", "BTC,ETH", "--start", "2026-06-01")
    code, default_out, _ = invoke(*base)
    assert code == EXIT_OK
    explicit_code, explicit_out, _ = invoke(*base, "--kinds", "klines,funding_rate")
    assert explicit_code == EXIT_OK and explicit_out == default_out
    plan = json.loads(default_out)
    assert "kinds" not in plan
    assert {item["kind"] for item in plan["archives"]} == {"klines", "funding_rate"}
    # ingest is a dry run without --allow-network and plans the same default
    code, out, _ = invoke("ingest", "--bases", "BTC,ETH", "--start", "2026-06-01")
    assert code == EXIT_OK and json.loads(out)["plan"] == plan


def test_cli_kinds_opt_in_plans_the_daily_usds_m_archives() -> None:
    code, out, _ = invoke(
        "plan",
        "--bases",
        "BTC",
        "--start",
        "2026-07-29",
        "--kinds",
        "klines,funding_rate,metrics,book_depth",
    )
    assert code == EXIT_OK
    plan = json.loads(out)
    assert plan["kinds"] == ["klines", "funding_rate", "metrics", "book_depth"]
    new = [a for a in plan["archives"] if a["kind"] in ("metrics", "book_depth")]
    assert len(new) == 6 and {a["market"] for a in new} == {"usds_m"}
    assert {a["period_kind"] for a in new} == {"daily"}
    assert max(a["period_end"] for a in plan["archives"]) == "2026-08-01T00:00:00+00:00"
    assert any(a["url"].endswith("/BTCUSDT-bookDepth-2026-07-31.zip") for a in new)


def test_cli_kinds_alone_and_bad_values() -> None:
    code, out, _ = invoke(
        "plan", "--bases", "BTC", "--start", "2026-07-30", "--kinds", "metrics"
    )
    assert code == EXIT_OK
    assert {a["kind"] for a in json.loads(out)["archives"]} == {"metrics"}
    for bad in ("tickers", "metrics,metrics"):
        with pytest.raises(SystemExit):
            invoke("plan", "--kinds", bad)
    code, _, err = invoke("plan", "--markets", "spot", "--kinds", "metrics")
    assert code == EXIT_FAILURE and "only for USD-M" in err


@pytest.mark.parametrize(
    "kinds", ["metrics", "book_depth", "klines,funding_rate,metrics,book_depth"]
)
@pytest.mark.parametrize("command", ["plan", "ingest"])
def test_cli_refuses_embargoed_ranges_for_the_new_kinds(
    command: str, kinds: str
) -> None:
    extra = ("--allow-network",) if command == "ingest" else ()
    code, out, err = invoke(command, "--kinds", kinds, "--end", "2026-08-02", *extra)
    assert code == EXIT_EMBARGO and out == "" and "warm-up" in err
    flags = ("--allow-fs1-warmup-span", "--fs1-warmup-authorization-url", AUTH)
    code, out, err = invoke(
        command, "--kinds", kinds, "--end", "2026-11-02", *flags, *extra
    )
    assert code == EXIT_EMBARGO and out == "" and ("sealed window" in err)


# -- reading ------------------------------------------------------------------------------------


def test_the_read_range_is_half_open_utc_and_embargo_checked() -> None:
    start, end = checked_read_range(
        datetime(2024, 3, 15, 9, tzinfo=timezone(timedelta(hours=9))),
        DAY + timedelta(days=1),
    )
    assert start == datetime(2024, 3, 15, 0, tzinfo=UTC) and end == NEXT_DAY
    with pytest.raises(ValueError, match="timezone-aware"):
        checked_read_range(datetime(2024, 3, 15), NEXT_DAY)
    with pytest.raises(ValueError, match="timezone-aware"):
        checked_read_range(DAY, datetime(2024, 3, 16))
    with pytest.raises(ValueError, match="later than start"):
        checked_read_range(DAY, DAY)
    last = datetime(2026, 8, 1, tzinfo=UTC)
    assert checked_read_range(last - timedelta(days=1), last)[1] == last
    with pytest.raises(Fs1WarmupSpanError):
        checked_read_range(last - timedelta(days=1), last + timedelta(seconds=1))
    sealed = datetime(2026, 11, 1, tzinfo=UTC)
    assert (
        checked_read_range(sealed - timedelta(days=1), sealed, policy=WARMUP)[1]
        == sealed
    )
    for policy in (None, WARMUP):
        with pytest.raises(SealedWindowEmbargoError):
            checked_read_range(
                sealed - timedelta(days=1), sealed + timedelta(seconds=1), policy=policy
            )
        with pytest.raises(SealedWindowEmbargoError):
            checked_read_range(sealed, sealed + timedelta(days=1), policy=policy)


def test_store_reads_refuse_a_closed_range_before_any_query() -> None:
    pytest.importorskip("psycopg", reason="install the `store` extra")
    from trade_rl.integrations.market_store.store import PostgresMarketStore

    class NoQueries:
        def cursor(self, *_a: object, **_k: object) -> object:
            raise AssertionError("a refused range must not reach the database")

    store = PostgresMarketStore(NoQueries())  # type: ignore[arg-type]
    end = datetime(2026, 8, 2, tzinfo=UTC)
    for read in (store.read_open_interest_metrics, store.read_book_depth):
        with pytest.raises(Fs1WarmupSpanError):
            read("BTCUSDT", DAY, end)
        with pytest.raises(SealedWindowEmbargoError):
            read("BTCUSDT", DAY, datetime(2026, 11, 2, tzinfo=UTC), embargo=WARMUP)
        with pytest.raises(ValueError, match="invalid symbol"):
            read("btc; drop table", DAY, NEXT_DAY)
