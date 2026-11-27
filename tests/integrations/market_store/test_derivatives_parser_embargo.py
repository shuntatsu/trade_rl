"""Synthetic archive-level embargo and URL identity regressions; no network."""

from __future__ import annotations

import io
import zipfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trade_rl.integrations.market_store.archive_url import (
    ArchiveSpecMismatchError,
    vision_daily_derivatives_url,
)
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    BookDepthRow,
    OpenInterestMetricRow,
    PeriodKind,
    StoreMarket,
)
from trade_rl.integrations.market_store.embargo import (
    EmbargoPolicy,
    Fs1WarmupSpanError,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.parse import (
    BOOK_DEPTH_HEADER,
    METRICS_HEADER,
    ArchiveContentError,
    parse_book_depth_archive,
    parse_open_interest_metrics_archive,
)

KINDS = (ArchiveKind.METRICS, ArchiveKind.BOOK_DEPTH)
ALLOWED = datetime(2026, 7, 31, tzinfo=UTC)
WARM = datetime(2026, 8, 1, tzinfo=UTC)
LAST_WARM = datetime(2026, 10, 31, tzinfo=UTC)
SEALED = datetime(2026, 11, 1, tzinfo=UTC)
DEFAULT = EmbargoPolicy.enforced()
# Synthetic authorization metadata only; no actual data or authorization is read.
WARMUP = EmbargoPolicy.with_warmup(
    "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1"
)


def _spec(kind: ArchiveKind, day: datetime) -> ArchiveSpec:
    url = vision_daily_derivatives_url(kind, StoreMarket.USDS_M, "BTCUSDT", day)
    return ArchiveSpec(
        kind=kind,
        market=StoreMarket.USDS_M,
        symbol="BTCUSDT",
        interval=None,
        period_kind=PeriodKind.DAILY,
        period_start=day,
        period_end=day + timedelta(days=1),
        url=url,
        checksum_url=f"{url}.CHECKSUM",
    )


def _parse(
    kind: ArchiveKind, payload: bytes, spec: ArchiveSpec, policy: EmbargoPolicy
) -> list[OpenInterestMetricRow] | list[BookDepthRow]:
    if kind is ArchiveKind.METRICS:
        return parse_open_interest_metrics_archive(payload, spec, embargo=policy)
    return parse_book_depth_archive(payload, spec, embargo=policy)


def _payload(kind: ArchiveKind, day: datetime, form: str) -> bytes:
    if form == "corrupt":
        return b"not a zip"
    stamp = day.strftime("%Y-%m-%d") + " 00:00:00"
    if kind is ArchiveKind.METRICS:
        header = METRICS_HEADER
        row = (stamp, "BTCUSDT", "1", "2", "3", "4", "5", "6")
    else:
        header = BOOK_DEPTH_HEADER
        row = (stamp, "-1", "1", "2")
    lines = [] if form == "empty" else [",".join(header)]
    if form == "one":
        lines.append(",".join(row))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("synthetic.csv", "\n".join(lines) + "\n")
    return buffer.getvalue()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("form", ["empty", "header", "one", "corrupt"])
@pytest.mark.parametrize(
    ("day", "policy", "error"),
    [
        (WARM, DEFAULT, Fs1WarmupSpanError),
        (SEALED, DEFAULT, SealedWindowEmbargoError),
        (SEALED, WARMUP, SealedWindowEmbargoError),
    ],
)
def test_closed_archive_refused_before_zip(
    kind: ArchiveKind,
    form: str,
    day: datetime,
    policy: EmbargoPolicy,
    error: type[ValueError],
) -> None:
    with pytest.raises(error):
        _parse(kind, _payload(kind, day, form), _spec(kind, day), policy)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("form", ["empty", "header", "one"])
@pytest.mark.parametrize("day", [WARM, SEALED])
def test_closed_url_cannot_be_relabelled_as_allowed(
    kind: ArchiveKind, form: str, day: datetime
) -> None:
    forged = replace(
        _spec(kind, day),
        period_start=ALLOWED,
        period_end=ALLOWED + timedelta(days=1),
    )
    with pytest.raises(ArchiveSpecMismatchError):
        _parse(kind, _payload(kind, ALLOWED, form), forged, DEFAULT)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("form", ["empty", "header", "one"])
@pytest.mark.parametrize(
    ("day", "policy"), [(ALLOWED, DEFAULT), (WARM, WARMUP), (LAST_WARM, WARMUP)]
)
def test_allowed_controls_unchanged(
    kind: ArchiveKind, form: str, day: datetime, policy: EmbargoPolicy
) -> None:
    rows = _parse(kind, _payload(kind, day, form), _spec(kind, day), policy)
    if form != "one":
        assert rows == []
    elif kind is ArchiveKind.METRICS:
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, OpenInterestMetricRow)
        assert row.create_time == day
        assert row.sum_open_interest == Decimal("1")
        assert row.sum_taker_long_short_vol_ratio == Decimal("6")
    else:
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, BookDepthRow)
        assert row.timestamp == day
        assert (row.percentage, row.depth, row.notional) == (
            -1,
            Decimal("1"),
            Decimal("2"),
        )


@pytest.mark.parametrize("kind", KINDS)
def test_per_row_embargo_remains_independent(kind: ArchiveKind) -> None:
    with pytest.raises(Fs1WarmupSpanError):
        _parse(kind, _payload(kind, WARM, "one"), _spec(kind, ALLOWED), DEFAULT)


@pytest.mark.parametrize("kind", KINDS)
def test_valid_archive_with_wrong_day_row_refused(kind: ArchiveKind) -> None:
    wrong_day = ALLOWED - timedelta(days=1)
    with pytest.raises(ArchiveContentError, match="outside the archive period"):
        _parse(kind, _payload(kind, wrong_day, "one"), _spec(kind, ALLOWED), DEFAULT)
