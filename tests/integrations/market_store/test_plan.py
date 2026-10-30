from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade_rl.integrations.market_store.contracts import (
    DEFAULT_BASE_ASSETS,
    ArchiveKind,
    PeriodKind,
    StoreMarket,
    Universe,
)
from trade_rl.integrations.market_store.embargo import (
    SEALED_WINDOW_START,
    EmbargoPolicy,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.plan import plan_ingestion

URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-42"
ONE = Universe(symbols=("BTCUSDT",), markets=(StoreMarket.USDS_M, StoreMarket.SPOT))


def day(year: int, month: int, number: int = 1) -> datetime:
    return datetime(year, month, number, tzinfo=UTC)


def names(plan_specs: object) -> list[str]:
    return [spec.url.rsplit("/", 1)[1] for spec in plan_specs]  # type: ignore[attr-defined]


def test_default_universe_is_the_six_carry_symbols_on_both_markets() -> None:
    universe = Universe()
    assert universe.symbols == tuple(f"{base}USDT" for base in DEFAULT_BASE_ASSETS)
    assert DEFAULT_BASE_ASSETS == ("BTC", "ETH", "BNB", "XRP", "ADA", "SOL")
    assert universe.markets == (StoreMarket.USDS_M, StoreMarket.SPOT)
    assert universe.interval == "1h"


def test_completed_months_are_monthly_and_funding_is_usds_m_only() -> None:
    plan = plan_ingestion(
        ONE, start=day(2024, 1), end=day(2024, 3), now=day(2026, 10, 30)
    )
    assert names(plan.specs) == [
        "BTCUSDT-1h-2024-01.zip",
        "BTCUSDT-1h-2024-02.zip",
        "BTCUSDT-fundingRate-2024-01.zip",
        "BTCUSDT-fundingRate-2024-02.zip",
        "BTCUSDT-1h-2024-01.zip",
        "BTCUSDT-1h-2024-02.zip",
    ]
    markets = [spec.market for spec in plan.specs]
    assert markets == [StoreMarket.USDS_M] * 4 + [StoreMarket.SPOT] * 2
    assert all(spec.period_kind is PeriodKind.MONTHLY for spec in plan.specs)
    assert all(spec.checksum_url == f"{spec.url}.CHECKSUM" for spec in plan.specs)
    futures = [s.url for s in plan.specs if s.market is StoreMarket.USDS_M]
    assert all("/futures/um/monthly/" in url for url in futures)
    spot = [s.url for s in plan.specs if s.market is StoreMarket.SPOT]
    assert all("/spot/monthly/klines/" in url for url in spot)


def test_current_partial_month_uses_daily_archives_up_to_yesterday() -> None:
    plan = plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.USDS_M,)),
        start=day(2026, 10),
        end=day(2026, 11),
        now=datetime(2026, 10, 12, 9, tzinfo=UTC),
    )
    klines = [s for s in plan.specs if s.kind is ArchiveKind.KLINES]
    assert [s.period_start.day for s in klines] == list(range(1, 12))
    assert all(s.period_kind is PeriodKind.DAILY for s in klines)
    assert plan.end == day(2026, 10, 12)
    funding_skips = [s for s in plan.skipped if s.kind is ArchiveKind.FUNDING_RATE]
    assert [s.period_start for s in funding_skips] == [day(2026, 10)]
    assert not [s for s in plan.specs if s.kind is ArchiveKind.FUNDING_RATE]


def test_recently_completed_month_waits_for_publication_lag() -> None:
    universe = Universe(symbols=("BTCUSDT",), markets=(StoreMarket.SPOT,))
    early = plan_ingestion(
        universe, start=day(2026, 9), end=day(2026, 10), now=day(2026, 10, 3)
    )
    assert all(s.period_kind is PeriodKind.DAILY for s in early.specs)
    assert len(early.specs) == 30
    late = plan_ingestion(
        universe, start=day(2026, 9), end=day(2026, 10), now=day(2026, 10, 20)
    )
    assert names(late.specs) == ["BTCUSDT-1h-2026-09.zip"]


def test_partial_head_month_is_daily_then_months_are_monthly() -> None:
    plan = plan_ingestion(
        Universe(symbols=("ETHUSDT",), markets=(StoreMarket.SPOT,)),
        start=day(2024, 1, 29),
        end=day(2024, 3),
        now=day(2026, 10, 30),
    )
    assert names(plan.specs) == [
        "ETHUSDT-1h-2024-01-29.zip",
        "ETHUSDT-1h-2024-01-30.zip",
        "ETHUSDT-1h-2024-01-31.zip",
        "ETHUSDT-1h-2024-02.zip",
    ]


def test_requested_end_inside_the_sealed_window_is_refused() -> None:
    for end in (day(2026, 11, 2), day(2027, 1), day(2030, 1)):
        with pytest.raises(SealedWindowEmbargoError):
            plan_ingestion(ONE, start=day(2026, 10), end=end, now=day(2031, 1))


def test_end_exactly_at_the_boundary_is_allowed_and_plans_october_monthly() -> None:
    plan = plan_ingestion(
        ONE, start=day(2026, 10), end=SEALED_WINDOW_START, now=day(2026, 12, 15)
    )
    assert "BTCUSDT-1h-2026-10.zip" in names(plan.specs)
    assert "BTCUSDT-fundingRate-2026-10.zip" in names(plan.specs)
    assert all(spec.period_end <= SEALED_WINDOW_START for spec in plan.specs)
    assert not any(
        "2026-11" in spec.url or "2026-12" in spec.url for spec in plan.specs
    )


def test_no_plan_ever_contains_an_archive_reaching_the_boundary() -> None:
    universe = Universe(symbols=("BTCUSDT", "ETHUSDT"))
    for offset in range(0, 400, 7):
        now = day(2026, 6) + timedelta(days=offset, hours=5)
        end = min(
            SEALED_WINDOW_START, now.replace(hour=0, minute=0, second=0, microsecond=0)
        )
        plan = plan_ingestion(universe, start=day(2026, 5), end=end, now=now)
        for spec in plan.specs:
            assert spec.period_end <= SEALED_WINDOW_START, spec.url
            assert spec.period_start < SEALED_WINDOW_START, spec.url


def test_monthly_archive_straddling_a_mid_month_boundary_is_never_planned() -> None:
    boundary = datetime(2026, 11, 15, tzinfo=UTC)
    policy = EmbargoPolicy.enforced(boundary)
    plan = plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.USDS_M,)),
        start=day(2026, 11),
        end=boundary,
        now=day(2027, 3),
        embargo=policy,
    )
    klines = [s for s in plan.specs if s.kind is ArchiveKind.KLINES]
    assert [s.period_kind for s in klines] == [PeriodKind.DAILY] * 14
    assert klines[-1].period_start == day(2026, 11, 14)
    assert all("2026-11.zip" not in s.url for s in plan.specs)
    # there is no daily funding archive, so the straddling month is skipped and reported
    assert not [s for s in plan.specs if s.kind is ArchiveKind.FUNDING_RATE]
    assert [s.period_start for s in plan.skipped] == [day(2026, 11)]
    for spec in plan.specs:
        assert spec.period_end <= boundary

    with pytest.raises(SealedWindowEmbargoError):
        plan_ingestion(
            ONE,
            start=day(2026, 11),
            end=day(2026, 11, 20),
            now=day(2027, 3),
            embargo=policy,
        )


def test_released_policy_plans_past_the_boundary() -> None:
    plan = plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.SPOT,)),
        start=day(2026, 11),
        end=day(2027, 1),
        now=day(2027, 6),
        embargo=EmbargoPolicy.release(URL),
    )
    assert names(plan.specs) == [
        "BTCUSDT-1h-2026-11.zip",
        "BTCUSDT-1h-2026-12.zip",
    ]
    assert plan.embargo.released


def test_planning_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="midnight"):
        plan_ingestion(
            ONE,
            start=datetime(2024, 1, 1, 1, tzinfo=UTC),
            end=day(2024, 2),
            now=day(2026, 1),
        )
    with pytest.raises(ValueError, match="later"):
        plan_ingestion(ONE, start=day(2024, 2), end=day(2024, 1), now=day(2026, 1))
    with pytest.raises(ValueError, match="timezone"):
        plan_ingestion(
            ONE, start=datetime(2024, 1, 1), end=day(2024, 2), now=day(2026, 1)
        )


def test_plan_is_deterministic_and_has_no_duplicates() -> None:
    first = plan_ingestion(
        Universe(), start=day(2023, 1), end=day(2024, 1), now=day(2026, 10, 30)
    )
    second = plan_ingestion(
        Universe(), start=day(2023, 1), end=day(2024, 1), now=day(2026, 10, 30)
    )
    assert first.to_dict() == second.to_dict()
    urls = [spec.url for spec in first.specs]
    assert len(urls) == len(set(urls))
