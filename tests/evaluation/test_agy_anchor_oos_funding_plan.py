"""The archive plan must cover the perpetual funding of the stage (``plan_does_not_cover``).

The first coverage runs at cdc7c0cf6 exited 3 for every perp leg with
``plan_does_not_cover`` for ``funding_rate``: the market-store planner plans a funding month
only when it ends inside the requested range, while a stage reads funding up to its read
limit, which lies inside a month. These tests use the real planner and the real archive
source with a fetcher that must never be called: nothing here touches the network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from tools import agy_anchor_oos as driver
from trade_rl.evaluation.agy_anchor_oos import data, protocol, runner
from trade_rl.integrations.binance.carry import CarryIntegrityHalt
from trade_rl.integrations.market_store.carry_source import VisionArchiveSource
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    StoreMarket,
    Universe,
    ms_from_utc,
)
from trade_rl.integrations.market_store.plan import plan_ingestion

NOW = datetime(2026, 11, 20, 6, 30, tzinfo=UTC)
FINAL_NOW = datetime(2027, 5, 2, 0, 0, 1, tzinfo=UTC)


class NeverFetch:
    def fetch(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("planning tests must not fetch")


@pytest.fixture(autouse=True)
def _no_real_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, "DECLARED_HALTS", ())
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", None)


def _stock_plan(window: data.StageWindow, limit: datetime, now: datetime) -> Any:
    policy = driver.embargo_for(limit, stage=window.stage)
    universe = Universe(
        symbols=protocol.ROSTER,
        interval=data.BASE_INTERVAL,
        markets=(StoreMarket.SPOT, StoreMarket.USDS_M),
    )
    return plan_ingestion(
        universe,
        start=window.load_start,
        end=driver._ceil_day(limit),
        now=now,
        embargo=policy,
    )


def _funding_months(plan: Any, symbol: str) -> list[datetime]:
    return sorted(
        spec.period_start
        for spec in plan.specs
        if spec.kind is ArchiveKind.FUNDING_RATE and spec.symbol == symbol
    )


def test_the_stock_plan_reproduces_plan_does_not_cover_for_the_backcast_funding() -> (
    None
):
    window = data.frozen_window(protocol.STAGE_BACKCAST)
    limit = data.read_limit(
        window, now=NOW, authorization_url=protocol.DATA_AUTHORIZATION_URL
    )
    assert limit == datetime(2021, 1, 1, 4, tzinfo=UTC)
    stock = _stock_plan(window, limit, NOW)
    # the stock planner stops at December 2020: January 2021 (holding the last 4 hours) is
    # not complete inside [load_start, ceil_day(limit)) and is skipped
    assert _funding_months(stock, "BTCUSDT")[-1] == datetime(2020, 12, 1, tzinfo=UTC)
    source = VisionArchiveSource(NeverFetch(), stock)  # type: ignore[arg-type]
    with pytest.raises(CarryIntegrityHalt) as caught:
        source.read_funding(
            StoreMarket.USDS_M,
            "BTCUSDT",
            ms_from_utc(window.load_start),
            ms_from_utc(limit),
        )
    assert caught.value.evidence["kind"] == "plan_does_not_cover"
    assert "no planned funding_rate archives cover usds_m BTCUSDT" in str(caught.value)


def test_the_tool_plan_covers_every_perp_leg_with_checksummed_monthly_funding() -> None:
    window = data.frozen_window(protocol.STAGE_BACKCAST)
    limit = data.read_limit(
        window, now=NOW, authorization_url=protocol.DATA_AUTHORIZATION_URL
    )
    plan = driver.build_plan(
        window, limit, NOW, driver.embargo_for(limit, stage=window.stage)
    )
    months = [datetime(2020, m, 1, tzinfo=UTC) for m in range(1, 13)] + [
        datetime(2021, 1, 1, tzinfo=UTC)
    ]
    for symbol in protocol.ROSTER:
        assert _funding_months(plan, symbol) == months, symbol
    funding = [s for s in plan.specs if s.kind is ArchiveKind.FUNDING_RATE]
    assert len(funding) == 13 * len(protocol.ROSTER)
    assert all(s.market is StoreMarket.USDS_M and s.interval is None for s in funding)
    assert all(s.checksum_url == f"{s.url}.CHECKSUM" for s in funding)
    assert all("/futures/um/monthly/fundingRate/" in s.url for s in funding)
    assert len({s.url for s in plan.specs}) == len(plan.specs)  # no duplicate archive
    source = VisionArchiveSource(NeverFetch(), plan)  # type: ignore[arg-type]
    chosen = source._selected(
        ArchiveKind.FUNDING_RATE,
        StoreMarket.USDS_M,
        "BTCUSDT",
        ms_from_utc(window.load_start),
        ms_from_utc(limit),
    )
    assert chosen[0].period_start == months[0] and chosen[-1].period_start == months[-1]
    # spot legs carry no funding archive and the kline plan is unchanged
    assert not any(
        s.kind is ArchiveKind.FUNDING_RATE and s.market is StoreMarket.SPOT
        for s in plan.specs
    )
    stock = _stock_plan(window, limit, NOW)
    assert {s.url for s in stock.specs} <= {s.url for s in plan.specs}


def test_a_limit_on_a_month_boundary_needs_no_extra_month_and_the_dev_plan_stays_in_bounds() -> (
    None
):
    dev = data.frozen_window(protocol.STAGE_DEV)
    limit = data.read_limit(dev, now=NOW)
    plan = driver.build_plan(
        dev, limit, NOW, driver.embargo_for(limit, stage=dev.stage)
    )
    last = _funding_months(plan, "ETHUSDT")[-1]
    assert last == datetime(
        2026, 7, 1, tzinfo=UTC
    )  # period end 2026-08-01, the embargo
    assert all(s.period_end <= data.DEFAULT_EMBARGO for s in plan.specs)
    assert _funding_months(plan, "ETHUSDT")[0] == datetime(2021, 10, 1, tzinfo=UTC)


def test_the_forward_interim_limit_is_the_last_published_funding_month() -> None:
    window = data.frozen_window(protocol.STAGE_FORWARD)
    args = SimpleNamespace(stage=protocol.STAGE_FORWARD, through=None)
    limit = driver._stage_limit(args, window, NOW, runner.MODE_INTERIM)
    # at 2026-11-20 October is complete and published, November is not
    assert driver.funding_boundary(NOW) == datetime(2026, 11, 1, tzinfo=UTC)
    assert limit == datetime(2026, 11, 1, tzinfo=UTC)
    policy = driver.embargo_for(limit, stage=window.stage)
    plan = driver.build_plan(window, limit, NOW, policy)
    assert _funding_months(plan, "BTCUSDT") == [
        datetime(2026, m, 1, tzinfo=UTC) for m in range(2, 11)
    ]
    # the funding month of every row read is planned; the unpublished one is not
    assert max(
        s.period_end for s in plan.specs if s.kind is ArchiveKind.FUNDING_RATE
    ) == (datetime(2026, 11, 1, tzinfo=UTC))


def test_a_through_inside_the_published_months_is_allowed_and_one_beyond_refuses() -> (
    None
):
    window = data.frozen_window(protocol.STAGE_FORWARD)
    ok = SimpleNamespace(
        stage=protocol.STAGE_FORWARD, through=datetime(2026, 10, 12, 8, tzinfo=UTC)
    )
    assert driver._stage_limit(ok, window, NOW, runner.MODE_INTERIM) == ok.through
    # in the unpublished current month: the funding of November does not exist yet
    late = SimpleNamespace(
        stage=protocol.STAGE_FORWARD, through=datetime(2026, 11, 10, tzinfo=UTC)
    )
    with pytest.raises(data.StageEmbargoError, match="funding archive.*not published"):
        driver._stage_limit(late, window, NOW, runner.MODE_INTERIM)


def test_the_final_scoring_is_refused_until_the_funding_of_its_last_month_is_published() -> (
    None
):
    window = data.frozen_window(protocol.STAGE_FORWARD)
    args = SimpleNamespace(stage=protocol.STAGE_FORWARD, through=None)
    # 2027-05-02 is the earliest scoring instant, but May 2027 funding is published only
    # after May ends and the publication lag: refuse rather than invent funding
    with pytest.raises(data.StageEmbargoError, match="funding archive.*not published"):
        driver._stage_limit(args, window, FINAL_NOW, runner.MODE_DECISION)
    published = datetime(2027, 6, 8, tzinfo=UTC)
    assert driver.funding_boundary(published) == datetime(2027, 6, 1, tzinfo=UTC)
    limit = driver._stage_limit(args, window, published, runner.MODE_DECISION)
    assert limit == window.last_row_close
    plan = driver.build_plan(
        window, limit, published, driver.embargo_for(limit, stage=window.stage)
    )
    assert _funding_months(plan, "SOLUSDT")[-1] == datetime(2027, 5, 1, tzinfo=UTC)
    # the FORWARD embargo admits the funding month holding the limit, and no further
    policy = driver.embargo_for(limit, stage=window.stage)
    assert policy.sealed_start == datetime(2027, 6, 1, tzinfo=UTC)
    assert max(s.period_end for s in plan.specs) == datetime(2027, 6, 1, tzinfo=UTC)
    boundary_limit = datetime(
        2026, 11, 1, tzinfo=UTC
    )  # on a month boundary: not widened
    assert (
        driver.embargo_for(boundary_limit, stage=window.stage).sealed_start
        == boundary_limit
    )
    # a day earlier the May archive is still within its publication lag
    with pytest.raises(data.StageEmbargoError, match="not published"):
        driver._stage_limit(
            args, window, published - timedelta(days=2), runner.MODE_DECISION
        )


def test_build_plan_refuses_an_unpublished_funding_month_before_any_read() -> None:
    window = data.frozen_window(protocol.STAGE_FORWARD)
    limit = datetime(2026, 11, 20, tzinfo=UTC)
    policy = driver.embargo_for(limit, stage=window.stage)
    with pytest.raises(data.StageEmbargoError, match="funding archive.*not published"):
        driver.build_plan(window, limit, NOW, policy)
    assert (
        driver.require_funding_published(datetime(2026, 11, 1, tzinfo=UTC), NOW) is None
    )
