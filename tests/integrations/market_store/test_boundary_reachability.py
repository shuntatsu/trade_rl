"""The planner never plans, and run_ingestion never fetches, a closed FS1 span archive.

Scope: every archive the planner emits and every URL a correctly described spec leads
the ingester to request stays before the closed span. A hand-built spec whose declared
metadata disagrees with its URL is refused from the URL (``test_url_gate.py``), and the
fetcher repeats the URL-derived check on its own before any request.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tests.integrations.market_store.support import (
    FakeFetcher,
    MemoryStore,
    archive_bytes,
    fetcher_for,
)
from trade_rl.integrations.market_store.archive_url import ArchiveSpecMismatchError
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    PeriodKind,
    StoreMarket,
    Universe,
)
from trade_rl.integrations.market_store.embargo import (
    FS1_WARMUP_START,
    SEALED_WINDOW_START,
    EmbargoPolicy,
    Fs1WarmupSpanError,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.ingest import run_ingestion
from trade_rl.integrations.market_store.plan import IngestionPlan, plan_ingestion

URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-77"
UNIVERSE = Universe(
    symbols=("BTCUSDT",), markets=(StoreMarket.USDS_M, StoreMarket.SPOT)
)
CONSTRUCTIONS = {
    "none": None,
    "plain": EmbargoPolicy(),
    "enforced": EmbargoPolicy.enforced(),
    "warmup": EmbargoPolicy.with_warmup(URL),
}


def day(year: int, month: int, number: int = 1) -> datetime:
    return datetime(year, month, number, tzinfo=UTC)


def closed_from(policy: EmbargoPolicy | None) -> datetime:
    return SEALED_WINDOW_START if policy and policy.warmup_allowed else FS1_WARMUP_START


@pytest.mark.parametrize("name", list(CONSTRUCTIONS))
def test_planner_never_produces_a_closed_span_archive(name: str) -> None:
    policy = CONSTRUCTIONS[name]
    limit = closed_from(policy)
    produced = 0
    for start_offset in range(0, 240, 17):
        start = day(2026, 3) + timedelta(days=start_offset)
        for end_offset in range(1, 400, 13):
            end = start + timedelta(days=end_offset)
            for now in (day(2026, 10, 30), day(2026, 12, 15), day(2027, 6, 1)):
                try:
                    plan = plan_ingestion(
                        UNIVERSE, start=start, end=end, now=now, embargo=policy
                    )
                except (Fs1WarmupSpanError, SealedWindowEmbargoError):
                    assert end > limit
                    continue
                for spec in plan.specs:
                    produced += 1
                    assert spec.period_end <= limit, (name, spec.url)
                    assert spec.period_start < limit, (name, spec.url)
                for skipped in plan.skipped:
                    assert skipped.period_end > skipped.period_start
    assert produced > 0


@pytest.mark.parametrize("name", ["none", "enforced", "warmup"])
def test_fetcher_never_requests_a_closed_span_url(name: str) -> None:
    policy = CONSTRUCTIONS[name]
    limit = closed_from(policy)
    plan = plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.SPOT,)),
        start=day(2026, 6),
        end=limit,
        now=day(2027, 6, 1),
        embargo=policy,
    )
    fetcher, store = fetcher_for(plan), MemoryStore()
    run_ingestion(store, fetcher, plan, code_commit="t", command_line="t")
    assert fetcher.requested
    requested = {url.removesuffix(".CHECKSUM") for url in fetcher.requested}
    assert requested == {spec.url for spec in plan.specs}
    assert all(spec.period_end <= limit for spec in plan.specs)
    banned = ("2026-08", "2026-09", "2026-10") if name != "warmup" else ()
    assert not any(token in url for url in requested for token in banned)
    assert not any("2026-11" in url or "2026-12" in url for url in requested)


def real_spec(month: int, *, declared: tuple[datetime, datetime]) -> ArchiveSpec:
    """A real monthly Vision URL whose declared period straddles a boundary."""

    url = (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/"
        f"BTCUSDT-1h-2026-{month:02d}.zip"
    )
    return ArchiveSpec(
        kind=ArchiveKind.KLINES,
        market=StoreMarket.SPOT,
        symbol="BTCUSDT",
        interval="1h",
        period_kind=PeriodKind.MONTHLY,
        period_start=declared[0],
        period_end=declared[1],
        url=url,
        checksum_url=f"{url}.CHECKSUM",
    )


@pytest.mark.parametrize("name", ["none", "enforced"])
def test_a_straddling_declaration_in_the_warmup_span_is_refused_without_authorization(
    name: str,
) -> None:
    policy = CONSTRUCTIONS[name] or EmbargoPolicy.enforced()
    spec = real_spec(7, declared=(day(2026, 7, 15), day(2026, 8, 15)))
    fetcher, store = FakeFetcher(), MemoryStore()
    fetcher.add(spec.url, archive_bytes(spec))
    plan = IngestionPlan(
        UNIVERSE, spec.period_start, spec.period_end, policy, (spec,), ()
    )
    with pytest.raises((ArchiveSpecMismatchError, Fs1WarmupSpanError)):
        run_ingestion(store, fetcher, plan, code_commit="t", command_line="t")
    assert fetcher.requested == [] and store.klines == {}
    with pytest.raises(Fs1WarmupSpanError):
        plan_ingestion(
            UNIVERSE,
            start=day(2026, 7, 20),
            end=day(2026, 8, 10),
            now=day(2026, 12, 15),
            embargo=CONSTRUCTIONS[name],
        )


def test_a_straddling_declaration_at_the_sealed_start_is_refused_even_with_authorization() -> (
    None
):
    spec = real_spec(10, declared=(day(2026, 10, 15), day(2026, 11, 15)))
    fetcher, store = FakeFetcher(), MemoryStore()
    fetcher.add(spec.url, archive_bytes(spec))
    policy = EmbargoPolicy.with_warmup(URL)
    plan = IngestionPlan(
        UNIVERSE, spec.period_start, spec.period_end, policy, (spec,), ()
    )
    with pytest.raises((ArchiveSpecMismatchError, SealedWindowEmbargoError)):
        run_ingestion(store, fetcher, plan, code_commit="t", command_line="t")
    assert fetcher.requested == [] and store.klines == {}
    with pytest.raises(SealedWindowEmbargoError):
        plan_ingestion(
            UNIVERSE,
            start=day(2026, 10, 20),
            end=day(2026, 11, 10),
            now=day(2027, 6, 1),
            embargo=policy,
        )
