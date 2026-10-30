from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from tests.integrations.market_store.support import (
    HOUR_MS,
    NOW,
    SYMBOLS,
    MemoryStore,
    archive_bytes,
    fetcher_for,
    file_backed_build,
    stored_exchange_info,
)
from trade_rl.integrations.market_store.contracts import (
    StoreMarket,
    Universe,
    ms_from_utc,
)
from trade_rl.integrations.market_store.embargo import (
    SEALED_WINDOW_START,
    EmbargoPolicy,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.export import (
    StoreCoverageError,
    export_market_dataset,
)
from trade_rl.integrations.market_store.ingest import run_ingestion
from trade_rl.integrations.market_store.plan import plan_ingestion

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 2, 1, tzinfo=UTC)


def populated(market: StoreMarket, *, gap: bool = False):  # type: ignore[no-untyped-def]
    plan = plan_ingestion(
        Universe(symbols=SYMBOLS, markets=(market,)), start=START, end=END, now=NOW
    )
    fetcher = fetcher_for(plan)
    if gap:
        spec = next(s for s in plan.specs if s.symbol == "ETHUSDT" and s.interval)
        fetcher.add(
            spec.url,
            archive_bytes(
                spec,
                skip=frozenset(
                    {
                        ms_from_utc(START) + 50 * HOUR_MS,
                        ms_from_utc(START) + 51 * HOUR_MS,
                    }
                ),
            ),
        )
    store = MemoryStore()
    run_ingestion(
        store, fetcher, plan, code_commit="t", command_line="t", clock=lambda: NOW
    )
    store.exchange_info.append(stored_exchange_info(market))
    return store, fetcher, plan


@pytest.mark.parametrize("market", [StoreMarket.USDS_M, StoreMarket.SPOT])
@pytest.mark.parametrize("gap", [False, True])
def test_store_backed_build_equals_file_backed_build(
    tmp_path: Path, market: StoreMarket, gap: bool
) -> None:
    store, fetcher, plan = populated(market, gap=gap)
    export = export_market_dataset(
        store,
        market=market,
        symbols=SYMBOLS,
        interval="1h",
        start_time=START,
        end_time=END,
    )
    file_build = file_backed_build(
        tmp_path,
        fetcher,
        plan,
        market=market,
        symbols=SYMBOLS,
        start=START,
        end=END,
        evidence=export.evidence,
    )
    left, right = export.dataset, file_build.dataset
    assert left.dataset_id == right.dataset_id
    assert left.symbols == right.symbols
    assert left.feature_names == right.feature_names
    assert np.array_equal(left.timestamps, right.timestamps)
    for name in ("close", "open", "high", "low", "volume", "funding_rate"):
        assert np.array_equal(left.resolved_array(name), right.resolved_array(name))
    if market is StoreMarket.USDS_M:
        assert left.resolved_array("funding_event_count").sum() > 0
    if gap:
        assert export.build.gap_report == file_build.gap_report
        assert any(any(counts.values()) for counts in export.build.gap_report.values())
    assert export.build.feature_timeframes == file_build.feature_timeframes


def test_export_identity_is_deterministic_and_tracks_content() -> None:
    store, _, _ = populated(StoreMarket.USDS_M)
    kwargs = dict(
        market=StoreMarket.USDS_M,
        symbols=SYMBOLS,
        interval="1h",
        start_time=START,
        end_time=END,
    )
    first = export_market_dataset(store, **kwargs)  # type: ignore[arg-type]
    second = export_market_dataset(store, **kwargs)  # type: ignore[arg-type]
    assert first.dataset.dataset_id == second.dataset.dataset_id
    assert first.receipt() == second.receipt()

    other_store, _, _ = populated(StoreMarket.USDS_M, gap=True)
    other = export_market_dataset(other_store, **kwargs)  # type: ignore[arg-type]
    assert other.dataset.dataset_id != first.dataset.dataset_id

    shorter = export_market_dataset(
        store,
        **{**kwargs, "end_time": datetime(2024, 1, 20, tzinfo=UTC)},  # type: ignore[arg-type]
    )
    assert shorter.dataset.dataset_id != first.dataset.dataset_id


def test_pinning_the_rules_snapshot_selects_it() -> None:
    store, _, _ = populated(StoreMarket.USDS_M)
    original = stored_exchange_info(StoreMarket.USDS_M)
    newer = type(original)(
        market=original.market,
        raw_payload=original.raw_payload.replace(b"0.10", b"0.20"),
        raw_payload_sha256="0" * 64,
        source_uri="fixture://newer",
        retrieved_at=datetime(2026, 10, 30, tzinfo=UTC),
    )
    store.exchange_info.append(newer)
    pinned = export_market_dataset(
        store,
        market=StoreMarket.USDS_M,
        symbols=SYMBOLS,
        interval="1h",
        start_time=START,
        end_time=END,
        rules_snapshot_sha256=original.raw_payload_sha256,
    )
    assert pinned.exchange_info_sha256 == original.raw_payload_sha256
    latest = export_market_dataset(
        store,
        market=StoreMarket.USDS_M,
        symbols=SYMBOLS,
        interval="1h",
        start_time=START,
        end_time=END,
    )
    assert latest.exchange_info_sha256 == "0" * 64


def test_export_refuses_the_sealed_window_and_empty_coverage() -> None:
    store, _, _ = populated(StoreMarket.USDS_M)
    with pytest.raises(SealedWindowEmbargoError):
        export_market_dataset(
            store,
            market=StoreMarket.USDS_M,
            symbols=SYMBOLS,
            interval="1h",
            start_time=START,
            end_time=datetime(2026, 11, 2, tzinfo=UTC),
        )
    assert EmbargoPolicy.enforced().start == SEALED_WINDOW_START
    with pytest.raises(StoreCoverageError):
        export_market_dataset(
            store,
            market=StoreMarket.USDS_M,
            symbols=("SOLUSDT",),
            interval="1h",
            start_time=START,
            end_time=END,
        )
