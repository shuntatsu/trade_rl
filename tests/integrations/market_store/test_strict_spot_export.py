"""An opt-in export must bind each stored row to its original admission source."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from tests.integrations.market_store.support import (
    NOW,
    FakeFetcher,
    stored_exchange_info,
    zip_csv,
)
from tests.integrations.market_store.test_binance_placeholder_v3 import (
    END,
    START,
    SYMBOL,
    rows,
)
from trade_rl.artifacts import content_digest
from trade_rl.integrations.market_store.checksum import verify_archive
from trade_rl.integrations.market_store.contracts import (
    StoreMarket,
    Universe,
    ms_from_utc,
)
from trade_rl.integrations.market_store.export import export_market_dataset
from trade_rl.integrations.market_store.ingest import RunCounts, run_ingestion
from trade_rl.integrations.market_store.parse import (
    StoreKlineAdmission,
    parse_kline_archive,
)
from trade_rl.integrations.market_store.plan import plan_ingestion
from trade_rl.integrations.market_store.rules import parse_symbol_rules

STRICT = "strict_spot_lineage_v2"
COMMIT = "a" * 40


def populate(
    store: Any, *, defect: str | None = None, member: str = "data.csv"
) -> None:
    plan = plan_ingestion(
        Universe(symbols=(SYMBOL,), markets=(StoreMarket.SPOT,)),
        start=START,
        end=END,
        now=NOW,
    )
    fetcher = FakeFetcher()
    (spec,) = plan.specs
    fetcher.add(spec.url, zip_csv(rows(), member=member))
    if defect is None:
        report = run_ingestion(
            store,
            fetcher,
            plan,
            code_commit=COMMIT,
            command_line="pytest",
            clock=lambda: NOW,
        )
        assert report.counts.rows_inserted == 739
    else:
        # Construct the historical or unqualified source as it was originally
        # recorded. Finished runs and archive evidence must remain immutable.
        request = plan.to_dict()
        if defect != "legacy":
            request["new_kline_admission"] = StoreKlineAdmission.STRICT_NATIVE_V2.value
        if defect == "wrong-plan":
            request["archives"] = []
        run_id = store.begin_run(
            code_commit="unknown" if defect == "unknown-commit" else COMMIT,
            command_line="pytest original fixture",
            request=request,
            embargo=plan.embargo,
        )
        payload = fetcher.files[spec.url]
        parsed = parse_kline_archive(
            payload,
            spec,
            embargo=plan.embargo,
            admission=StoreKlineAdmission.STRICT_NATIVE_V2,
        )
        with store.transaction():
            instrument = store.ensure_instrument(spec.market, spec.symbol)
            source_id = store.write_source(
                run_id=run_id,
                spec=spec,
                instrument_id=instrument,
                http_status=200,
                fetched_at=NOW,
                evidence=verify_archive(
                    payload, fetcher.files[spec.checksum_url], filename=spec.filename
                ),
                row_count=len(parsed.rows),
            )
            assert store.insert_klines(instrument, "1h", source_id, parsed.rows) == 739
        failed = defect == "failed"
        store.finish_run(
            run_id,
            status="failed" if failed else "succeeded",
            counts=RunCounts(
                archives_planned=1, archives_ingested=1, rows_inserted=739
            ),
            error="fixture: later archive failed" if failed else None,
        )
    info = stored_exchange_info(StoreMarket.SPOT)
    store.store_exchange_info(
        info,
        parse_symbol_rules(
            info.raw_payload, market=StoreMarket.SPOT, symbols=(SYMBOL,)
        ),
        run_id=None,
    )


def export(store: Any, **kwargs: Any) -> Any:
    params = dict(
        market=StoreMarket.SPOT,
        symbols=(SYMBOL,),
        interval="1h",
        start_time=START,
        end_time=END,
        feature_timeframes=(),
    )
    params.update(kwargs)
    return export_market_dataset(store, **params)


def test_strict_export_binds_each_row_without_changing_v1(pg_store: Any) -> None:
    populate(pg_store)
    legacy = export(pg_store)
    strict = export(pg_store, export_policy=STRICT)
    assert legacy.receipt()["schema_version"] == "market_store_export_v1"
    assert strict.receipt()["schema_version"] == "market_store_export_strict_spot_v2"
    assert strict.evidence["schema_version"] == strict.receipt()["schema_version"]
    assert strict.dataset.dataset_id != legacy.dataset.dataset_id
    assert strict.dataset.recomputed_dataset_id() == strict.dataset.dataset_id
    assert strict.receipt() == export(pg_store, export_policy=STRICT).receipt()
    assert legacy.receipt() == export(pg_store).receipt()
    pg_store.connection.execute("SET TIME ZONE 'Asia/Tokyo'")
    assert strict.receipt() == export(pg_store, export_policy=STRICT).receipt()
    assert legacy.receipt() == export(pg_store).receipt()
    for name, arr in legacy.dataset.identity_arrays().items():
        np.testing.assert_array_equal(arr, strict.dataset.identity_arrays()[name])
    lineage = strict.evidence["kline_lineage"][SYMBOL]
    assert lineage["rows"] == 739
    (source,) = lineage["sources"]
    assert source["rows"] == 739
    evidence = source["evidence"]
    assert evidence["code_commit"] == COMMIT
    assert evidence["admission"] == "store_native_close_v2"
    assert source["sha256"] == content_digest(evidence)
    with pg_store.connection.cursor() as cur:
        cur.execute(
            "SELECT open_time, s.url, s.sha256, s.checksum_file_sha256 "
            "FROM klines k JOIN sources s USING (source_id) ORDER BY open_time"
        )
        readback = cur.fetchall()
    assert all(
        r[1:]
        == (
            evidence["url"],
            evidence["archive_sha256"],
            evidence["checksum_file_sha256"],
        )
        for r in readback
    )
    assert lineage["row_source_sha256"] == content_digest(
        [[ms_from_utc(r[0]), source["sha256"]] for r in readback]
    )


@pytest.mark.parametrize("defect", ["legacy", "failed", "unknown-commit", "wrong-plan"])
def test_strict_export_refuses_unqualified_source_runs(
    pg_store: Any, defect: str
) -> None:
    populate(pg_store, defect=defect)
    assert export(pg_store).dataset.n_bars == 744
    with pytest.raises(ValueError, match="strict spot lineage"):
        export(pg_store, export_policy=STRICT)


def test_strict_export_refuses_inconsistent_row_source_scope(pg_store: Any) -> None:
    populate(pg_store)
    # An append-only row may otherwise point at an unrelated instrument's source.
    store = pg_store
    source_id = store.connection.execute("SELECT source_id FROM sources").fetchone()[0]
    instrument = store.ensure_instrument(StoreMarket.SPOT, "ETHUSDT")
    raw = store.read_klines(
        StoreMarket.SPOT, SYMBOL, "1h", ms_from_utc(START), ms_from_utc(END)
    )[0]
    assert store.insert_klines(instrument, "1h", source_id, [raw]) == 1
    with pytest.raises(ValueError, match="strict spot lineage"):
        export(store, symbols=("ETHUSDT",), export_policy=STRICT)


def test_strict_scope_is_explicit_before_store_reads() -> None:
    class NoReads:
        def __getattribute__(self, name: str) -> Any:
            pytest.fail("unsupported strict scope accessed store: " + name)

    for kwargs in (
        {"market": StoreMarket.USDS_M},
        {"interval": "4h"},
        {"feature_timeframes": ("4h",)},
        {"end_time": datetime(2026, 9, 1, tzinfo=UTC)},
    ):
        with pytest.raises(ValueError, match="strict spot lineage"):
            export(NoReads(), export_policy=STRICT, **kwargs)


def test_strict_identity_changes_for_same_rows_from_distinct_archive_bytes(
    pg_store: Any, postgres_url: str
) -> None:
    from trade_rl.integrations.market_store.store import PostgresMarketStore

    populate(pg_store)
    first = export(pg_store, export_policy=STRICT)
    legacy_first = export(pg_store)
    schema = "strict_" + uuid.uuid4().hex[:16]
    with pg_store.connection.cursor() as cur:
        cur.execute(f'CREATE SCHEMA "{schema}"')
    try:
        with PostgresMarketStore.open(postgres_url, schema=schema) as other:
            other.migrate()
            populate(other, member="another.csv")
            assert export(other).receipt() == legacy_first.receipt()
            second = export(other, export_policy=STRICT)
            assert second.dataset.dataset_id != first.dataset.dataset_id
            a = first.evidence["kline_lineage"][SYMBOL]
            b = second.evidence["kline_lineage"][SYMBOL]
            assert a["row_value_sha256"] == b["row_value_sha256"]
            assert a["row_source_sha256"] != b["row_source_sha256"]
    finally:
        with pg_store.connection.cursor() as cur:
            cur.execute(f'DROP SCHEMA "{schema}" CASCADE')


def test_exact_numeric_identity_does_not_round_decimal_context() -> None:
    from trade_rl.integrations.market_store.lineage import _exact_value

    a = Decimal("1.1234567890123456789012345678901")
    b = Decimal("1.1234567890123456789012345678902")
    assert float(a) == float(b)
    assert _exact_value(a) == str(a)
    assert _exact_value(b) == str(b)
    assert _exact_value(a) != _exact_value(b)
    assert _exact_value(Decimal("1.2300")) == "1.23"
    assert _exact_value(Decimal("-0.000")) == "0"


def test_direct_lineage_reader_rejects_out_of_scope_ranges(pg_store: Any) -> None:
    from trade_rl.integrations.market_store.embargo import FS1_WARMUP_START

    bound = ms_from_utc(FS1_WARMUP_START)
    with pytest.raises(ValueError, match="pre-warm-up"):
        pg_store.read_strict_spot_klines(SYMBOL, bound, bound + 3_600_000)
    with pytest.raises(ValueError, match="aligned"):
        pg_store.read_strict_spot_klines(
            SYMBOL, ms_from_utc(START) + 1, ms_from_utc(END)
        )


def test_strict_gap_policy_still_refuses_the_missing_hours(pg_store: Any) -> None:
    from trade_rl.integrations.binance.dataset import BinanceGapPolicy

    populate(pg_store)
    with pytest.raises(ValueError, match="complete and exactly regular"):
        export(pg_store, export_policy=STRICT, gap_policy=BinanceGapPolicy.STRICT)


def test_cli_selects_policy_without_changing_legacy_default() -> None:
    from trade_rl.integrations.market_store.cli import build_parser

    argv = [
        "export",
        "--market",
        "spot",
        "--start",
        "2020-12-01",
        "--end",
        "2021-01-01",
        "--output-dir",
        "never-written",
    ]
    parser = build_parser()
    assert parser.parse_args(argv).export_policy == "legacy_v1"
    parsed = parser.parse_args([*argv, "--export-policy", STRICT])
    assert parsed.export_policy == STRICT and parsed.feature_timeframes is None
