"""Materialize and transactionally publish the maintained real-data cache."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence, cast

from trade_rl.integrations.postgres_market_tables import (
    UNIVERSAL_202411_202607_CACHE_ID,
    UNIVERSAL_202411_202607_TABLES,
)
from trade_rl.integrations.postgres_native_cache_publisher import (
    NativeCacheConnection,
    publish_native_cache,
)
from trade_rl.integrations.postgres_universal_source import (
    MAINTAINED_SYMBOLS,
    UniversalSourceConnection,
    UniversalSourceScope,
    load_postgres_universal_source,
)
from trade_rl.workflows.native_indicator_materializer import (
    NativeCacheBuild,
    build_native_indicator_cache,
    combine_native_indicator_builds,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish the maintained 15-symbol PostgreSQL native cache"
    )
    parser.add_argument("--postgres-url", required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--cache-id", default=UNIVERSAL_202411_202607_CACHE_ID)
    parser.add_argument("--start", default="2024-11-13T00:00:00Z")
    parser.add_argument("--end", default="2026-07-05T00:00:00Z")
    return parser


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("cache timestamps must include a timezone")
    return parsed


def _write_report(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_source_streaming(
    connection: UniversalSourceConnection,
    *,
    scope: UniversalSourceScope,
) -> NativeCacheBuild:
    """Read and build one symbol at a time to bound raw-source memory."""

    builds: list[NativeCacheBuild] = []
    for symbol in scope.symbols:
        symbol_scope = UniversalSourceScope(
            symbols=(symbol,),
            start=scope.start,
            end=scope.end,
            source=scope.source,
        )
        source = load_postgres_universal_source(connection, scope=symbol_scope)
        builds.append(build_native_indicator_cache(source, scope=symbol_scope))
    return combine_native_indicator_builds(builds, scope=scope)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cache_id != UNIVERSAL_202411_202607_CACHE_ID:
        raise ValueError("cache-id must match the maintained immutable generation")
    scope = UniversalSourceScope(
        symbols=MAINTAINED_SYMBOLS,
        start=_datetime(args.start),
        end=_datetime(args.end),
    )
    import psycopg

    with psycopg.connect(args.postgres_url) as connection:
        build = _build_source_streaming(
            cast(UniversalSourceConnection, connection), scope=scope
        )
        published = publish_native_cache(
            cast(NativeCacheConnection, connection),
            build,
            tables=UNIVERSAL_202411_202607_TABLES,
        )
    report_path = (
        args.report_root / build.manifest.cache_id / "intermediate-data-report.json"
    )
    report_payload = {
        "cache_id": build.manifest.cache_id,
        "manifest_digest": build.manifest.digest,
        "report": asdict(build.report),
        "report_digest": build.report.digest,
        "schema_version": "native_indicator_intermediate_report_v1",
        "tables": asdict(UNIVERSAL_202411_202607_TABLES),
    }
    _write_report(report_path, report_payload)
    print(
        json.dumps(
            {
                "artifact_count": published.artifact_count,
                "cache_id": published.cache_id,
                "funding_row_count": published.funding_row_count,
                "kline_row_count": published.kline_row_count,
                "manifest_digest": published.manifest_digest,
                "report_path": str(report_path.resolve()),
                "tables": asdict(published.tables),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
