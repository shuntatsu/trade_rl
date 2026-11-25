"""Read-only provenance validation for newly admitted, direct spot 1h rows.

The database records an ingestion claim and its original archive digests. This
does not attest the ingest executable or historical exchange-rule availability.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from trade_rl.artifacts import content_digest
from trade_rl.integrations.market_store.archive_url import verify_spec_matches_url
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    KlineRow,
    PeriodKind,
    StoreMarket,
    ms_from_utc,
)
from trade_rl.integrations.market_store.embargo import EmbargoPolicy
from trade_rl.integrations.market_store.parse import StoreKlineAdmission, validate_kline


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _exact_value(value: object) -> str | int:
    if isinstance(value, Decimal):
        # Decimal.normalize() uses the ambient context and can round an exact
        # PostgreSQL NUMERIC. Strip scale without arithmetic instead.
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if value.is_zero() else text
    if type(value) is int:
        return value
    raise ValueError("strict spot lineage requires exact stored values")


def _source_evidence(source: Mapping[str, Any], *, symbol: str) -> dict[str, Any]:
    if (
        source["scope_matches"] is not True
        or source["plan_matches"] is not True
        or source["source_status"] != "ingested"
        or source["http_status"] != 200
        or source["run_status"] != "succeeded"
        or source["admission"] != StoreKlineAdmission.STRICT_NATIVE_V2.value
        or not isinstance(source["code_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", source["code_commit"]) is None
        or source["code_commit"] == "0" * 40
        or not _digest(source["archive_sha256"])
        or not _digest(source["checksum_file_sha256"])
        or source["checksum_line_sha256"] != source["archive_sha256"]
        or type(source["size_bytes"]) is not int
        or source["size_bytes"] <= 0
        or type(source["archive_rows"]) is not int
        or source["archive_rows"] <= 0
    ):
        raise ValueError("strict spot lineage requires a qualified original source run")
    spec = ArchiveSpec(
        kind=ArchiveKind.KLINES,
        market=StoreMarket.SPOT,
        symbol=symbol,
        interval="1h",
        period_kind=PeriodKind(source["period_kind"]),
        period_start=source["period_start"].astimezone(UTC),
        period_end=source["period_end"].astimezone(UTC),
        url=source["url"],
        checksum_url=source["checksum_url"],
    )
    try:
        verify_spec_matches_url(spec)
        EmbargoPolicy.enforced().check_archive_end(spec.period_end, url=spec.url)
    except ValueError as error:
        raise ValueError("strict spot lineage source scope is invalid") from error
    return {
        **spec.to_dict(),
        "archive_sha256": source["archive_sha256"],
        "checksum_file_sha256": source["checksum_file_sha256"],
        "checksum_line_sha256": source["checksum_line_sha256"],
        "size_bytes": source["size_bytes"],
        "archive_rows": source["archive_rows"],
        "fetched_at": _utc(source["fetched_at"]),
        "code_commit": source["code_commit"],
        "admission": source["admission"],
    }


def bind_strict_spot_rows(
    joined: Sequence[tuple[KlineRow, Mapping[str, Any]]],
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> tuple[list[KlineRow], dict[str, Any]]:
    """Validate every joined row before building one stable content-bound receipt."""
    rows: list[KlineRow] = []
    sources: dict[int, tuple[str, dict[str, Any]]] = {}
    selected: dict[str, int] = {}
    by_digest: dict[str, dict[str, Any]] = {}
    assignment: list[list[int | str]] = []
    previous = start_ms - 1
    for row, source in joined:
        source_id = source["source_id"]
        if source_id not in sources:
            evidence = _source_evidence(source, symbol=symbol)
            key = content_digest(evidence)
            sources[source_id] = key, evidence
            by_digest[key] = evidence
        key, evidence = sources[source_id]
        if (
            source["scope_matches"] is not True
            or source["open_time"].microsecond % 1000
            or source["close_time"].microsecond % 1000
            or not start_ms <= row.open_time_ms < end_ms
            or row.open_time_ms <= previous
            or row.open_time_ms % 3_600_000
            or row.close_time_ms != row.open_time_ms + 3_599_999
            or not ms_from_utc(source["period_start"])
            <= row.open_time_ms
            < ms_from_utc(source["period_end"])
            or not all(
                value.is_finite()
                for value in (
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    row.volume,
                    row.quote_volume,
                    row.taker_buy_base_volume,
                    row.taker_buy_quote_volume,
                )
            )
        ):
            raise ValueError("strict spot lineage row clock or source scope is invalid")
        validate_kline(row, url=evidence["url"])
        previous = row.open_time_ms
        selected[key] = selected.get(key, 0) + 1
        if selected[key] > evidence["archive_rows"]:
            raise ValueError("strict spot lineage exceeds original archive row count")
        rows.append(row)
        assignment.append([row.open_time_ms, key])
    return rows, {
        "schema_version": "store_spot_kline_lineage_v1",
        "rows": len(rows),
        "row_source_sha256": content_digest(assignment),
        "row_value_sha256": content_digest(
            [
                [
                    _exact_value(v)
                    for v in (row.open_time_ms, *row.comparable().values())
                ]
                for row in rows
            ]
        ),
        "sources": [
            {"sha256": key, "rows": selected[key], "evidence": by_digest[key]}
            for key in sorted(selected)
        ],
    }
