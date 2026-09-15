from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, datetime

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.spot_aggtrades_source_prereg import (
    canonical_spot_aggtrades_source_protocol,
)
from trade_rl.evaluation.experiments.bootstrap.spot_aggtrades_structural_validator import (
    build_structural_report,
    canonical_structural_report_bytes,
    validate_archive_bytes,
)

PROTOCOL = canonical_spot_aggtrades_source_protocol()


def _valid_entry(symbol: str, date: str) -> dict[str, object]:
    url = PROTOCOL.url_template.format(symbol=symbol, date=date)
    name = url.rsplit("/", 1)[-1]
    timestamp = int(
        datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000
    )
    digest = "a" * 64
    return {
        "symbol": symbol,
        "date": date,
        "url": url,
        "checksum_url": url + PROTOCOL.checksum_suffix,
        "archive_available": True,
        "raw_zip_size_bytes": 1,
        "raw_zip_sha256": digest,
        "checksum_available": True,
        "checksum_text": f"{digest}  {name}",
        "checksum_digest": digest,
        "checksum_verified": True,
        "member_name": name.removesuffix(".zip") + ".csv",
        "header_present": True,
        "normalized_schema": list(PROTOCOL.expected_header),
        "total_row_count": 1,
        "usable_row_count": 1,
        "provider_invalid_sentinel_count": 0,
        "malformed_row_count": 0,
        "first_usable_aggregate_id": 1,
        "last_usable_aggregate_id": 1,
        "first_usable_event_timestamp_ms": timestamp,
        "last_usable_event_timestamp_ms": timestamp,
        "usable_ids_strictly_increasing_unique": True,
        "usable_timestamps_nondecreasing": True,
        "timestamps_inside_requested_utc_date": True,
        "schema_valid": True,
    }


def _report() -> dict[str, object]:
    entries = [
        _valid_entry(symbol, date)
        for symbol in PROTOCOL.symbols
        for date in PROTOCOL.dates
    ]
    return build_structural_report(
        PROTOCOL,
        entries,
        protocol_head="9" * 40,
        protocol_seal_run_id=1,
        protocol_seal_artifact_id=2,
        protocol_seal_artifact_api_digest="a" * 64,
        protocol_fresh_artifact_id=3,
        protocol_fresh_artifact_api_digest="b" * 64,
        validator_head="8" * 40,
        validator_verification_run_id=4,
    )


def _redigest(report: dict[str, object]) -> None:
    payload = dict(report)
    payload.pop("content_digest")
    report["content_digest"] = content_digest(payload)


def test_nested_archive_report_cannot_gain_economic_field_even_with_new_digest() -> (
    None
):
    report = _report()
    entries = [dict(item) for item in report["archive_reports"]]  # type: ignore[arg-type]
    entries[0]["price_mean"] = 123.0
    report["archive_reports"] = entries
    _redigest(report)

    with pytest.raises(ValueError, match="archive|structural|field"):
        canonical_structural_report_bytes(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("available_archive_count", 19),
        ("checksum_verified_count", 19),
        ("structurally_valid_available_archive_count", 19),
        ("status", PROTOCOL.partial_status),
    ],
)
def test_summary_counts_and_status_are_recomputed_not_trusted(
    field: str,
    value: object,
) -> None:
    report = _report()
    report[field] = value
    _redigest(report)

    with pytest.raises(ValueError, match="count|status|structural|canonical"):
        canonical_structural_report_bytes(report)


def test_archive_report_order_and_roster_are_revalidated() -> None:
    report = _report()
    entries = [dict(item) for item in report["archive_reports"]]  # type: ignore[arg-type]
    entries[0], entries[1] = entries[1], entries[0]
    report["archive_reports"] = entries
    _redigest(report)

    with pytest.raises(ValueError, match="roster|order|archive"):
        canonical_structural_report_bytes(report)


def test_noncanonical_header_is_not_silently_treated_as_headerless_data() -> None:
    symbol = "BTCUSDT"
    date = "2021-01-15"
    url = PROTOCOL.url_template.format(symbol=symbol, date=date)
    archive_name = url.rsplit("/", 1)[-1]
    member_name = archive_name.removesuffix(".zip") + ".csv"
    wrong_header = (
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
        "timestamp,buyer_is_maker,best_price_match\n"
    )
    row = "1,100,2,10,10,1610668800000,False,True\n"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member_name, wrong_header + row)
    payload = stream.getvalue()
    checksum = f"{hashlib.sha256(payload).hexdigest()}  {archive_name}\n".encode()

    entry = validate_archive_bytes(
        PROTOCOL,
        symbol=symbol,
        date=date,
        archive_bytes=payload,
        checksum_bytes=checksum,
    )

    assert entry["schema_valid"] is False
