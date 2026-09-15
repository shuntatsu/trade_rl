from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, datetime

import pytest

import tools.tmp_issue578_spot_aggtrades_source_probe as probe
from trade_rl.evaluation.experiments.bootstrap.spot_aggtrades_source_prereg import (
    canonical_spot_aggtrades_source_protocol,
)

SYMBOL = "BTCUSDT"
DATE = "2021-01-15"
HEADER = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
)


def _timestamp(offset_ms: int = 0, *, date: str = DATE) -> int:
    parsed = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000) + offset_ms


def _zip_bytes(
    rows: list[list[object]], *, header: tuple[str, ...] | None = HEADER
) -> bytes:
    member = f"{SYMBOL}-aggTrades-{DATE}.csv"
    lines: list[str] = []
    if header is not None:
        lines.append(",".join(header))
    lines.extend(",".join(str(value) for value in row) for row in rows)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member, "\n".join(lines) + "\n")
    return stream.getvalue()


def _checksum(payload: bytes, *, name: str | None = None) -> bytes:
    filename = name or f"{SYMBOL}-aggTrades-{DATE}.zip"
    return f"{hashlib.sha256(payload).hexdigest()}  {filename}\n".encode()


def _parse(
    rows: list[list[object]], *, header: tuple[str, ...] | None = HEADER
) -> dict[str, object]:
    payload = _zip_bytes(rows, header=header)
    return probe.parse_archive_bytes(
        symbol=SYMBOL,
        date=DATE,
        payload=payload,
        checksum_payload=_checksum(payload),
    )


def test_exact_provider_sentinel_is_structural_but_never_usable() -> None:
    entry = _parse(
        [
            [1, "100.0", "2.0", 10, 11, _timestamp(), "False", "True"],
            [2, "0", "0", -1, -1, _timestamp(1_000), "True", "False"],
            [3, "101.0", "1.5", 12, 12, _timestamp(2_000), "True", "True"],
        ]
    )

    protocol = canonical_spot_aggtrades_source_protocol()
    assert set(entry) == set(protocol.allowed_archive_report_fields)
    assert entry["schema_valid"] is True
    assert entry["total_row_count"] == 3
    assert entry["usable_row_count"] == 2
    assert entry["provider_invalid_sentinel_count"] == 1
    assert entry["malformed_row_count"] == 0
    assert entry["first_usable_aggregate_id"] == 1
    assert entry["last_usable_aggregate_id"] == 3
    assert entry["usable_ids_strictly_increasing_unique"] is True
    assert entry["usable_timestamps_nondecreasing"] is True
    assert entry["timestamps_inside_requested_utc_date"] is True


def test_headerless_is_accepted_but_noncanonical_header_is_rejected() -> None:
    rows = [[1, "100", "2", 10, 10, _timestamp(), "False", "True"]]
    assert _parse(rows, header=None)["schema_valid"] is True

    wrong = HEADER[:5] + ("timestamp",) + HEADER[6:]
    invalid = _parse(rows, header=wrong)
    assert invalid["schema_valid"] is False


@pytest.mark.parametrize(
    "row",
    [
        [1, "0", "1", -1, -1, _timestamp(), "False", "True"],
        [1, "100", "2", -1, 10, _timestamp(), "False", "True"],
        [1, "nan", "2", 10, 10, _timestamp(), "False", "True"],
        [1, "100", "0", 10, 10, _timestamp(), "False", "True"],
        [1, "100", "2", 11, 10, _timestamp(), "False", "True"],
        [1, "100", "2", 10, 10, _timestamp(), "false", "True"],
        [-1, "100", "2", 10, 10, _timestamp(), "False", "True"],
        [1, "100", "2", 10, 10, _timestamp(-1), "False", "True"],
    ],
)
def test_malformed_or_near_sentinel_rows_fail_closed(row: list[object]) -> None:
    entry = _parse([row])
    assert entry["schema_valid"] is False
    assert entry["malformed_row_count"] == 1


def test_usable_ordering_is_strict_for_ids_and_nondecreasing_for_time() -> None:
    duplicate_id = _parse(
        [
            [2, "100", "2", 10, 10, _timestamp(), "False", "True"],
            [2, "101", "2", 11, 11, _timestamp(1), "True", "True"],
        ]
    )
    assert duplicate_id["schema_valid"] is False
    assert duplicate_id["usable_ids_strictly_increasing_unique"] is False

    decreasing_time = _parse(
        [
            [1, "100", "2", 10, 10, _timestamp(2), "False", "True"],
            [2, "101", "2", 11, 11, _timestamp(1), "True", "True"],
        ]
    )
    assert decreasing_time["schema_valid"] is False
    assert decreasing_time["usable_timestamps_nondecreasing"] is False


def test_checksum_and_member_identity_fail_closed() -> None:
    rows = [[1, "100", "2", 10, 10, _timestamp(), "False", "True"]]
    payload = _zip_bytes(rows)
    wrong_checksum_name = probe.parse_archive_bytes(
        symbol=SYMBOL,
        date=DATE,
        payload=payload,
        checksum_payload=_checksum(payload, name="other.zip"),
    )
    assert wrong_checksum_name["checksum_verified"] is False
    assert wrong_checksum_name["schema_valid"] is False

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("wrong.csv", "1,100,2,10,10,1600000000000,False,True\n")
    wrong_member_payload = stream.getvalue()
    wrong_member = probe.parse_archive_bytes(
        symbol=SYMBOL,
        date=DATE,
        payload=wrong_member_payload,
        checksum_payload=_checksum(wrong_member_payload),
    )
    assert wrong_member["schema_valid"] is False


def _status_entry(
    *,
    available: bool = True,
    checksum_available: bool = True,
    checksum_verified: bool = True,
    schema_valid: bool = True,
) -> dict[str, object]:
    return {
        "archive_available": available,
        "checksum_available": checksum_available,
        "checksum_verified": checksum_verified,
        "schema_valid": schema_valid,
        "malformed_row_count": 0,
        "usable_ids_strictly_increasing_unique": schema_valid,
        "usable_timestamps_nondecreasing": schema_valid,
        "timestamps_inside_requested_utc_date": schema_valid,
    }


def test_frozen_status_gate_distinguishes_missing_from_incompatible() -> None:
    assert (
        probe.classify_status([_status_entry() for _ in range(20)])
        == "PASS_SPOT_AGGTRADES_SOURCE"
    )

    partial = [_status_entry() for _ in range(20)]
    partial[-1] = _status_entry(
        available=False,
        checksum_available=False,
        checksum_verified=False,
        schema_valid=False,
    )
    assert probe.classify_status(partial) == "PARTIAL_SPOT_AGGTRADES_SOURCE"

    missing_checksum = [_status_entry() for _ in range(20)]
    missing_checksum[-1] = _status_entry(
        checksum_available=False,
        checksum_verified=False,
        schema_valid=True,
    )
    assert probe.classify_status(missing_checksum) == "PARTIAL_SPOT_AGGTRADES_SOURCE"

    incompatible = [_status_entry() for _ in range(20)]
    incompatible[-1] = _status_entry(checksum_verified=False)
    assert probe.classify_status(incompatible) == "INCOMPATIBLE_SPOT_AGGTRADES_SOURCE"


def _valid_structural_entry(symbol: str, date: str) -> dict[str, object]:
    protocol = canonical_spot_aggtrades_source_protocol()
    url = protocol.url_template.format(symbol=symbol, date=date)
    filename = url.rsplit("/", 1)[-1]
    digest = "a" * 64
    timestamp = _timestamp(date=date)
    return {
        "symbol": symbol,
        "date": date,
        "url": url,
        "checksum_url": url + protocol.checksum_suffix,
        "archive_available": True,
        "raw_zip_size_bytes": 1,
        "raw_zip_sha256": digest,
        "checksum_available": True,
        "checksum_text": f"{digest}  {filename}",
        "checksum_digest": digest,
        "checksum_verified": True,
        "member_name": filename.removesuffix(".zip") + ".csv",
        "header_present": True,
        "normalized_schema": list(HEADER),
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


def test_report_contains_only_frozen_structural_evidence() -> None:
    protocol = canonical_spot_aggtrades_source_protocol()
    entries = [
        _valid_structural_entry(symbol, date)
        for symbol in protocol.symbols
        for date in protocol.dates
    ]
    report = probe.build_report_from_entries(entries, probe_commit="a" * 40)

    assert report["status"] == "PASS_SPOT_AGGTRADES_SOURCE"
    assert report["planned_archives"] == 20
    assert report["economic_values_inspected"] is False
    assert report["target_relation_computed"] is False
    assert report["evaluation_pnl_inspected"] is False
    assert report["feature_hypothesis_selected"] is False
    assert report["production_eligible"] is False
    assert report["live_trading_authorized"] is False
    assert report["content_digest"] == probe.report_content_digest(report)

    forbidden_keys = {
        "price_min",
        "price_max",
        "price_mean",
        "quantity_min",
        "quantity_max",
        "quantity_mean",
        "notional",
        "imbalance",
        "return",
        "correlation",
        "beta",
        "ic",
        "pnl",
    }
    observed_keys = {str(key).lower() for key in report}
    for entry in report["entries"]:
        observed_keys.update(str(key).lower() for key in entry)
    assert observed_keys.isdisjoint(forbidden_keys)
