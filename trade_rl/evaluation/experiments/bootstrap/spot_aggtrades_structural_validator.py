"""Strict structural-only validator for the sealed Issue 578 Spot aggTrades probe."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.spot_aggtrades_source_prereg import (
    SpotAggTradesSourceProtocol,
    canonical_spot_aggtrades_source_protocol,
)

_REPORT_SCHEMA = "issue578_spot_aggtrades_structural_report_v1"


def _canonical_protocol(protocol: SpotAggTradesSourceProtocol) -> None:
    canonical = canonical_spot_aggtrades_source_protocol()
    if protocol != canonical or protocol.digest != canonical.digest:
        raise ValueError("Spot aggTrades protocol is not the sealed canonical authority")


def _hex(value: object, *, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be {length} lowercase hexadecimal characters")
    if value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be {length} lowercase hexadecimal characters")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _parse_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        text = str(value)
        if not text or text.strip() != text:
            return None
        if text.startswith("+"):
            return None
        parsed = int(text)
    except (TypeError, ValueError):
        return None
    return parsed if str(parsed) == text or (text == "-0" and parsed == 0) else None


def _parse_finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _day_bounds_ms(date: str) -> tuple[int, int]:
    try:
        start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("date must use YYYY-MM-DD") from error
    if start.strftime("%Y-%m-%d") != date:
        raise ValueError("date must use YYYY-MM-DD")
    return int(start.timestamp() * 1_000), int((start + timedelta(days=1)).timestamp() * 1_000)


def _blank_archive_report(
    protocol: SpotAggTradesSourceProtocol,
    *,
    symbol: str,
    date: str,
) -> dict[str, object]:
    url = protocol.url_template.format(symbol=symbol, date=date)
    return {
        "symbol": symbol,
        "date": date,
        "url": url,
        "checksum_url": url + protocol.checksum_suffix,
        "archive_available": False,
        "raw_zip_size_bytes": None,
        "raw_zip_sha256": None,
        "checksum_available": False,
        "checksum_text": None,
        "checksum_digest": None,
        "checksum_verified": False,
        "member_name": None,
        "header_present": None,
        "normalized_schema": list(protocol.expected_header),
        "total_row_count": None,
        "usable_row_count": None,
        "provider_invalid_sentinel_count": None,
        "malformed_row_count": None,
        "first_usable_aggregate_id": None,
        "last_usable_aggregate_id": None,
        "first_usable_event_timestamp_ms": None,
        "last_usable_event_timestamp_ms": None,
        "usable_ids_strictly_increasing_unique": None,
        "usable_timestamps_nondecreasing": None,
        "timestamps_inside_requested_utc_date": None,
        "schema_valid": None,
    }


def _parse_checksum(
    payload: bytes,
    *,
    expected_name: str,
    actual_archive_digest: str | None,
) -> tuple[str | None, str | None, bool]:
    try:
        text = payload.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None, None, False
    parts = text.split()
    if len(parts) != 2:
        return text, None, False
    digest, raw_name = parts
    name = raw_name.lstrip("*")
    try:
        _hex(digest, length=64, field="checksum digest")
    except ValueError:
        return text, None, False
    verified = (
        name == expected_name
        and actual_archive_digest is not None
        and digest == actual_archive_digest
    )
    return text, digest, verified


def validate_archive_bytes(
    protocol: SpotAggTradesSourceProtocol,
    *,
    symbol: str,
    date: str,
    archive_bytes: bytes | None,
    checksum_bytes: bytes | None,
) -> dict[str, object]:
    """Validate one exact frozen archive using structural information only."""

    _canonical_protocol(protocol)
    if symbol not in protocol.symbols or date not in protocol.dates:
        raise ValueError("symbol/date is outside the frozen Spot aggTrades roster")

    report = _blank_archive_report(protocol, symbol=symbol, date=date)
    expected_fields = set(protocol.allowed_archive_report_fields)
    if set(report) != expected_fields:
        raise AssertionError("validator report schema differs from sealed allowlist")

    if archive_bytes is None:
        if checksum_bytes is not None:
            report["checksum_available"] = True
            text, digest, _ = _parse_checksum(
                checksum_bytes,
                expected_name=report["url"].rsplit("/", 1)[-1],
                actual_archive_digest=None,
            )
            report["checksum_text"] = text
            report["checksum_digest"] = digest
        return report

    if not isinstance(archive_bytes, bytes):
        raise TypeError("archive_bytes must be bytes or None")
    report["archive_available"] = True
    report["raw_zip_size_bytes"] = len(archive_bytes)
    archive_digest = hashlib.sha256(archive_bytes).hexdigest()
    report["raw_zip_sha256"] = archive_digest

    archive_name = str(report["url"]).rsplit("/", 1)[-1]
    expected_member = archive_name.removesuffix(".zip") + ".csv"
    integrity_ok = True
    if checksum_bytes is None:
        report["checksum_available"] = False
    else:
        if not isinstance(checksum_bytes, bytes):
            raise TypeError("checksum_bytes must be bytes or None")
        report["checksum_available"] = True
        text, digest, verified = _parse_checksum(
            checksum_bytes,
            expected_name=archive_name,
            actual_archive_digest=archive_digest,
        )
        report["checksum_text"] = text
        report["checksum_digest"] = digest
        report["checksum_verified"] = verified
        integrity_ok = verified

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = tuple(item for item in archive.infolist() if not item.is_dir())
            if len(members) != 1:
                report["schema_valid"] = False
                return report
            member = members[0]
            report["member_name"] = member.filename
            if member.filename != expected_member:
                report["schema_valid"] = False
                return report
            raw_text = archive.read(member).decode("utf-8-sig")
    except (UnicodeDecodeError, zipfile.BadZipFile, KeyError):
        report["schema_valid"] = False
        return report

    rows = list(csv.reader(io.StringIO(raw_text)))
    header_present = bool(rows and tuple(rows[0]) == protocol.expected_header)
    report["header_present"] = header_present
    if header_present:
        rows = rows[1:]
    report["total_row_count"] = len(rows)

    malformed = 0
    sentinel_count = 0
    usable_ids: list[int] = []
    usable_timestamps: list[int] = []
    timestamps_inside = True
    day_start_ms, day_end_ms = _day_bounds_ms(date)
    allowed_booleans = set(protocol.strict_boolean_tokens)

    for row in rows:
        if len(row) != protocol.field_count:
            malformed += 1
            timestamps_inside = False
            continue

        aggregate_raw, price_raw, quantity_raw, first_raw, last_raw, timestamp_raw, buyer_raw, best_raw = row
        timestamp = _parse_integer(timestamp_raw)
        common_valid = (
            timestamp is not None
            and timestamp >= 0
            and day_start_ms <= timestamp < day_end_ms
            and buyer_raw in allowed_booleans
            and best_raw in allowed_booleans
        )
        if timestamp is None or not (day_start_ms <= timestamp < day_end_ms):
            timestamps_inside = False

        sentinel = (
            price_raw == protocol.provider_invalid_sentinel["price"]
            and quantity_raw == protocol.provider_invalid_sentinel["quantity"]
            and first_raw == protocol.provider_invalid_sentinel["first_trade_id"]
            and last_raw == protocol.provider_invalid_sentinel["last_trade_id"]
        )
        if sentinel:
            sentinel_count += 1
            if not common_valid:
                malformed += 1
            continue

        aggregate_id = _parse_integer(aggregate_raw)
        price = _parse_finite(price_raw)
        quantity = _parse_finite(quantity_raw)
        first_trade_id = _parse_integer(first_raw)
        last_trade_id = _parse_integer(last_raw)
        row_valid = (
            common_valid
            and aggregate_id is not None
            and aggregate_id >= 0
            and price is not None
            and price > 0.0
            and quantity is not None
            and quantity > 0.0
            and first_trade_id is not None
            and first_trade_id >= 0
            and last_trade_id is not None
            and last_trade_id >= 0
            and first_trade_id <= last_trade_id
        )
        if not row_valid:
            malformed += 1
            continue
        assert aggregate_id is not None and timestamp is not None
        usable_ids.append(aggregate_id)
        usable_timestamps.append(timestamp)

    ids_increasing = all(
        current > previous for previous, current in zip(usable_ids, usable_ids[1:])
    )
    timestamps_nondecreasing = all(
        current >= previous
        for previous, current in zip(usable_timestamps, usable_timestamps[1:])
    )

    report["usable_row_count"] = len(usable_ids)
    report["provider_invalid_sentinel_count"] = sentinel_count
    report["malformed_row_count"] = malformed
    report["first_usable_aggregate_id"] = usable_ids[0] if usable_ids else None
    report["last_usable_aggregate_id"] = usable_ids[-1] if usable_ids else None
    report["first_usable_event_timestamp_ms"] = (
        usable_timestamps[0] if usable_timestamps else None
    )
    report["last_usable_event_timestamp_ms"] = (
        usable_timestamps[-1] if usable_timestamps else None
    )
    report["usable_ids_strictly_increasing_unique"] = ids_increasing
    report["usable_timestamps_nondecreasing"] = timestamps_nondecreasing
    report["timestamps_inside_requested_utc_date"] = timestamps_inside
    report["schema_valid"] = bool(
        integrity_ok
        and rows
        and malformed == 0
        and ids_increasing
        and timestamps_nondecreasing
        and timestamps_inside
    )
    return report


def _require_archive_report_fields(
    protocol: SpotAggTradesSourceProtocol,
    report: Mapping[str, object],
) -> None:
    if set(report) != set(protocol.allowed_archive_report_fields):
        raise ValueError("archive report contains missing or forbidden structural fields")


def decide_source_status(
    protocol: SpotAggTradesSourceProtocol,
    reports: Sequence[Mapping[str, object]],
) -> str:
    """Apply only the frozen PASS/PARTIAL/INCOMPATIBLE structural gate."""

    _canonical_protocol(protocol)
    expected_pairs = [
        (symbol, date) for symbol in protocol.symbols for date in protocol.dates
    ]
    if len(reports) != protocol.planned_archive_count:
        raise ValueError("structural gate requires exactly 20 frozen archive reports")

    missing = False
    for expected, report in zip(expected_pairs, reports):
        _require_archive_report_fields(protocol, report)
        pair = (report.get("symbol"), report.get("date"))
        if pair != expected:
            raise ValueError("archive reports do not match the frozen ordered roster")
        archive_available = report.get("archive_available")
        checksum_available = report.get("checksum_available")
        checksum_verified = report.get("checksum_verified")
        schema_valid = report.get("schema_valid")
        if type(archive_available) is not bool or type(checksum_available) is not bool:
            raise ValueError("archive availability fields must be strict booleans")
        if type(checksum_verified) is not bool:
            raise ValueError("checksum_verified must be a strict boolean")
        if not archive_available or not checksum_available:
            missing = True
            if archive_available and schema_valid is not True:
                return protocol.incompatible_status
            continue
        if checksum_verified is not True or schema_valid is not True:
            return protocol.incompatible_status

    return protocol.partial_status if missing else protocol.pass_status


def build_structural_report(
    protocol: SpotAggTradesSourceProtocol,
    reports: Sequence[Mapping[str, object]],
    *,
    protocol_head: str,
    protocol_seal_run_id: int,
    protocol_seal_artifact_id: int,
    protocol_seal_artifact_api_digest: str,
    protocol_fresh_artifact_id: int,
    protocol_fresh_artifact_api_digest: str,
) -> dict[str, object]:
    """Build a content-addressed structural-only result with sealed authority binding."""

    _canonical_protocol(protocol)
    head = _hex(protocol_head, length=40, field="protocol_head")
    seal_digest = _hex(
        protocol_seal_artifact_api_digest,
        length=64,
        field="protocol_seal_artifact_api_digest",
    )
    fresh_digest = _hex(
        protocol_fresh_artifact_api_digest,
        length=64,
        field="protocol_fresh_artifact_api_digest",
    )
    archive_reports = [dict(report) for report in reports]
    status = decide_source_status(protocol, archive_reports)
    payload: dict[str, object] = {
        "schema_version": _REPORT_SCHEMA,
        "issue_number": 578,
        "protocol_digest": protocol.digest,
        "protocol_head": head,
        "protocol_seal_run_id": _positive_int(
            protocol_seal_run_id, field="protocol_seal_run_id"
        ),
        "protocol_seal_artifact_id": _positive_int(
            protocol_seal_artifact_id, field="protocol_seal_artifact_id"
        ),
        "protocol_seal_artifact_api_digest": seal_digest,
        "protocol_fresh_artifact_id": _positive_int(
            protocol_fresh_artifact_id, field="protocol_fresh_artifact_id"
        ),
        "protocol_fresh_artifact_api_digest": fresh_digest,
        "planned_archive_count": protocol.planned_archive_count,
        "available_archive_count": sum(
            report["archive_available"] is True for report in archive_reports
        ),
        "available_checksum_count": sum(
            report["checksum_available"] is True for report in archive_reports
        ),
        "checksum_verified_count": sum(
            report["checksum_verified"] is True for report in archive_reports
        ),
        "structurally_valid_available_archive_count": sum(
            report["archive_available"] is True and report["schema_valid"] is True
            for report in archive_reports
        ),
        "status": status,
        "archive_reports": archive_reports,
        "economic_values_inspected": False,
        "target_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "feature_hypothesis_selected": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    payload["content_digest"] = content_digest(payload)
    return payload


def canonical_structural_report_bytes(report: Mapping[str, Any]) -> bytes:
    """Validate report closure and return its exact canonical JSON bytes."""

    expected_fields = {
        "schema_version",
        "issue_number",
        "protocol_digest",
        "protocol_head",
        "protocol_seal_run_id",
        "protocol_seal_artifact_id",
        "protocol_seal_artifact_api_digest",
        "protocol_fresh_artifact_id",
        "protocol_fresh_artifact_api_digest",
        "planned_archive_count",
        "available_archive_count",
        "available_checksum_count",
        "checksum_verified_count",
        "structurally_valid_available_archive_count",
        "status",
        "archive_reports",
        "economic_values_inspected",
        "target_relation_computed",
        "evaluation_pnl_inspected",
        "feature_hypothesis_selected",
        "production_eligible",
        "live_trading_authorized",
        "content_digest",
    }
    if set(report) != expected_fields:
        raise ValueError("structural report fields are not canonical")
    if report.get("schema_version") != _REPORT_SCHEMA or report.get("issue_number") != 578:
        raise ValueError("structural report schema authority differs")
    for field in (
        "economic_values_inspected",
        "target_relation_computed",
        "evaluation_pnl_inspected",
        "feature_hypothesis_selected",
        "production_eligible",
        "live_trading_authorized",
    ):
        if report.get(field) is not False:
            raise ValueError(f"forbidden structural-report flag differs: {field}")
    payload = dict(report)
    observed_digest = payload.pop("content_digest")
    if observed_digest != content_digest(payload):
        raise ValueError("structural report content_digest differs")
    return canonical_json_bytes(dict(report))
