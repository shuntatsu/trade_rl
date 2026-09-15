"""Strict structural validator for the Issue 602 USD-M 1h target source."""

from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

TARGET_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
)
TARGET_MONTHS = tuple(
    f"{year}-{month:02d}" for year in (2021, 2022) for month in range(1, 13)
)

_INTERVAL_MS = 60 * 60 * 1000
_REPORT_SCHEMA = "issue602_usdm_1h_target_source_report_v1"
_EXPECTED_HEADER = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)

_ISSUE600_PROTOCOL_HEAD = "da97eb59b08ef042a96ebea4aaa645af02b450e9"
_ISSUE600_PROTOCOL_DIGEST = (
    "c78d556f94e247d5bdc1c0e9a891d160e4c48c4f0b62cd5be7600874b3017d22"
)
_ISSUE600_FULL_VERIFY_RUN_ID = 34958973042
_ISSUE600_DIGEST_VERIFY_RUN_ID = 34959535015
_ISSUE600_SEAL_RUN_ID = 34959723495
_ISSUE600_SEAL_ARTIFACT_ID = 10392597937
_ISSUE600_SEAL_ARTIFACT_API_DIGEST = (
    "c485b90f4be41556c85e50e146c663dca8bcaa84c4d5dae3c2d26ce179fd60f3"
)
_ISSUE600_FRESH_ARTIFACT_ID = 10392916717
_ISSUE600_FRESH_ARTIFACT_API_DIGEST = (
    "3cf2580fc29240583025da1f7bb2b7b5a4d8bf15f24e49c525c3ad58757470ed"
)
_ISSUE600_AUDIT_RECOVERY_RUN_ID = 34959887850
_ISSUE600_AUDIT_ARTIFACT_ID = 10392966885
_ISSUE600_AUDIT_ARTIFACT_API_DIGEST = (
    "79a363b394529fbd1364cb08d0792592ed69e3876fc337ee36c76f877fa2e454"
)
_SUPERSEDED_RAW_ENDPOINT_HEAD = "9e0345a9f51f56cd84efcb856be810f97a4b8ca1"

_ENTRY_FIELDS = {
    "symbol",
    "month",
    "url",
    "checksum_url",
    "archive_available",
    "raw_zip_size_bytes",
    "raw_zip_sha256",
    "checksum_available",
    "checksum_text",
    "checksum_digest",
    "checksum_verified",
    "member_name",
    "header_present",
    "normalized_schema",
    "row_count",
    "expected_row_count",
    "missing_grid_rows",
    "missing_grid_open_times_sha256",
    "first_open_time",
    "last_open_time",
    "first_close_time",
    "last_close_time",
    "open_times_strictly_increasing_unique",
    "native_grid_valid",
    "close_time_valid",
    "rows_inside_requested_utc_month",
    "dataset_timestamp_semantics",
    "dataset_timestamp_offset_ms",
    "schema_valid",
}


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


def _strict_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    text = str(value)
    if not text or text.strip() != text or text.startswith("+"):
        return None
    try:
        parsed = int(text)
    except ValueError:
        return None
    if str(parsed) != text and not (text == "-0" and parsed == 0):
        return None
    return parsed


def _finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    text = str(value)
    if not text or text.strip() != text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_month(month: str) -> tuple[int, int]:
    if not isinstance(month, str):
        raise ValueError("month must use YYYY-MM")
    try:
        value = datetime.strptime(month + "-01", "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("month must use YYYY-MM") from error
    if value.strftime("%Y-%m") != month:
        raise ValueError("month must use YYYY-MM")
    return value.year, value.month


def expected_month_row_count(month: str) -> int:
    """Return the exact number of native UTC 1h rows in a calendar month."""

    year, month_number = _parse_month(month)
    return calendar.monthrange(year, month_number)[1] * 24


def _month_bounds_ms(month: str) -> tuple[int, int]:
    year, month_number = _parse_month(month)
    start = datetime(year, month_number, 1, tzinfo=UTC)
    if month_number == 12:
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(year, month_number + 1, 1, tzinfo=UTC)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _url(symbol: str, month: str) -> str:
    return (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{symbol}/1h/{symbol}-1h-{month}.zip"
    )


def _blank_entry(symbol: str, month: str) -> dict[str, object]:
    url = _url(symbol, month)
    return {
        "symbol": symbol,
        "month": month,
        "url": url,
        "checksum_url": url + ".CHECKSUM",
        "archive_available": False,
        "raw_zip_size_bytes": None,
        "raw_zip_sha256": None,
        "checksum_available": False,
        "checksum_text": None,
        "checksum_digest": None,
        "checksum_verified": False,
        "member_name": None,
        "header_present": None,
        "normalized_schema": list(_EXPECTED_HEADER),
        "row_count": None,
        "expected_row_count": expected_month_row_count(month),
        "missing_grid_rows": None,
        "missing_grid_open_times_sha256": None,
        "first_open_time": None,
        "last_open_time": None,
        "first_close_time": None,
        "last_close_time": None,
        "open_times_strictly_increasing_unique": None,
        "native_grid_valid": None,
        "close_time_valid": None,
        "rows_inside_requested_utc_month": None,
        "dataset_timestamp_semantics": "completed_bar_close_boundary",
        "dataset_timestamp_offset_ms": _INTERVAL_MS,
        "schema_valid": None,
    }


def _parse_checksum(
    payload: bytes,
    *,
    expected_name: str,
    actual_digest: str | None,
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
        _hex(digest, length=64, field="checksum_digest")
    except ValueError:
        return text, None, False
    verified = (
        actual_digest is not None and name == expected_name and digest == actual_digest
    )
    return text, digest, verified


def validate_target_archive_bytes(
    *,
    symbol: str,
    month: str,
    archive_bytes: bytes | None,
    checksum_bytes: bytes | None,
) -> dict[str, object]:
    """Validate one frozen monthly archive without computing economic summaries."""

    if symbol not in TARGET_SYMBOLS or month not in TARGET_MONTHS:
        raise ValueError("symbol/month is outside the frozen Issue 602 roster")
    entry = _blank_entry(symbol, month)
    if set(entry) != _ENTRY_FIELDS:
        raise AssertionError("target archive entry schema differs from frozen contract")

    expected_archive_name = f"{symbol}-1h-{month}.zip"
    expected_member = f"{symbol}-1h-{month}.csv"

    if archive_bytes is None:
        if checksum_bytes is not None:
            if not isinstance(checksum_bytes, bytes):
                raise TypeError("checksum_bytes must be bytes or None")
            entry["checksum_available"] = True
            text, digest, _ = _parse_checksum(
                checksum_bytes,
                expected_name=expected_archive_name,
                actual_digest=None,
            )
            entry["checksum_text"] = text
            entry["checksum_digest"] = digest
        return entry
    if not isinstance(archive_bytes, bytes):
        raise TypeError("archive_bytes must be bytes or None")

    entry["archive_available"] = True
    entry["raw_zip_size_bytes"] = len(archive_bytes)
    archive_digest = hashlib.sha256(archive_bytes).hexdigest()
    entry["raw_zip_sha256"] = archive_digest

    if checksum_bytes is not None:
        if not isinstance(checksum_bytes, bytes):
            raise TypeError("checksum_bytes must be bytes or None")
        entry["checksum_available"] = True
        text, digest, verified = _parse_checksum(
            checksum_bytes,
            expected_name=expected_archive_name,
            actual_digest=archive_digest,
        )
        entry["checksum_text"] = text
        entry["checksum_digest"] = digest
        entry["checksum_verified"] = verified

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = tuple(item for item in archive.infolist() if not item.is_dir())
            if len(members) != 1:
                entry["schema_valid"] = False
                return entry
            member = members[0]
            entry["member_name"] = member.filename
            if member.filename != expected_member:
                entry["schema_valid"] = False
                return entry
            raw_text = archive.read(member).decode("utf-8-sig")
    except (UnicodeDecodeError, zipfile.BadZipFile, KeyError):
        entry["schema_valid"] = False
        return entry

    rows = list(csv.reader(io.StringIO(raw_text)))
    header_present = bool(rows and tuple(rows[0]) == _EXPECTED_HEADER)
    entry["header_present"] = header_present
    if header_present:
        rows = rows[1:]
    entry["row_count"] = len(rows)

    start_ms, end_ms = _month_bounds_ms(month)
    expected_rows = expected_month_row_count(month)
    expected_grid = tuple(
        start_ms + index * _INTERVAL_MS for index in range(expected_rows)
    )
    malformed = False
    grid_valid = True
    close_valid = True
    inside_month = True
    open_times: list[int] = []
    close_times: list[int] = []

    for row in rows:
        if len(row) != len(_EXPECTED_HEADER):
            malformed = True
            continue
        open_time = _strict_int(row[0])
        close_time = _strict_int(row[6])
        ohlc = tuple(_finite(row[index]) for index in (1, 2, 3, 4))
        nonnegative = tuple(_finite(row[index]) for index in (5, 7, 9, 10))
        count = _strict_int(row[8])
        ignore = _finite(row[11])
        row_numeric_valid = (
            open_time is not None
            and close_time is not None
            and all(value is not None and value > 0.0 for value in ohlc)
            and all(value is not None and value >= 0.0 for value in nonnegative)
            and count is not None
            and count >= 0
            and ignore is not None
        )
        if not row_numeric_valid:
            malformed = True
            continue
        assert open_time is not None and close_time is not None
        if not (start_ms <= open_time < end_ms and start_ms <= close_time < end_ms):
            inside_month = False
        if (open_time - start_ms) % _INTERVAL_MS != 0:
            grid_valid = False
        if close_time != open_time + _INTERVAL_MS - 1:
            close_valid = False
        open_times.append(open_time)
        close_times.append(close_time)

    increasing_unique = bool(open_times) and all(
        current > previous for previous, current in zip(open_times, open_times[1:])
    )
    observed = set(open_times)
    missing_grid = [
        timestamp for timestamp in expected_grid if timestamp not in observed
    ]
    missing_encoded = ",".join(str(timestamp) for timestamp in missing_grid).encode(
        "ascii"
    )

    entry["missing_grid_rows"] = len(missing_grid)
    entry["missing_grid_open_times_sha256"] = hashlib.sha256(
        missing_encoded
    ).hexdigest()
    entry["first_open_time"] = open_times[0] if open_times else None
    entry["last_open_time"] = open_times[-1] if open_times else None
    entry["first_close_time"] = close_times[0] if close_times else None
    entry["last_close_time"] = close_times[-1] if close_times else None
    entry["open_times_strictly_increasing_unique"] = increasing_unique
    entry["native_grid_valid"] = grid_valid
    entry["close_time_valid"] = close_valid
    entry["rows_inside_requested_utc_month"] = inside_month
    entry["schema_valid"] = bool(
        rows
        and not malformed
        and increasing_unique
        and grid_valid
        and close_valid
        and inside_month
        and len(open_times) <= expected_rows
        and len(observed) == len(open_times)
        and observed.issubset(set(expected_grid))
    )
    return entry


def _entry_semantics_valid(report: Mapping[str, object]) -> bool:
    if set(report) != _ENTRY_FIELDS:
        return False
    symbol = report.get("symbol")
    month = report.get("month")
    if not isinstance(symbol, str) or not isinstance(month, str):
        return False
    if symbol not in TARGET_SYMBOLS or month not in TARGET_MONTHS:
        return False
    expected_url = _url(symbol, month)
    if (
        report.get("url") != expected_url
        or report.get("checksum_url") != expected_url + ".CHECKSUM"
    ):
        return False
    if report.get("normalized_schema") != list(_EXPECTED_HEADER):
        return False
    expected_rows = expected_month_row_count(month)
    if report.get("expected_row_count") != expected_rows or isinstance(
        report.get("expected_row_count"), bool
    ):
        return False
    if report.get("dataset_timestamp_semantics") != "completed_bar_close_boundary":
        return False
    if report.get("dataset_timestamp_offset_ms") != _INTERVAL_MS or isinstance(
        report.get("dataset_timestamp_offset_ms"), bool
    ):
        return False

    for field in ("archive_available", "checksum_available", "checksum_verified"):
        if type(report.get(field)) is not bool:
            return False
    for field in (
        "header_present",
        "open_times_strictly_increasing_unique",
        "native_grid_valid",
        "close_time_valid",
        "rows_inside_requested_utc_month",
        "schema_valid",
    ):
        value = report.get(field)
        if value is not None and type(value) is not bool:
            return False

    for field in ("raw_zip_size_bytes", "row_count", "missing_grid_rows"):
        value = report.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            return False
    for field in (
        "first_open_time",
        "last_open_time",
        "first_close_time",
        "last_close_time",
    ):
        value = report.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            return False

    for field in (
        "raw_zip_sha256",
        "checksum_digest",
        "missing_grid_open_times_sha256",
    ):
        value = report.get(field)
        if value is not None:
            try:
                _hex(value, length=64, field=field)
            except ValueError:
                return False
    for field in ("checksum_text", "member_name"):
        value = report.get(field)
        if value is not None and not isinstance(value, str):
            return False

    archive_available = report["archive_available"]
    checksum_available = report["checksum_available"]
    checksum_verified = report["checksum_verified"]
    schema_valid = report["schema_valid"]

    if archive_available is False:
        for field in (
            "raw_zip_size_bytes",
            "raw_zip_sha256",
            "member_name",
            "header_present",
            "row_count",
            "missing_grid_rows",
            "missing_grid_open_times_sha256",
            "first_open_time",
            "last_open_time",
            "first_close_time",
            "last_close_time",
            "open_times_strictly_increasing_unique",
            "native_grid_valid",
            "close_time_valid",
            "rows_inside_requested_utc_month",
            "schema_valid",
        ):
            if report.get(field) is not None:
                return False
    else:
        size = report.get("raw_zip_size_bytes")
        raw_digest = report.get("raw_zip_sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            return False
        if not isinstance(raw_digest, str):
            return False

    if checksum_available is False:
        if (
            report.get("checksum_text") is not None
            or report.get("checksum_digest") is not None
        ):
            return False
        if checksum_verified is not False:
            return False
    if checksum_verified is True:
        if archive_available is not True or checksum_available is not True:
            return False
        if report.get("checksum_digest") != report.get("raw_zip_sha256"):
            return False
        if not isinstance(report.get("checksum_text"), str):
            return False

    if schema_valid is True:
        if archive_available is not True:
            return False
        if type(report.get("header_present")) is not bool:
            return False
        if report.get("member_name") != f"{symbol}-1h-{month}.csv":
            return False
        row_count = report.get("row_count")
        missing_rows = report.get("missing_grid_rows")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or isinstance(missing_rows, bool)
            or not isinstance(missing_rows, int)
            or not (0 < row_count <= expected_rows)
            or not (0 <= missing_rows <= expected_rows)
            or row_count + missing_rows != expected_rows
        ):
            return False
        missing_hash = report.get("missing_grid_open_times_sha256")
        if not isinstance(missing_hash, str):
            return False
        if missing_rows == 0 and missing_hash != hashlib.sha256(b"").hexdigest():
            return False
        for field in (
            "open_times_strictly_increasing_unique",
            "native_grid_valid",
            "close_time_valid",
            "rows_inside_requested_utc_month",
        ):
            if report.get(field) is not True:
                return False
        first_open = report.get("first_open_time")
        last_open = report.get("last_open_time")
        first_close = report.get("first_close_time")
        last_close = report.get("last_close_time")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (first_open, last_open, first_close, last_close)
        ):
            return False
        assert isinstance(first_open, int)
        assert isinstance(last_open, int)
        assert isinstance(first_close, int)
        assert isinstance(last_close, int)
        start_ms, end_ms = _month_bounds_ms(month)
        if not (
            start_ms <= first_open <= last_open < end_ms
            and start_ms <= first_close <= last_close < end_ms
            and (first_open - start_ms) % _INTERVAL_MS == 0
            and (last_open - start_ms) % _INTERVAL_MS == 0
            and first_close == first_open + _INTERVAL_MS - 1
            and last_close == last_open + _INTERVAL_MS - 1
        ):
            return False
        if row_count == expected_rows and (
            first_open != start_ms
            or last_open != end_ms - _INTERVAL_MS
            or first_close != start_ms + _INTERVAL_MS - 1
            or last_close != end_ms - 1
        ):
            return False
    return True


def _ordered_entries(
    reports: Sequence[Mapping[str, object]],
) -> list[dict[str, object]] | None:
    expected_count = len(TARGET_SYMBOLS) * len(TARGET_MONTHS)
    if len(reports) != expected_count:
        return None
    by_pair: dict[tuple[object, object], dict[str, object]] = {}
    for report in reports:
        if not _entry_semantics_valid(report):
            return None
        pair = (report.get("symbol"), report.get("month"))
        if pair in by_pair:
            return None
        by_pair[pair] = dict(report)
    expected_pairs = [
        (symbol, month) for symbol in TARGET_SYMBOLS for month in TARGET_MONTHS
    ]
    if set(by_pair) != set(expected_pairs):
        return None
    return [by_pair[pair] for pair in expected_pairs]


def decide_target_source_status(reports: Sequence[Mapping[str, object]]) -> str:
    """Apply the frozen PASS/PARTIAL/INCOMPATIBLE Issue 602 gate."""

    entries = _ordered_entries(reports)
    if entries is None:
        return "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"
    partial = False
    for entry in entries:
        archive_available = entry["archive_available"]
        checksum_available = entry["checksum_available"]
        checksum_verified = entry["checksum_verified"]
        schema_valid = entry["schema_valid"]
        if archive_available is False or checksum_available is False:
            partial = True
            if archive_available is True and schema_valid is not True:
                return "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"
            continue
        if checksum_verified is not True or schema_valid is not True:
            return "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"
    return "PARTIAL_USDM_1H_TARGET_SOURCE" if partial else "PASS_USDM_1H_TARGET_SOURCE"


def build_target_source_report(
    reports: Sequence[Mapping[str, object]],
    *,
    validator_head: str,
    validator_verification_run_id: int,
) -> dict[str, object]:
    """Build the canonical result-blind structural target-source report."""

    entries = _ordered_entries(reports)
    if entries is None:
        raise ValueError(
            "target-source report roster/semantics differ from frozen Issue 602"
        )
    validator = _hex(validator_head, length=40, field="validator_head")
    validator_run = _positive_int(
        validator_verification_run_id, field="validator_verification_run_id"
    )
    status = decide_target_source_status(entries)
    payload: dict[str, object] = {
        "schema_version": _REPORT_SCHEMA,
        "issue_number": 602,
        "premium_prereg_issue": 600,
        "issue600_protocol_head": _ISSUE600_PROTOCOL_HEAD,
        "issue600_protocol_digest": _ISSUE600_PROTOCOL_DIGEST,
        "issue600_full_verify_run_id": _ISSUE600_FULL_VERIFY_RUN_ID,
        "issue600_digest_verify_run_id": _ISSUE600_DIGEST_VERIFY_RUN_ID,
        "issue600_seal_run_id": _ISSUE600_SEAL_RUN_ID,
        "issue600_seal_artifact_id": _ISSUE600_SEAL_ARTIFACT_ID,
        "issue600_seal_artifact_api_digest": _ISSUE600_SEAL_ARTIFACT_API_DIGEST,
        "issue600_fresh_artifact_id": _ISSUE600_FRESH_ARTIFACT_ID,
        "issue600_fresh_artifact_api_digest": _ISSUE600_FRESH_ARTIFACT_API_DIGEST,
        "issue600_audit_recovery_run_id": _ISSUE600_AUDIT_RECOVERY_RUN_ID,
        "issue600_audit_artifact_id": _ISSUE600_AUDIT_ARTIFACT_ID,
        "issue600_audit_artifact_api_digest": _ISSUE600_AUDIT_ARTIFACT_API_DIGEST,
        "superseded_raw_endpoint_head": _SUPERSEDED_RAW_ENDPOINT_HEAD,
        "superseded_raw_endpoint_authority_used": False,
        "target_source_family": "binance_vision_contract_klines",
        "target_transport_mode": "VISION",
        "timeframe": "1h",
        "dataset_timestamp_semantics": "completed_bar_close_boundary",
        "dataset_timestamp_offset_ms": _INTERVAL_MS,
        "validator_head": validator,
        "validator_verification_run_id": validator_run,
        "planned_archive_count": len(TARGET_SYMBOLS) * len(TARGET_MONTHS),
        "available_archive_count": sum(
            entry["archive_available"] is True for entry in entries
        ),
        "available_checksum_count": sum(
            entry["checksum_available"] is True for entry in entries
        ),
        "checksum_verified_count": sum(
            entry["checksum_verified"] is True for entry in entries
        ),
        "structurally_valid_archive_count": sum(
            entry["schema_valid"] is True for entry in entries
        ),
        "complete_archive_count": sum(
            entry["schema_valid"] is True
            and entry["row_count"] == entry["expected_row_count"]
            and entry["missing_grid_rows"] == 0
            for entry in entries
        ),
        "total_missing_grid_rows": sum(
            int(entry["missing_grid_rows"])
            for entry in entries
            if isinstance(entry["missing_grid_rows"], int)
            and not isinstance(entry["missing_grid_rows"], bool)
        ),
        "status": status,
        "entries": entries,
        "premium_economic_values_inspected": False,
        "target_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    payload["content_digest"] = content_digest(payload)
    return payload


def _report_fields() -> set[str]:
    return {
        "schema_version",
        "issue_number",
        "premium_prereg_issue",
        "issue600_protocol_head",
        "issue600_protocol_digest",
        "issue600_full_verify_run_id",
        "issue600_digest_verify_run_id",
        "issue600_seal_run_id",
        "issue600_seal_artifact_id",
        "issue600_seal_artifact_api_digest",
        "issue600_fresh_artifact_id",
        "issue600_fresh_artifact_api_digest",
        "issue600_audit_recovery_run_id",
        "issue600_audit_artifact_id",
        "issue600_audit_artifact_api_digest",
        "superseded_raw_endpoint_head",
        "superseded_raw_endpoint_authority_used",
        "target_source_family",
        "target_transport_mode",
        "timeframe",
        "dataset_timestamp_semantics",
        "dataset_timestamp_offset_ms",
        "validator_head",
        "validator_verification_run_id",
        "planned_archive_count",
        "available_archive_count",
        "available_checksum_count",
        "checksum_verified_count",
        "structurally_valid_archive_count",
        "complete_archive_count",
        "total_missing_grid_rows",
        "status",
        "entries",
        "premium_economic_values_inspected",
        "target_relation_computed",
        "evaluation_pnl_inspected",
        "production_eligible",
        "live_trading_authorized",
        "content_digest",
    }


def canonical_target_source_report_bytes(report: Mapping[str, object]) -> bytes:
    """Validate report semantic closure and return canonical JSON bytes."""

    if set(report) != _report_fields():
        raise ValueError("target-source report contains missing or forbidden fields")
    entries = report.get("entries")
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    validator_run = report.get("validator_verification_run_id")
    if isinstance(validator_run, bool) or not isinstance(validator_run, int):
        raise ValueError("validator_verification_run_id must be an integer")
    validator_head = report.get("validator_head")
    if not isinstance(validator_head, str):
        raise ValueError("validator_head must be a string")
    rebuilt = build_target_source_report(
        entries,
        validator_head=validator_head,
        validator_verification_run_id=validator_run,
    )
    if dict(report) != rebuilt:
        raise ValueError("target-source report semantic closure differs")
    return canonical_json_bytes(rebuilt)


def load_target_source_report_bytes(payload: bytes) -> dict[str, object]:
    """Strictly load a canonical Issue 602 structural report."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    try:
        raw: Any = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("target-source report is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("target-source report must be a JSON object")
    if canonical_json_bytes(raw) != payload:
        raise ValueError("target-source report bytes are not canonical JSON")
    if canonical_target_source_report_bytes(raw) != payload:
        raise ValueError("target-source report semantic closure differs")
    return dict(raw)


__all__ = [
    "TARGET_MONTHS",
    "TARGET_SYMBOLS",
    "build_target_source_report",
    "canonical_target_source_report_bytes",
    "decide_target_source_status",
    "expected_month_row_count",
    "load_target_source_report_bytes",
    "validate_target_archive_bytes",
]
