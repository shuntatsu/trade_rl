"""Structural-only validator for the frozen Issue 586 USD-M 15m target roster."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

TARGET_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
)
TARGET_DATES = (
    "2021-01-15",
    "2021-01-16",
    "2021-07-15",
    "2021-07-16",
    "2022-01-15",
    "2022-01-16",
    "2022-07-15",
    "2022-07-16",
)

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
_INTERVAL_MS = 15 * 60 * 1000
_EXPECTED_ROWS = 96
_URL_TEMPLATE = (
    "https://data.binance.vision/data/futures/um/daily/klines/"
    "{symbol}/15m/{symbol}-15m-{date}.zip"
)
_REPORT_SCHEMA = "issue586_usdm_15m_target_source_report_v1"
_ISSUE584_PROTOCOL_HEAD = "89ec1092e438d69657840567e684739ca0c4e3d5"
_ISSUE584_PROTOCOL_DIGEST = (
    "bf0aa2db248e9dcab745e92e6bde54de6b48276a785eec74670e0dedba8cb5e3"
)
_ISSUE584_PREREG_SEAL_RUN_ID = 34932424904
_ISSUE584_PREREG_SEAL_ARTIFACT_ID = 10381763665
_ISSUE584_PREREG_SEAL_ARTIFACT_API_DIGEST = (
    "f0a7de4e5b477f86d773793f3d4a154ac307c68a88b4eee4cf10a48190fd0f02"
)
_ISSUE584_PREREG_FRESH_ARTIFACT_ID = 10381379452
_ISSUE584_PREREG_FRESH_ARTIFACT_API_DIGEST = (
    "012126c32a0d65c0164624dff9b9657a4a8ff208bf24397256048f65ea0f89b8"
)

_ENTRY_FIELDS = {
    "symbol",
    "date",
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
    "rows_inside_requested_utc_date",
    "schema_valid",
}


def _day_bounds_ms(date: str) -> tuple[int, int]:
    try:
        start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("date must use YYYY-MM-DD") from error
    if start.strftime("%Y-%m-%d") != date:
        raise ValueError("date must use YYYY-MM-DD")
    return int(start.timestamp() * 1000), int(
        (start + timedelta(days=1)).timestamp() * 1000
    )


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
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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


def _url(symbol: str, date: str) -> str:
    return _URL_TEMPLATE.format(symbol=symbol, date=date)


def _blank_entry(symbol: str, date: str) -> dict[str, object]:
    url = _url(symbol, date)
    return {
        "symbol": symbol,
        "date": date,
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
        "expected_row_count": _EXPECTED_ROWS,
        "missing_grid_rows": None,
        "missing_grid_open_times_sha256": None,
        "first_open_time": None,
        "last_open_time": None,
        "first_close_time": None,
        "last_close_time": None,
        "open_times_strictly_increasing_unique": None,
        "native_grid_valid": None,
        "close_time_valid": None,
        "rows_inside_requested_utc_date": None,
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
        _hex(digest, length=64, field="checksum digest")
    except ValueError:
        return text, None, False
    return text, digest, bool(
        name == expected_name and actual_digest is not None and digest == actual_digest
    )


def validate_target_archive_bytes(
    *,
    symbol: str,
    date: str,
    archive_bytes: bytes | None,
    checksum_bytes: bytes | None,
) -> dict[str, object]:
    """Validate one exact frozen daily archive without reporting economic values."""

    if symbol not in TARGET_SYMBOLS or date not in TARGET_DATES:
        raise ValueError("symbol/date is outside the frozen Issue 586 target roster")
    entry = _blank_entry(symbol, date)
    expected_archive_name = f"{symbol}-15m-{date}.zip"
    expected_member = f"{symbol}-15m-{date}.csv"

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

    integrity_ok = False
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
        integrity_ok = verified

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
            text = archive.read(member).decode("utf-8-sig")
    except (UnicodeDecodeError, zipfile.BadZipFile, KeyError):
        entry["schema_valid"] = False
        return entry

    rows = list(csv.reader(io.StringIO(text)))
    header_present = bool(rows and tuple(rows[0]) == _EXPECTED_HEADER)
    entry["header_present"] = header_present
    if header_present:
        rows = rows[1:]
    entry["row_count"] = len(rows)

    start_ms, end_ms = _day_bounds_ms(date)
    expected_grid = tuple(start_ms + index * _INTERVAL_MS for index in range(96))
    malformed = False
    open_times: list[int] = []
    close_times: list[int] = []
    grid_valid = True
    close_valid = True
    inside_date = True

    for row in rows:
        if len(row) != len(_EXPECTED_HEADER):
            malformed = True
            continue
        open_time = _strict_int(row[0])
        close_time = _strict_int(row[6])
        finite_columns = (1, 2, 3, 4, 5, 7, 9, 10, 11)
        finite_ok = all(_finite(row[index]) is not None for index in finite_columns)
        count = _strict_int(row[8])
        count_ok = count is not None and count >= 0
        if open_time is None or close_time is None or not finite_ok or not count_ok:
            malformed = True
            continue

        row_inside = start_ms <= open_time < end_ms and start_ms <= close_time < end_ms
        if not row_inside:
            inside_date = False
        if (open_time - start_ms) % _INTERVAL_MS != 0:
            grid_valid = False
        if close_time != open_time + _INTERVAL_MS - 1:
            close_valid = False
        open_times.append(open_time)
        close_times.append(close_time)

    increasing_unique = bool(open_times) and all(
        current > previous for previous, current in zip(open_times, open_times[1:])
    )
    observed_grid = set(open_times)
    missing_grid = [timestamp for timestamp in expected_grid if timestamp not in observed_grid]
    missing_encoded = ",".join(str(timestamp) for timestamp in missing_grid).encode("ascii")

    entry["missing_grid_rows"] = len(missing_grid)
    entry["missing_grid_open_times_sha256"] = hashlib.sha256(missing_encoded).hexdigest()
    entry["first_open_time"] = open_times[0] if open_times else None
    entry["last_open_time"] = open_times[-1] if open_times else None
    entry["first_close_time"] = close_times[0] if close_times else None
    entry["last_close_time"] = close_times[-1] if close_times else None
    entry["open_times_strictly_increasing_unique"] = increasing_unique
    entry["native_grid_valid"] = grid_valid
    entry["close_time_valid"] = close_valid
    entry["rows_inside_requested_utc_date"] = inside_date
    entry["schema_valid"] = bool(
        integrity_ok
        and rows
        and not malformed
        and increasing_unique
        and grid_valid
        and close_valid
        and inside_date
    )
    return entry


def _ordered_entries(
    reports: Sequence[Mapping[str, object]],
) -> list[dict[str, object]] | None:
    if len(reports) != len(TARGET_SYMBOLS) * len(TARGET_DATES):
        return None
    by_pair: dict[tuple[object, object], dict[str, object]] = {}
    for report in reports:
        if set(report) != _ENTRY_FIELDS:
            return None
        pair = (report.get("symbol"), report.get("date"))
        if pair in by_pair:
            return None
        by_pair[pair] = dict(report)
    expected = [(symbol, date) for symbol in TARGET_SYMBOLS for date in TARGET_DATES]
    if set(by_pair) != set(expected):
        return None
    return [by_pair[pair] for pair in expected]


def decide_target_source_status(reports: Sequence[Mapping[str, object]]) -> str:
    """Apply the frozen PASS/PARTIAL/INCOMPATIBLE structural gate."""

    entries = _ordered_entries(reports)
    if entries is None:
        return "INCOMPATIBLE_USDM_15M_TARGET_SOURCE"

    partial = False
    for entry in entries:
        archive_available = entry["archive_available"]
        checksum_available = entry["checksum_available"]
        checksum_verified = entry["checksum_verified"]
        schema_valid = entry["schema_valid"]
        if any(
            type(value) is not bool
            for value in (archive_available, checksum_available, checksum_verified)
        ):
            return "INCOMPATIBLE_USDM_15M_TARGET_SOURCE"

        if archive_available is False or checksum_available is False:
            partial = True
            if archive_available is True and schema_valid is False:
                return "INCOMPATIBLE_USDM_15M_TARGET_SOURCE"
            continue
        if checksum_verified is not True or schema_valid is not True:
            return "INCOMPATIBLE_USDM_15M_TARGET_SOURCE"

        row_count = entry["row_count"]
        missing_rows = entry["missing_grid_rows"]
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or isinstance(missing_rows, bool)
            or not isinstance(missing_rows, int)
            or row_count < 0
            or missing_rows < 0
        ):
            return "INCOMPATIBLE_USDM_15M_TARGET_SOURCE"
        if row_count != _EXPECTED_ROWS or missing_rows != 0:
            partial = True

    return (
        "PARTIAL_USDM_15M_TARGET_SOURCE"
        if partial
        else "PASS_USDM_15M_TARGET_SOURCE"
    )


def build_target_source_report(
    reports: Sequence[Mapping[str, object]],
    *,
    validator_head: str,
    validator_verification_run_id: int,
) -> dict[str, object]:
    """Build the canonical result-blind structural target-source report."""

    entries = _ordered_entries(reports)
    if entries is None:
        raise ValueError("target-source report roster is not the frozen 40 archive roster")
    validator = _hex(validator_head, length=40, field="validator_head")
    validator_run = _positive_int(
        validator_verification_run_id,
        field="validator_verification_run_id",
    )
    status = decide_target_source_status(entries)
    payload: dict[str, object] = {
        "schema_version": _REPORT_SCHEMA,
        "issue_number": 586,
        "issue584_protocol_head": _ISSUE584_PROTOCOL_HEAD,
        "issue584_protocol_digest": _ISSUE584_PROTOCOL_DIGEST,
        "issue584_prereg_seal_run_id": _ISSUE584_PREREG_SEAL_RUN_ID,
        "issue584_prereg_seal_artifact_id": _ISSUE584_PREREG_SEAL_ARTIFACT_ID,
        "issue584_prereg_seal_artifact_api_digest": (
            _ISSUE584_PREREG_SEAL_ARTIFACT_API_DIGEST
        ),
        "issue584_prereg_fresh_artifact_id": _ISSUE584_PREREG_FRESH_ARTIFACT_ID,
        "issue584_prereg_fresh_artifact_api_digest": (
            _ISSUE584_PREREG_FRESH_ARTIFACT_API_DIGEST
        ),
        "target_source_family": "binance_vision_contract_klines",
        "target_transport_mode": "VISION",
        "timeframe": "15m",
        "validator_head": validator,
        "validator_verification_run_id": validator_run,
        "planned_archive_count": 40,
        "available_archive_count": sum(entry["archive_available"] is True for entry in entries),
        "available_checksum_count": sum(entry["checksum_available"] is True for entry in entries),
        "checksum_verified_count": sum(entry["checksum_verified"] is True for entry in entries),
        "structurally_valid_archive_count": sum(entry["schema_valid"] is True for entry in entries),
        "complete_96_row_archive_count": sum(
            entry["schema_valid"] is True
            and entry["row_count"] == _EXPECTED_ROWS
            and entry["missing_grid_rows"] == 0
            for entry in entries
        ),
        "status": status,
        "entries": entries,
        "economic_values_inspected": False,
        "target_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    payload["content_digest"] = content_digest(payload)
    return payload


def canonical_target_source_report_bytes(report: Mapping[str, object]) -> bytes:
    """Validate semantic closure and return canonical JSON bytes."""

    expected_fields = {
        "schema_version",
        "issue_number",
        "issue584_protocol_head",
        "issue584_protocol_digest",
        "issue584_prereg_seal_run_id",
        "issue584_prereg_seal_artifact_id",
        "issue584_prereg_seal_artifact_api_digest",
        "issue584_prereg_fresh_artifact_id",
        "issue584_prereg_fresh_artifact_api_digest",
        "target_source_family",
        "target_transport_mode",
        "timeframe",
        "validator_head",
        "validator_verification_run_id",
        "planned_archive_count",
        "available_archive_count",
        "available_checksum_count",
        "checksum_verified_count",
        "structurally_valid_archive_count",
        "complete_96_row_archive_count",
        "status",
        "entries",
        "economic_values_inspected",
        "target_relation_computed",
        "evaluation_pnl_inspected",
        "production_eligible",
        "live_trading_authorized",
        "content_digest",
    }
    if set(report) != expected_fields:
        raise ValueError("target-source report contains missing or forbidden fields")
    entries = report.get("entries")
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    rebuilt = build_target_source_report(
        entries,
        validator_head=str(report.get("validator_head")),
        validator_verification_run_id=report.get("validator_verification_run_id"),  # type: ignore[arg-type]
    )
    if dict(report) != rebuilt:
        raise ValueError("target-source report semantic closure differs")
    return canonical_json_bytes(rebuilt)
