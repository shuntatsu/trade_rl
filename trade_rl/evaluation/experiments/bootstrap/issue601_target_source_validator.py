"""Structural-only validator for the frozen Issue 601 USD-M 1h target roster."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    f"{year:04d}-{month:02d}" for year in (2021, 2022) for month in range(1, 13)
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
_HOUR_MS = 60 * 60 * 1000
_URL_TEMPLATE = (
    "https://data.binance.vision/data/futures/um/monthly/klines/"
    "{symbol}/1h/{symbol}-1h-{month}.zip"
)
_REPORT_SCHEMA = "issue601_usdm_1h_target_source_report_v1"
_MIN_TARGET_WINDOWS = 16_621
_NOMINAL_DECISIONS = 17_495
_FIT_START_MS = int(datetime(2021, 1, 1, 1, tzinfo=UTC).timestamp() * 1000)
_LAST_DECISION_MS = int(datetime(2022, 12, 30, 23, tzinfo=UTC).timestamp() * 1000)

_ISSUE600_PROTOCOL_HEAD = "6ffc414baa10e91f34df258fe5cfabacce65fd77"
_ISSUE600_PROTOCOL_DIGEST = (
    "18bf9625502eccbe475d157df90635f1c4645d3c4cced48ec0fc565723206731"
)
_ISSUE600_PREREG_SEAL_RUN_ID = 34959849347
_ISSUE600_PREREG_SEAL_ARTIFACT_ID = 10392936042
_ISSUE600_PREREG_SEAL_ARTIFACT_API_DIGEST = (
    "4b611462f8d4f9bb4d23ab15bb4d8a20525d4093e590c1afb62d1b74ca875e15"
)
_ISSUE600_PREREG_FRESH_ARTIFACT_ID = 10391714897
_ISSUE600_PREREG_FRESH_ARTIFACT_API_DIGEST = (
    "858e930a03e442f047556801c3d9a536fbc89435b5f261cba621988dd8062acd"
)

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
    "schema_valid",
}
_WINDOW_FIELDS = {
    "nominal_decisions",
    "structurally_present_windows",
    "structurally_missing_windows",
    "missing_decision_timestamps_sha256",
}


@dataclass(frozen=True, slots=True)
class TargetArchiveValidation:
    """One archive's public structural report plus private timestamp-only state."""

    report: dict[str, object]
    open_times: tuple[int, ...]


def _month_bounds_ms(month: str) -> tuple[int, int]:
    if month not in TARGET_MONTHS:
        raise ValueError("month is outside the frozen Issue 601 target roster")
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def expected_rows_in_month(month: str) -> int:
    start_ms, end_ms = _month_bounds_ms(month)
    return (end_ms - start_ms) // _HOUR_MS


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


def _url(symbol: str, month: str) -> str:
    return _URL_TEMPLATE.format(symbol=symbol, month=month)


def _missing_digest(values: Sequence[int]) -> str:
    encoded = ",".join(str(value) for value in values).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


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
        "expected_row_count": expected_rows_in_month(month),
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
    return (
        text,
        digest,
        bool(
            name == expected_name
            and actual_digest is not None
            and digest == actual_digest
        ),
    )


def validate_target_archive_bytes(
    *,
    symbol: str,
    month: str,
    archive_bytes: bytes | None,
    checksum_bytes: bytes | None,
) -> TargetArchiveValidation:
    """Validate one exact monthly archive without reporting economic values."""

    if symbol not in TARGET_SYMBOLS or month not in TARGET_MONTHS:
        raise ValueError("symbol/month is outside the frozen Issue 601 target roster")
    entry = _blank_entry(symbol, month)
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
        return TargetArchiveValidation(report=entry, open_times=())

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
                return TargetArchiveValidation(report=entry, open_times=())
            member = members[0]
            entry["member_name"] = member.filename
            if member.filename != expected_member:
                entry["schema_valid"] = False
                return TargetArchiveValidation(report=entry, open_times=())
            text = archive.read(member).decode("utf-8-sig")
    except (UnicodeDecodeError, zipfile.BadZipFile, KeyError):
        entry["schema_valid"] = False
        return TargetArchiveValidation(report=entry, open_times=())

    rows = list(csv.reader(io.StringIO(text)))
    header_present = bool(rows and tuple(rows[0]) == _EXPECTED_HEADER)
    entry["header_present"] = header_present
    if header_present:
        rows = rows[1:]
    entry["row_count"] = len(rows)

    start_ms, end_ms = _month_bounds_ms(month)
    expected_count = expected_rows_in_month(month)
    expected_grid = tuple(
        start_ms + index * _HOUR_MS for index in range(expected_count)
    )
    malformed = False
    open_times: list[int] = []
    close_times: list[int] = []
    grid_valid = True
    close_valid = True
    inside_month = True

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

        if not (start_ms <= open_time < end_ms and start_ms <= close_time < end_ms):
            inside_month = False
        if (open_time - start_ms) % _HOUR_MS != 0:
            grid_valid = False
        if close_time != open_time + _HOUR_MS - 1:
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

    entry["missing_grid_rows"] = len(missing_grid)
    entry["missing_grid_open_times_sha256"] = _missing_digest(missing_grid)
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
        and len(observed) == len(open_times)
        and observed.issubset(set(expected_grid))
    )
    return TargetArchiveValidation(report=entry, open_times=tuple(open_times))


def _decision_times() -> tuple[int, ...]:
    values: list[int] = []
    current = _FIT_START_MS
    while current <= _LAST_DECISION_MS:
        values.append(current)
        current += _HOUR_MS
    if len(values) != _NOMINAL_DECISIONS:
        raise RuntimeError("frozen Issue 600 decision count is inconsistent")
    return tuple(values)


def count_structural_target_windows(open_times: Sequence[int]) -> dict[str, object]:
    """Count only timestamp-present t..t+24h target windows."""

    observed: set[int] = set()
    for value in open_times:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("target open timestamps must be integers")
        observed.add(value)
    missing_decisions: list[int] = []
    for decision in _decision_times():
        if not all(decision + offset * _HOUR_MS in observed for offset in range(25)):
            missing_decisions.append(decision)
    missing_count = len(missing_decisions)
    return {
        "nominal_decisions": _NOMINAL_DECISIONS,
        "structurally_present_windows": _NOMINAL_DECISIONS - missing_count,
        "structurally_missing_windows": missing_count,
        "missing_decision_timestamps_sha256": _missing_digest(missing_decisions),
    }


def _entry_semantics_valid(report: Mapping[str, object]) -> bool:
    if set(report) != _ENTRY_FIELDS:
        return False
    symbol = report.get("symbol")
    month = report.get("month")
    if not isinstance(symbol, str) or symbol not in TARGET_SYMBOLS:
        return False
    if not isinstance(month, str) or month not in TARGET_MONTHS:
        return False
    expected_url = _url(symbol, month)
    if (
        report.get("url") != expected_url
        or report.get("checksum_url") != expected_url + ".CHECKSUM"
    ):
        return False
    if report.get("normalized_schema") != list(_EXPECTED_HEADER):
        return False
    expected_rows = report.get("expected_row_count")
    if isinstance(expected_rows, bool) or expected_rows != expected_rows_in_month(
        month
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
        try:
            _hex(raw_digest, length=64, field="raw_zip_sha256")
        except ValueError:
            return False

    if checksum_available is False:
        if (
            report.get("checksum_text") is not None
            or report.get("checksum_digest") is not None
        ):
            return False
        if checksum_verified is not False:
            return False
    else:
        checksum_text = report.get("checksum_text")
        if not isinstance(checksum_text, str):
            return False
        raw_digest = report.get("raw_zip_sha256")
        parsed_text, parsed_digest, parsed_verified = _parse_checksum(
            checksum_text.encode("utf-8"),
            expected_name=f"{symbol}-1h-{month}.zip",
            actual_digest=raw_digest if isinstance(raw_digest, str) else None,
        )
        if parsed_text != checksum_text:
            return False
        if parsed_digest != report.get("checksum_digest"):
            return False
        if parsed_verified is not checksum_verified:
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
        expected_count = expected_rows_in_month(month)
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or isinstance(missing_rows, bool)
            or not isinstance(missing_rows, int)
            or not (0 < row_count <= expected_count)
            or not (0 <= missing_rows <= expected_count)
            or row_count + missing_rows != expected_count
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
            and (first_open - start_ms) % _HOUR_MS == 0
            and (last_open - start_ms) % _HOUR_MS == 0
            and first_close == first_open + _HOUR_MS - 1
            and last_close == last_open + _HOUR_MS - 1
        ):
            return False
        if row_count == expected_count and (
            first_open != start_ms
            or last_open != end_ms - _HOUR_MS
            or first_close != start_ms + _HOUR_MS - 1
            or last_close != end_ms - 1
        ):
            return False
    return True


def _ordered_entries(
    reports: Sequence[Mapping[str, object]],
) -> list[dict[str, object]] | None:
    if len(reports) != len(TARGET_SYMBOLS) * len(TARGET_MONTHS):
        return None
    by_pair: dict[tuple[str, str], dict[str, object]] = {}
    for report in reports:
        if not _entry_semantics_valid(report):
            return None
        symbol = report.get("symbol")
        month = report.get("month")
        assert isinstance(symbol, str)
        assert isinstance(month, str)
        pair = (symbol, month)
        if pair in by_pair:
            return None
        by_pair[pair] = dict(report)
    expected_pairs = [
        (symbol, month) for symbol in TARGET_SYMBOLS for month in TARGET_MONTHS
    ]
    if set(by_pair) != set(expected_pairs):
        return None
    return [by_pair[pair] for pair in expected_pairs]


def _window_summary_valid(value: Mapping[str, object]) -> bool:
    if set(value) != _WINDOW_FIELDS:
        return False
    nominal = value.get("nominal_decisions")
    present = value.get("structurally_present_windows")
    missing = value.get("structurally_missing_windows")
    digest = value.get("missing_decision_timestamps_sha256")
    if any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in (nominal, present, missing)
    ):
        return False
    assert isinstance(nominal, int)
    assert isinstance(present, int)
    assert isinstance(missing, int)
    if (
        nominal != _NOMINAL_DECISIONS
        or present < 0
        or missing < 0
        or present + missing != nominal
    ):
        return False
    try:
        _hex(digest, length=64, field="missing_decision_timestamps_sha256")
    except ValueError:
        return False
    return True


def _ordered_window_summaries(
    summaries: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]] | None:
    if set(summaries) != set(TARGET_SYMBOLS):
        return None
    result: dict[str, dict[str, object]] = {}
    for symbol in TARGET_SYMBOLS:
        value = summaries[symbol]
        if not _window_summary_valid(value):
            return None
        result[symbol] = dict(value)
    return result


def decide_target_source_status(
    reports: Sequence[Mapping[str, object]],
    target_windows_by_symbol: Mapping[str, Mapping[str, object]],
) -> str:
    entries = _ordered_entries(reports)
    summaries = _ordered_window_summaries(target_windows_by_symbol)
    if entries is None or summaries is None:
        return "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"

    partial = False
    for entry in entries:
        archive_available = entry["archive_available"]
        checksum_available = entry["checksum_available"]
        checksum_verified = entry["checksum_verified"]
        schema_valid = entry["schema_valid"]
        if archive_available is True and schema_valid is not True:
            return "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"
        if checksum_available is True and entry.get("checksum_digest") is None:
            return "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"
        if (
            archive_available is True
            and checksum_available is True
            and checksum_verified is not True
        ):
            return "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"
        if archive_available is not True or checksum_available is not True:
            partial = True

    if any(
        summary["structurally_present_windows"] < _MIN_TARGET_WINDOWS
        for summary in summaries.values()
    ):
        partial = True
    return "PARTIAL_USDM_1H_TARGET_SOURCE" if partial else "PASS_USDM_1H_TARGET_SOURCE"


def build_target_source_report(
    validations: Sequence[TargetArchiveValidation],
    *,
    validator_head: str,
    validator_verification_run_id: int,
) -> dict[str, object]:
    """Build canonical result-blind source report from structural validations."""

    _hex(validator_head, length=40, field="validator_head")
    _positive_int(validator_verification_run_id, field="validator_verification_run_id")
    reports = [validation.report for validation in validations]
    entries = _ordered_entries(reports)
    if entries is None:
        raise ValueError("target archive roster/report semantics are not canonical")

    open_times_by_symbol: dict[str, list[int]] = {
        symbol: [] for symbol in TARGET_SYMBOLS
    }
    seen: set[tuple[str, str]] = set()
    for validation in validations:
        symbol = validation.report.get("symbol")
        month = validation.report.get("month")
        if not isinstance(symbol, str) or not isinstance(month, str):
            raise ValueError("validation symbol/month is malformed")
        pair = (symbol, month)
        if pair in seen:
            raise ValueError("duplicate target archive validation")
        seen.add(pair)
        if validation.report.get("schema_valid") is True:
            open_times_by_symbol[symbol].extend(validation.open_times)

    windows = {
        symbol: count_structural_target_windows(open_times_by_symbol[symbol])
        for symbol in TARGET_SYMBOLS
    }
    status = decide_target_source_status(entries, windows)
    unsigned: dict[str, object] = {
        "schema_version": _REPORT_SCHEMA,
        "issue_number": 601,
        "upstream_issue_number": 600,
        "issue600_protocol_head": _ISSUE600_PROTOCOL_HEAD,
        "issue600_protocol_digest": _ISSUE600_PROTOCOL_DIGEST,
        "issue600_prereg_seal_run_id": _ISSUE600_PREREG_SEAL_RUN_ID,
        "issue600_prereg_seal_artifact_id": _ISSUE600_PREREG_SEAL_ARTIFACT_ID,
        "issue600_prereg_seal_artifact_api_digest": (
            _ISSUE600_PREREG_SEAL_ARTIFACT_API_DIGEST
        ),
        "issue600_prereg_fresh_artifact_id": _ISSUE600_PREREG_FRESH_ARTIFACT_ID,
        "issue600_prereg_fresh_artifact_api_digest": (
            _ISSUE600_PREREG_FRESH_ARTIFACT_API_DIGEST
        ),
        "validator_head": validator_head,
        "validator_verification_run_id": validator_verification_run_id,
        "target_source_market": "USD_M",
        "target_source_family": "klines",
        "target_interval": "1h",
        "planned_archive_count": len(TARGET_SYMBOLS) * len(TARGET_MONTHS),
        "minimum_structural_target_windows_per_symbol": _MIN_TARGET_WINDOWS,
        "entries": entries,
        "target_windows_by_symbol": windows,
        "status": status,
        "fallback_used": False,
        "replacement_source_used": False,
        "economic_values_inspected": False,
        "target_price_values_inspected": False,
        "target_return_computed": False,
        "premium_values_inspected": False,
        "training_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "final_test_authorized": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    return {**unsigned, "content_digest": content_digest(unsigned)}


_REPORT_FIELDS = {
    "schema_version",
    "issue_number",
    "upstream_issue_number",
    "issue600_protocol_head",
    "issue600_protocol_digest",
    "issue600_prereg_seal_run_id",
    "issue600_prereg_seal_artifact_id",
    "issue600_prereg_seal_artifact_api_digest",
    "issue600_prereg_fresh_artifact_id",
    "issue600_prereg_fresh_artifact_api_digest",
    "validator_head",
    "validator_verification_run_id",
    "target_source_market",
    "target_source_family",
    "target_interval",
    "planned_archive_count",
    "minimum_structural_target_windows_per_symbol",
    "entries",
    "target_windows_by_symbol",
    "status",
    "fallback_used",
    "replacement_source_used",
    "economic_values_inspected",
    "target_price_values_inspected",
    "target_return_computed",
    "premium_values_inspected",
    "training_relation_computed",
    "evaluation_pnl_inspected",
    "final_test_authorized",
    "production_eligible",
    "live_trading_authorized",
    "content_digest",
}


def _validate_report_mapping(report: Mapping[str, object]) -> dict[str, object]:
    if set(report) != _REPORT_FIELDS:
        raise ValueError("target-source report fields are not canonical")
    fixed = {
        "schema_version": _REPORT_SCHEMA,
        "issue_number": 601,
        "upstream_issue_number": 600,
        "issue600_protocol_head": _ISSUE600_PROTOCOL_HEAD,
        "issue600_protocol_digest": _ISSUE600_PROTOCOL_DIGEST,
        "issue600_prereg_seal_run_id": _ISSUE600_PREREG_SEAL_RUN_ID,
        "issue600_prereg_seal_artifact_id": _ISSUE600_PREREG_SEAL_ARTIFACT_ID,
        "issue600_prereg_seal_artifact_api_digest": _ISSUE600_PREREG_SEAL_ARTIFACT_API_DIGEST,
        "issue600_prereg_fresh_artifact_id": _ISSUE600_PREREG_FRESH_ARTIFACT_ID,
        "issue600_prereg_fresh_artifact_api_digest": _ISSUE600_PREREG_FRESH_ARTIFACT_API_DIGEST,
        "target_source_market": "USD_M",
        "target_source_family": "klines",
        "target_interval": "1h",
        "planned_archive_count": 120,
        "minimum_structural_target_windows_per_symbol": _MIN_TARGET_WINDOWS,
        "fallback_used": False,
        "replacement_source_used": False,
        "economic_values_inspected": False,
        "target_price_values_inspected": False,
        "target_return_computed": False,
        "premium_values_inspected": False,
        "training_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "final_test_authorized": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    for field, expected in fixed.items():
        value = report.get(field)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"{field} is not canonical")
    _hex(report.get("validator_head"), length=40, field="validator_head")
    _positive_int(
        report.get("validator_verification_run_id"),
        field="validator_verification_run_id",
    )
    digest = _hex(report.get("content_digest"), length=64, field="content_digest")

    raw_entries = report.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("entries must be a JSON array")
    entry_mappings: list[Mapping[str, object]] = []
    for value in raw_entries:
        if not isinstance(value, dict):
            raise ValueError("entry must be a JSON object")
        entry_mappings.append(value)
    ordered_entries = _ordered_entries(entry_mappings)
    if ordered_entries is None or ordered_entries != raw_entries:
        raise ValueError("entries are not the exact canonical frozen roster")

    raw_windows = report.get("target_windows_by_symbol")
    if not isinstance(raw_windows, dict):
        raise ValueError("target_windows_by_symbol must be an object")
    window_mappings: dict[str, Mapping[str, object]] = {}
    for key, value in raw_windows.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError("target window summary is malformed")
        window_mappings[key] = value
    ordered_windows = _ordered_window_summaries(window_mappings)
    if ordered_windows is None or ordered_windows != raw_windows:
        raise ValueError("target window summaries are not canonical")

    expected_status = decide_target_source_status(ordered_entries, ordered_windows)
    if report.get("status") != expected_status:
        raise ValueError("target-source status is not implied by structural evidence")
    unsigned = dict(report)
    unsigned.pop("content_digest")
    if content_digest(unsigned) != digest:
        raise ValueError("target-source report content digest mismatch")
    return dict(report)


def canonical_target_source_report_bytes(report: Mapping[str, object]) -> bytes:
    validated = _validate_report_mapping(report)
    return canonical_json_bytes(validated)


def load_target_source_report_bytes(payload: bytes) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    try:
        raw: Any = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("target-source report is not valid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("target-source report must be a JSON object")
    validated = _validate_report_mapping(raw)
    if payload != canonical_json_bytes(validated):
        raise ValueError("target-source report bytes are not canonical")
    return validated
