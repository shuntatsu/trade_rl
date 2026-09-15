from __future__ import annotations

from pathlib import Path

MODULE = Path(
    "trade_rl/evaluation/experiments/bootstrap/issue586_target_source_validator.py"
)
BASE_TEST = Path(
    "tests/evaluation/experiments/bootstrap/test_issue586_target_source_validator.py"
)


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{label}: expected exactly one replacement target")
    return text.replace(old, new, 1)


def patch_module() -> None:
    text = MODULE.read_text()

    text = _replace_once(
        text,
        '    integrity_ok = False\n    if checksum_bytes is not None:\n',
        '    if checksum_bytes is not None:\n',
        label="remove integrity_ok initializer",
    )
    text = _replace_once(
        text,
        '        entry["checksum_verified"] = verified\n        integrity_ok = verified\n',
        '        entry["checksum_verified"] = verified\n',
        label="remove integrity_ok assignment",
    )
    text = _replace_once(
        text,
        '''    entry["schema_valid"] = bool(\n        integrity_ok\n        and rows\n        and not malformed\n''',
        '''    entry["schema_valid"] = bool(\n        rows\n        and not malformed\n''',
        label="separate structural schema from checksum integrity",
    )

    marker = "\n\ndef _ordered_entries(\n"
    if marker not in text:
        raise SystemExit("ordered_entries marker missing")
    semantic_validator = r'''


def _entry_semantics_valid(report: Mapping[str, object]) -> bool:
    symbol = report.get("symbol")
    date = report.get("date")
    if not isinstance(symbol, str) or not isinstance(date, str):
        return False
    if symbol not in TARGET_SYMBOLS or date not in TARGET_DATES:
        return False

    expected_url = _url(symbol, date)
    if report.get("url") != expected_url:
        return False
    if report.get("checksum_url") != expected_url + ".CHECKSUM":
        return False
    if report.get("normalized_schema") != list(_EXPECTED_HEADER):
        return False
    expected_rows = report.get("expected_row_count")
    if isinstance(expected_rows, bool) or expected_rows != _EXPECTED_ROWS:
        return False

    for field in ("archive_available", "checksum_available", "checksum_verified"):
        if type(report.get(field)) is not bool:
            return False
    for field in (
        "header_present",
        "open_times_strictly_increasing_unique",
        "native_grid_valid",
        "close_time_valid",
        "rows_inside_requested_utc_date",
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
            "rows_inside_requested_utc_date",
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
        if report.get("checksum_text") is not None:
            return False
        if report.get("checksum_digest") is not None:
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
            expected_name=f"{symbol}-15m-{date}.zip",
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
        if report.get("member_name") != f"{symbol}-15m-{date}.csv":
            return False

        row_count = report.get("row_count")
        missing_rows = report.get("missing_grid_rows")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or isinstance(missing_rows, bool)
            or not isinstance(missing_rows, int)
            or not (0 < row_count <= _EXPECTED_ROWS)
            or not (0 <= missing_rows <= _EXPECTED_ROWS)
            or row_count + missing_rows != _EXPECTED_ROWS
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
            "rows_inside_requested_utc_date",
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

        start_ms, end_ms = _day_bounds_ms(date)
        if not (
            start_ms <= first_open <= last_open < end_ms
            and start_ms <= first_close <= last_close < end_ms
            and (first_open - start_ms) % _INTERVAL_MS == 0
            and (last_open - start_ms) % _INTERVAL_MS == 0
            and first_close == first_open + _INTERVAL_MS - 1
            and last_close == last_open + _INTERVAL_MS - 1
        ):
            return False
        if row_count == _EXPECTED_ROWS and (
            first_open != start_ms
            or last_open != end_ms - _INTERVAL_MS
            or first_close != start_ms + _INTERVAL_MS - 1
            or last_close != end_ms - 1
        ):
            return False
    return True
'''
    text = text.replace(marker, semantic_validator + marker, 1)

    text = _replace_once(
        text,
        '''        if set(report) != _ENTRY_FIELDS:\n            return None\n        pair = (report.get("symbol"), report.get("date"))\n''',
        '''        if set(report) != _ENTRY_FIELDS:\n            return None\n        if not _entry_semantics_valid(report):\n            return None\n        pair = (report.get("symbol"), report.get("date"))\n''',
        label="enforce nested entry semantic closure",
    )
    MODULE.write_text(text)


def patch_base_test() -> None:
    text = BASE_TEST.read_text()
    old = '''    assert report["checksum_verified"] is False\n    assert report["schema_valid"] is False\n\n    wrong_name = f"{hashlib.sha256(payload).hexdigest()}  wrong.zip\\n".encode()\n'''
    new = '''    assert report["checksum_verified"] is False\n    assert report["schema_valid"] is True\n\n    wrong_name = f"{hashlib.sha256(payload).hexdigest()}  wrong.zip\\n".encode()\n'''
    text = _replace_once(text, old, new, label="first checksum schema assertion")
    old = '''    assert report["checksum_verified"] is False\n    assert report["schema_valid"] is False\n\n\ndef test_timestamp_order_grid_date_and_close_time_fail_closed() -> None:\n'''
    new = '''    assert report["checksum_verified"] is False\n    assert report["schema_valid"] is True\n\n\ndef test_timestamp_order_grid_date_and_close_time_fail_closed() -> None:\n'''
    text = _replace_once(text, old, new, label="second checksum schema assertion")
    BASE_TEST.write_text(text)


def main() -> None:
    patch_module()
    patch_base_test()


if __name__ == "__main__":
    main()
