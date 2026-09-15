from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime

from trade_rl.evaluation.experiments.bootstrap.issue586_target_source_validator import (
    TARGET_DATES,
    TARGET_SYMBOLS,
    build_target_source_report,
    canonical_target_source_report_bytes,
    decide_target_source_status,
    validate_target_archive_bytes,
)

_HEADER = (
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


def _day_ms(date: str) -> int:
    return int(
        datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000
    )


def _rows(date: str, *, count: int = 96) -> list[list[object]]:
    start = _day_ms(date)
    step = 15 * 60 * 1000
    return [
        [
            start + i * step,
            "1",
            "1",
            "1",
            "1",
            "0",
            start + (i + 1) * step - 1,
            "0",
            "0",
            "0",
            "0",
            "0",
        ]
        for i in range(count)
    ]


def _zip(
    symbol: str,
    date: str,
    rows: list[list[object]],
    *,
    member: str | None = None,
    extra: bool = False,
    header: tuple[str, ...] | None = None,
) -> bytes:
    name = member or f"{symbol}-15m-{date}.csv"
    encoded_rows = []
    if header is not None:
        encoded_rows.append(",".join(header))
    encoded_rows.extend(",".join(str(value) for value in row) for row in rows)
    data = ("\n".join(encoded_rows) + "\n").encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(name, data)
        if extra:
            archive.writestr("extra.csv", b"x\n")
    return stream.getvalue()


def _checksum(payload: bytes, symbol: str, date: str) -> bytes:
    return f"{hashlib.sha256(payload).hexdigest()}  {symbol}-15m-{date}.zip\n".encode()


def _validate(
    symbol: str,
    date: str,
    *,
    rows: list[list[object]] | None = None,
    count: int = 96,
    header: tuple[str, ...] | None = None,
) -> dict[str, object]:
    payload = _zip(
        symbol,
        date,
        _rows(date, count=count) if rows is None else rows,
        header=header,
    )
    return validate_target_archive_bytes(
        symbol=symbol,
        date=date,
        archive_bytes=payload,
        checksum_bytes=_checksum(payload, symbol, date),
    )


def _valid(symbol: str, date: str, *, count: int = 96) -> dict[str, object]:
    return _validate(symbol, date, count=count)


def test_frozen_roster_is_exact() -> None:
    assert TARGET_SYMBOLS == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )
    assert TARGET_DATES == (
        "2021-01-15",
        "2021-01-16",
        "2021-07-15",
        "2021-07-16",
        "2022-01-15",
        "2022-01-16",
        "2022-07-15",
        "2022-07-16",
    )


def test_complete_daily_archive_passes_structural_oracles() -> None:
    report = _valid("BTCUSDT", "2021-01-15")
    assert report["archive_available"] is True
    assert report["checksum_available"] is True
    assert report["checksum_verified"] is True
    assert report["row_count"] == 96
    assert report["expected_row_count"] == 96
    assert report["missing_grid_rows"] == 0
    assert report["open_times_strictly_increasing_unique"] is True
    assert report["native_grid_valid"] is True
    assert report["close_time_valid"] is True
    assert report["rows_inside_requested_utc_date"] is True
    assert report["schema_valid"] is True


def test_headerless_and_exact_header_normalize_identically_but_bad_header_fails() -> (
    None
):
    headerless = _validate("BTCUSDT", "2021-01-15")
    headered = _validate("BTCUSDT", "2021-01-15", header=_HEADER)
    assert headerless["header_present"] is False
    assert headered["header_present"] is True
    for field in (
        "row_count",
        "missing_grid_rows",
        "missing_grid_open_times_sha256",
        "first_open_time",
        "last_open_time",
        "schema_valid",
    ):
        assert headerless[field] == headered[field]

    bad = _validate(
        "BTCUSDT",
        "2021-01-15",
        header=(*_HEADER[:-1], "unexpected"),
    )
    assert bad["schema_valid"] is False


def test_wrong_member_extra_member_wrong_field_count_fail_closed() -> None:
    symbol, date = "BTCUSDT", "2021-01-15"
    rows = _rows(date)
    for payload in (
        _zip(symbol, date, rows, member="wrong.csv"),
        _zip(symbol, date, rows, extra=True),
    ):
        report = validate_target_archive_bytes(
            symbol=symbol,
            date=date,
            archive_bytes=payload,
            checksum_bytes=_checksum(payload, symbol, date),
        )
        assert report["schema_valid"] is False

    wrong_fields = _rows(date)
    wrong_fields[3] = wrong_fields[3][:-1]
    assert _validate(symbol, date, rows=wrong_fields)["schema_valid"] is False


def test_checksum_digest_and_filename_mismatch_fail_closed() -> None:
    symbol, date = "BTCUSDT", "2021-01-15"
    payload = _zip(symbol, date, _rows(date))
    wrong_digest = b"0" * 64 + f"  {symbol}-15m-{date}.zip\n".encode()
    report = validate_target_archive_bytes(
        symbol=symbol,
        date=date,
        archive_bytes=payload,
        checksum_bytes=wrong_digest,
    )
    assert report["checksum_verified"] is False
    assert report["schema_valid"] is True

    wrong_name = f"{hashlib.sha256(payload).hexdigest()}  wrong.zip\n".encode()
    report = validate_target_archive_bytes(
        symbol=symbol,
        date=date,
        archive_bytes=payload,
        checksum_bytes=wrong_name,
    )
    assert report["checksum_verified"] is False
    assert report["schema_valid"] is True


def test_timestamp_order_grid_date_and_close_time_fail_closed() -> None:
    symbol, date = "BTCUSDT", "2021-01-15"
    fixtures: list[list[list[object]]] = []

    off_grid = _rows(date)
    off_grid[3][0] = int(off_grid[3][0]) + 1
    fixtures.append(off_grid)

    wrong_close = _rows(date)
    wrong_close[3][6] = int(wrong_close[3][6]) + 1
    fixtures.append(wrong_close)

    duplicate = _rows(date)
    duplicate[3][0] = duplicate[2][0]
    duplicate[3][6] = duplicate[2][6]
    fixtures.append(duplicate)

    decreasing = _rows(date)
    decreasing[3], decreasing[4] = decreasing[4], decreasing[3]
    fixtures.append(decreasing)

    out_of_day = _rows(date)
    out_of_day[0][0] = _day_ms("2021-01-14")
    out_of_day[0][6] = _day_ms("2021-01-14") + 15 * 60 * 1000 - 1
    fixtures.append(out_of_day)

    for rows in fixtures:
        assert _validate(symbol, date, rows=rows)["schema_valid"] is False


def test_nonfinite_required_numeric_field_fails_closed() -> None:
    for column in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11):
        rows = _rows("2021-01-15")
        rows[4][column] = "nan"
        assert _validate("BTCUSDT", "2021-01-15", rows=rows)["schema_valid"] is False


def test_native_gap_is_recorded_exactly_and_never_filled() -> None:
    rows = _rows("2021-01-15")
    missing_open = int(rows[17][0])
    del rows[17]
    report = _validate("BTCUSDT", "2021-01-15", rows=rows)
    assert report["schema_valid"] is True
    assert report["row_count"] == 95
    assert report["missing_grid_rows"] == 1
    assert (
        report["missing_grid_open_times_sha256"]
        == hashlib.sha256(str(missing_open).encode("ascii")).hexdigest()
    )


def test_status_gate_distinguishes_pass_partial_incompatible_and_missing_source() -> (
    None
):
    reports = [
        _valid(symbol, date) for symbol in TARGET_SYMBOLS for date in TARGET_DATES
    ]
    assert decide_target_source_status(reports) == "PASS_USDM_15M_TARGET_SOURCE"

    partial = [dict(item) for item in reports]
    partial[0] = _valid(TARGET_SYMBOLS[0], TARGET_DATES[0], count=95)
    assert decide_target_source_status(partial) == "PARTIAL_USDM_15M_TARGET_SOURCE"

    missing = [dict(item) for item in reports]
    missing[0] = validate_target_archive_bytes(
        symbol=TARGET_SYMBOLS[0],
        date=TARGET_DATES[0],
        archive_bytes=None,
        checksum_bytes=None,
    )
    assert decide_target_source_status(missing) == "PARTIAL_USDM_15M_TARGET_SOURCE"

    incompatible = [dict(item) for item in reports]
    incompatible[0]["schema_valid"] = False
    assert (
        decide_target_source_status(incompatible)
        == "INCOMPATIBLE_USDM_15M_TARGET_SOURCE"
    )


def test_status_gate_rejects_duplicate_missing_or_unexpected_roster() -> None:
    reports = [
        _valid(symbol, date) for symbol in TARGET_SYMBOLS for date in TARGET_DATES
    ]
    assert len(reports) == 40

    for changed in (
        reports[:-1],
        [*reports[:-1], dict(reports[0])],
        [*reports[:-1], {**reports[-1], "symbol": "DOGEUSDT"}],
    ):
        assert (
            decide_target_source_status(changed)
            == "INCOMPATIBLE_USDM_15M_TARGET_SOURCE"
        )


def test_canonical_report_binds_authority_is_deterministic_and_result_blind() -> None:
    reports = [
        _valid(symbol, date) for symbol in TARGET_SYMBOLS for date in TARGET_DATES
    ]
    report = build_target_source_report(
        reports,
        validator_head="9" * 40,
        validator_verification_run_id=123,
    )
    reordered = build_target_source_report(
        list(reversed(reports)),
        validator_head="9" * 40,
        validator_verification_run_id=123,
    )
    raw = canonical_target_source_report_bytes(report)
    assert raw == canonical_target_source_report_bytes(reordered)
    assert json.loads(raw) == report
    assert report["status"] == "PASS_USDM_15M_TARGET_SOURCE"
    assert report["planned_archive_count"] == 40
    assert report["issue584_protocol_head"] == (
        "89ec1092e438d69657840567e684739ca0c4e3d5"
    )
    assert report["issue584_protocol_digest"] == (
        "bf0aa2db248e9dcab745e92e6bde54de6b48276a785eec74670e0dedba8cb5e3"
    )
    assert report["issue584_prereg_seal_run_id"] == 34932424904
    assert report["issue584_prereg_seal_artifact_id"] == 10381763665
    assert report["issue584_prereg_fresh_artifact_id"] == 10381379452
    assert report["target_relation_computed"] is False
    assert report["economic_values_inspected"] is False
    assert report["evaluation_pnl_inspected"] is False
    assert report["production_eligible"] is False
    assert report["live_trading_authorized"] is False
    assert isinstance(report["content_digest"], str)
    assert len(report["content_digest"]) == 64

    forbidden = {
        "target_return",
        "beta",
        "pnl",
        "mean_close",
        "median_close",
        "mean_volume",
        "volatility",
    }
    assert forbidden.isdisjoint(report)
    for entry in report["entries"]:
        assert forbidden.isdisjoint(entry)
