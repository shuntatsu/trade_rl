from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, datetime, timedelta

import pytest

from trade_rl.evaluation.experiments.bootstrap.issue586_target_source_validator import (
    TARGET_DATES,
    TARGET_SYMBOLS,
    build_target_source_report,
    canonical_target_source_report_bytes,
    decide_target_source_status,
    validate_target_archive_bytes,
)


def _day_ms(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def _rows(date: str, *, count: int = 96) -> list[list[object]]:
    start = _day_ms(date)
    step = 15 * 60 * 1000
    return [[start + i * step, "1", "1", "1", "1", "0", start + (i + 1) * step - 1, "0", "0", "0", "0", "0"] for i in range(count)]


def _zip(symbol: str, date: str, rows: list[list[object]], *, member: str | None = None, extra: bool = False) -> bytes:
    name = member or f"{symbol}-15m-{date}.csv"
    data = ("\n".join(",".join(str(v) for v in row) for row in rows) + "\n").encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(name, data)
        if extra:
            archive.writestr("extra.csv", b"x\n")
    return stream.getvalue()


def _checksum(payload: bytes, symbol: str, date: str) -> bytes:
    return f"{hashlib.sha256(payload).hexdigest()}  {symbol}-15m-{date}.zip\n".encode()


def _valid(symbol: str, date: str, *, count: int = 96) -> dict[str, object]:
    payload = _zip(symbol, date, _rows(date, count=count))
    return validate_target_archive_bytes(symbol=symbol, date=date, archive_bytes=payload, checksum_bytes=_checksum(payload, symbol, date))


def test_frozen_roster_is_exact() -> None:
    assert TARGET_SYMBOLS == ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
    assert TARGET_DATES == ("2021-01-15", "2021-01-16", "2021-07-15", "2021-07-16", "2022-01-15", "2022-01-16", "2022-07-15", "2022-07-16")


def test_complete_daily_archive_passes_structural_oracles() -> None:
    report = _valid("BTCUSDT", "2021-01-15")
    assert report["archive_available"] is True
    assert report["checksum_verified"] is True
    assert report["row_count"] == 96
    assert report["open_times_strictly_increasing_unique"] is True
    assert report["native_grid_valid"] is True
    assert report["close_time_valid"] is True
    assert report["rows_inside_requested_utc_date"] is True
    assert report["schema_valid"] is True


def test_wrong_member_extra_member_offgrid_and_close_time_fail_closed() -> None:
    symbol, date = "BTCUSDT", "2021-01-15"
    rows = _rows(date)
    for payload in (
        _zip(symbol, date, rows, member="wrong.csv"),
        _zip(symbol, date, rows, extra=True),
    ):
        report = validate_target_archive_bytes(symbol=symbol, date=date, archive_bytes=payload, checksum_bytes=_checksum(payload, symbol, date))
        assert report["schema_valid"] is False
    bad = _rows(date)
    bad[3][0] = int(bad[3][0]) + 1
    payload = _zip(symbol, date, bad)
    assert validate_target_archive_bytes(symbol=symbol, date=date, archive_bytes=payload, checksum_bytes=_checksum(payload, symbol, date))["schema_valid"] is False
    bad = _rows(date)
    bad[3][6] = int(bad[3][6]) + 1
    payload = _zip(symbol, date, bad)
    assert validate_target_archive_bytes(symbol=symbol, date=date, archive_bytes=payload, checksum_bytes=_checksum(payload, symbol, date))["schema_valid"] is False


def test_status_gate_distinguishes_pass_partial_and_incompatible() -> None:
    reports = [_valid(symbol, date) for symbol in TARGET_SYMBOLS for date in TARGET_DATES]
    assert decide_target_source_status(reports) == "PASS_USDM_15M_TARGET_SOURCE"
    partial = [dict(item) for item in reports]
    partial[0] = _valid(TARGET_SYMBOLS[0], TARGET_DATES[0], count=95)
    assert decide_target_source_status(partial) == "PARTIAL_USDM_15M_TARGET_SOURCE"
    incompatible = [dict(item) for item in reports]
    incompatible[0]["schema_valid"] = False
    assert decide_target_source_status(incompatible) == "INCOMPATIBLE_USDM_15M_TARGET_SOURCE"


def test_canonical_report_binds_issue584_authority_and_stays_result_blind() -> None:
    reports = [_valid(symbol, date) for symbol in TARGET_SYMBOLS for date in TARGET_DATES]
    report = build_target_source_report(reports, validator_head="9" * 40, validator_verification_run_id=123)
    raw = canonical_target_source_report_bytes(report)
    assert report["status"] == "PASS_USDM_15M_TARGET_SOURCE"
    assert report["planned_archive_count"] == 40
    assert report["issue584_protocol_head"] == "89ec1092e438d69657840567e684739ca0c4e3d5"
    assert report["issue584_protocol_digest"] == "bf0aa2db248e9dcab745e92e6bde54de6b48276a785eec74670e0dedba8cb5e3"
    assert report["target_relation_computed"] is False
    assert report["economic_values_inspected"] is False
    assert report["evaluation_pnl_inspected"] is False
    assert b"return" not in raw.lower()
    assert b"beta" not in raw.lower()
