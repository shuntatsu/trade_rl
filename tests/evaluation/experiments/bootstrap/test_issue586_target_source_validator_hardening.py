from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, datetime

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.issue586_target_source_validator import (
    TARGET_DATES,
    TARGET_SYMBOLS,
    build_target_source_report,
    canonical_target_source_report_bytes,
    decide_target_source_status,
    validate_target_archive_bytes,
)


def _day_ms(date: str) -> int:
    return int(
        datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000
    )


def _valid(
    symbol: str,
    date: str,
    *,
    checksum_mode: str = "valid",
) -> dict[str, object]:
    start = _day_ms(date)
    step = 15 * 60 * 1000
    rows = [
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
        for i in range(96)
    ]
    body = (
        "\n".join(",".join(str(value) for value in row) for row in rows) + "\n"
    ).encode()
    stream = io.BytesIO()
    member = f"{symbol}-15m-{date}.csv"
    archive_name = f"{symbol}-15m-{date}.zip"
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member, body)
    payload = stream.getvalue()
    if checksum_mode == "valid":
        checksum: bytes | None = (
            f"{hashlib.sha256(payload).hexdigest()}  {archive_name}\n".encode()
        )
    elif checksum_mode == "missing":
        checksum = None
    elif checksum_mode == "mismatch":
        checksum = ("0" * 64 + f"  {archive_name}\n").encode()
    else:
        raise AssertionError("unknown checksum_mode")
    return validate_target_archive_bytes(
        symbol=symbol,
        date=date,
        archive_bytes=payload,
        checksum_bytes=checksum,
    )


def _reports() -> list[dict[str, object]]:
    return [_valid(symbol, date) for symbol in TARGET_SYMBOLS for date in TARGET_DATES]


def test_builder_rejects_impossible_grid_count_semantics() -> None:
    reports = _reports()
    reports[0] = dict(reports[0])
    reports[0]["row_count"] = 97
    reports[0]["missing_grid_rows"] = 0
    reports[0]["schema_valid"] = True
    reports[0]["native_grid_valid"] = True
    reports[0]["open_times_strictly_increasing_unique"] = True
    reports[0]["rows_inside_requested_utc_date"] = True

    with pytest.raises(ValueError):
        build_target_source_report(
            reports,
            validator_head="9" * 40,
            validator_verification_run_id=123,
        )


def test_builder_rejects_bool_int_spoofing_in_nested_entry() -> None:
    reports = _reports()
    reports[0] = dict(reports[0])
    reports[0]["header_present"] = 1

    with pytest.raises(ValueError):
        build_target_source_report(
            reports,
            validator_head="9" * 40,
            validator_verification_run_id=123,
        )


def test_missing_checksum_keeps_structure_valid_and_is_partial() -> None:
    reports = _reports()
    missing = _valid(TARGET_SYMBOLS[0], TARGET_DATES[0], checksum_mode="missing")
    assert missing["archive_available"] is True
    assert missing["checksum_available"] is False
    assert missing["checksum_verified"] is False
    assert missing["schema_valid"] is True
    reports[0] = missing
    assert decide_target_source_status(reports) == "PARTIAL_USDM_15M_TARGET_SOURCE"


def test_checksum_mismatch_keeps_structure_valid_but_is_incompatible() -> None:
    reports = _reports()
    mismatch = _valid(TARGET_SYMBOLS[0], TARGET_DATES[0], checksum_mode="mismatch")
    assert mismatch["archive_available"] is True
    assert mismatch["checksum_available"] is True
    assert mismatch["checksum_verified"] is False
    assert mismatch["schema_valid"] is True
    reports[0] = mismatch
    assert (
        decide_target_source_status(reports)
        == "INCOMPATIBLE_USDM_15M_TARGET_SOURCE"
    )


def test_canonical_loader_recomputes_summary_and_rejects_redigested_tampering() -> None:
    report = build_target_source_report(
        _reports(),
        validator_head="9" * 40,
        validator_verification_run_id=123,
    )
    report["available_archive_count"] = 39
    report.pop("content_digest")
    report["content_digest"] = content_digest(report)

    with pytest.raises(ValueError):
        canonical_target_source_report_bytes(report)


def test_canonical_loader_rejects_nested_economic_field_even_after_redigest() -> None:
    report = build_target_source_report(
        _reports(),
        validator_head="9" * 40,
        validator_verification_run_id=123,
    )
    entries = report["entries"]
    assert isinstance(entries, list)
    first = dict(entries[0])
    first["mean_close"] = 1.0
    entries[0] = first
    report.pop("content_digest")
    report["content_digest"] = content_digest(report)

    with pytest.raises(ValueError):
        canonical_target_source_report_bytes(report)
