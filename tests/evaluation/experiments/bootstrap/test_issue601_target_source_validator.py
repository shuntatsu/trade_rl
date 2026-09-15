from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.issue601_target_source_validator import (
    TARGET_MONTHS,
    TARGET_SYMBOLS,
    TargetArchiveValidation,
    build_target_source_report,
    canonical_target_source_report_bytes,
    count_structural_target_windows,
    decide_target_source_status,
    expected_rows_in_month,
    load_target_source_report_bytes,
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
_HOUR_MS = 60 * 60 * 1000


def _month_start(month: str) -> datetime:
    return datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)


def _next_month(month: str) -> datetime:
    start = _month_start(month)
    if start.month == 12:
        return start.replace(year=start.year + 1, month=1)
    return start.replace(month=start.month + 1)


def _rows(month: str) -> list[list[object]]:
    start = int(_month_start(month).timestamp() * 1000)
    return [
        [
            start + index * _HOUR_MS,
            "1",
            "1",
            "1",
            "1",
            "0",
            start + (index + 1) * _HOUR_MS - 1,
            "0",
            "0",
            "0",
            "0",
            "0",
        ]
        for index in range(expected_rows_in_month(month))
    ]


def _zip(
    symbol: str,
    month: str,
    rows: list[list[object]],
    *,
    member: str | None = None,
    extra: bool = False,
    header: tuple[str, ...] | None = None,
) -> bytes:
    name = member or f"{symbol}-1h-{month}.csv"
    encoded: list[str] = []
    if header is not None:
        encoded.append(",".join(header))
    encoded.extend(",".join(str(value) for value in row) for row in rows)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(name, ("\n".join(encoded) + "\n").encode())
        if extra:
            archive.writestr("extra.csv", b"x\n")
    return stream.getvalue()


def _checksum(payload: bytes, symbol: str, month: str) -> bytes:
    digest = hashlib.sha256(payload).hexdigest()
    return f"{digest}  {symbol}-1h-{month}.zip\n".encode()


def _validate(
    symbol: str = "BTCUSDT",
    month: str = "2021-01",
    *,
    rows: list[list[object]] | None = None,
    checksum: bool = True,
    header: tuple[str, ...] | None = None,
) -> TargetArchiveValidation:
    payload = _zip(symbol, month, _rows(month) if rows is None else rows, header=header)
    return validate_target_archive_bytes(
        symbol=symbol,
        month=month,
        archive_bytes=payload,
        checksum_bytes=_checksum(payload, symbol, month) if checksum else None,
    )


def _ideal_open_times() -> tuple[int, ...]:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    cutoff = datetime(2023, 1, 1, tzinfo=UTC)
    hours = int((cutoff - start).total_seconds() // 3600)
    start_ms = int(start.timestamp() * 1000)
    return tuple(start_ms + index * _HOUR_MS for index in range(hours))


def test_frozen_roster_and_calendar_row_counts_are_exact() -> None:
    assert TARGET_SYMBOLS == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )
    assert TARGET_MONTHS[0] == "2021-01"
    assert TARGET_MONTHS[-1] == "2022-12"
    assert len(TARGET_MONTHS) == 24
    assert expected_rows_in_month("2021-01") == 31 * 24
    assert expected_rows_in_month("2021-02") == 28 * 24
    assert expected_rows_in_month("2022-12") == 31 * 24
    with pytest.raises(ValueError):
        expected_rows_in_month("2023-01")


def test_complete_month_is_structurally_valid_without_exposing_economic_values() -> None:
    validation = _validate()
    report = validation.report
    assert report["archive_available"] is True
    assert report["checksum_available"] is True
    assert report["checksum_verified"] is True
    assert report["row_count"] == 744
    assert report["expected_row_count"] == 744
    assert report["missing_grid_rows"] == 0
    assert report["open_times_strictly_increasing_unique"] is True
    assert report["native_grid_valid"] is True
    assert report["close_time_valid"] is True
    assert report["rows_inside_requested_utc_month"] is True
    assert report["schema_valid"] is True
    assert len(validation.open_times) == 744

    forbidden = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "return",
        "premium",
        "beta",
        "alpha",
        "pnl",
    }
    assert forbidden.isdisjoint(report)


def test_headerless_and_exact_header_normalize_but_bad_header_fails() -> None:
    headerless = _validate()
    headered = _validate(header=_HEADER)
    assert headerless.report["header_present"] is False
    assert headered.report["header_present"] is True
    assert headerless.report["schema_valid"] is True
    assert headered.report["schema_valid"] is True
    assert headerless.open_times == headered.open_times

    bad = _validate(header=(*_HEADER[:-1], "unexpected"))
    assert bad.report["schema_valid"] is False


def test_checksum_state_is_separate_from_archive_schema() -> None:
    missing = _validate(checksum=False)
    assert missing.report["schema_valid"] is True
    assert missing.report["checksum_available"] is False
    assert missing.report["checksum_verified"] is False

    symbol, month = "BTCUSDT", "2021-01"
    payload = _zip(symbol, month, _rows(month))
    wrong_digest = b"0" * 64 + f"  {symbol}-1h-{month}.zip\n".encode()
    mismatch = validate_target_archive_bytes(
        symbol=symbol,
        month=month,
        archive_bytes=payload,
        checksum_bytes=wrong_digest,
    )
    assert mismatch.report["schema_valid"] is True
    assert mismatch.report["checksum_available"] is True
    assert mismatch.report["checksum_verified"] is False

    wrong_name = f"{hashlib.sha256(payload).hexdigest()}  wrong.zip\n".encode()
    mismatch_name = validate_target_archive_bytes(
        symbol=symbol,
        month=month,
        archive_bytes=payload,
        checksum_bytes=wrong_name,
    )
    assert mismatch_name.report["schema_valid"] is True
    assert mismatch_name.report["checksum_verified"] is False


def test_member_schema_order_grid_month_and_close_time_fail_closed() -> None:
    symbol, month = "BTCUSDT", "2021-01"
    rows = _rows(month)
    for payload in (
        _zip(symbol, month, rows, member="wrong.csv"),
        _zip(symbol, month, rows, extra=True),
    ):
        result = validate_target_archive_bytes(
            symbol=symbol,
            month=month,
            archive_bytes=payload,
            checksum_bytes=_checksum(payload, symbol, month),
        )
        assert result.report["schema_valid"] is False

    wrong_fields = _rows(month)
    wrong_fields[3] = wrong_fields[3][:-1]
    assert _validate(rows=wrong_fields).report["schema_valid"] is False

    off_grid = _rows(month)
    off_grid[3][0] = int(off_grid[3][0]) + 1
    assert _validate(rows=off_grid).report["schema_valid"] is False

    wrong_close = _rows(month)
    wrong_close[3][6] = int(wrong_close[3][6]) + 1
    assert _validate(rows=wrong_close).report["schema_valid"] is False

    duplicate = _rows(month)
    duplicate[3][0] = duplicate[2][0]
    duplicate[3][6] = duplicate[2][6]
    assert _validate(rows=duplicate).report["schema_valid"] is False

    reordered = _rows(month)
    reordered[3], reordered[4] = reordered[4], reordered[3]
    assert _validate(rows=reordered).report["schema_valid"] is False

    outside = _rows(month)
    previous_month = int(datetime(2020, 12, 31, 23, tzinfo=UTC).timestamp() * 1000)
    outside[0][0] = previous_month
    outside[0][6] = previous_month + _HOUR_MS - 1
    assert _validate(rows=outside).report["schema_valid"] is False


def test_nonfinite_or_malformed_required_numeric_field_fails_closed() -> None:
    for column in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11):
        rows = _rows("2021-01")
        rows[4][column] = "nan"
        assert _validate(rows=rows).report["schema_valid"] is False


def test_sparse_on_grid_month_is_valid_and_records_exact_missing_digest() -> None:
    rows = _rows("2021-01")
    missing_open = int(rows[17][0])
    del rows[17]
    result = _validate(rows=rows)
    assert result.report["schema_valid"] is True
    assert result.report["row_count"] == 743
    assert result.report["missing_grid_rows"] == 1
    assert result.open_times[16] < missing_open < result.open_times[17]
    assert result.report["missing_grid_open_times_sha256"] == hashlib.sha256(
        str(missing_open).encode("ascii")
    ).hexdigest()


def test_timestamp_only_target_window_count_uses_raw_t_through_t_plus_24h() -> None:
    ideal = _ideal_open_times()
    summary = count_structural_target_windows(ideal)
    assert summary["nominal_decisions"] == 17_495
    assert summary["structurally_present_windows"] == 17_495
    assert summary["structurally_missing_windows"] == 0

    last_endpoint = int(datetime(2022, 12, 31, 23, tzinfo=UTC).timestamp() * 1000)
    without_last_endpoint = tuple(value for value in ideal if value != last_endpoint)
    missing = count_structural_target_windows(without_last_endpoint)
    assert missing["structurally_present_windows"] == 17_494
    assert missing["structurally_missing_windows"] == 1

    irrelevant = int(datetime(2021, 1, 1, 0, tzinfo=UTC).timestamp() * 1000)
    without_irrelevant = tuple(value for value in ideal if value != irrelevant)
    unchanged = count_structural_target_windows(without_irrelevant)
    assert unchanged["structurally_present_windows"] == 17_495

    assert int(datetime(2022, 12, 30, 23, tzinfo=UTC).timestamp() * 1000) + 24 * _HOUR_MS == last_endpoint
    assert last_endpoint < int(datetime(2023, 1, 1, tzinfo=UTC).timestamp() * 1000)


def test_status_gate_distinguishes_pass_partial_and_incompatible() -> None:
    ideal_summary = {
        symbol: count_structural_target_windows(_ideal_open_times())
        for symbol in TARGET_SYMBOLS
    }
    reports = []
    for symbol in TARGET_SYMBOLS:
        for month in TARGET_MONTHS:
            reports.append(_validate(symbol, month).report)
    assert len(reports) == 120
    assert decide_target_source_status(reports, ideal_summary) == "PASS_USDM_1H_TARGET_SOURCE"

    missing_checksum = [dict(item) for item in reports]
    missing_checksum[0]["checksum_available"] = False
    missing_checksum[0]["checksum_text"] = None
    missing_checksum[0]["checksum_digest"] = None
    missing_checksum[0]["checksum_verified"] = False
    assert decide_target_source_status(missing_checksum, ideal_summary) == "PARTIAL_USDM_1H_TARGET_SOURCE"

    low_coverage = {symbol: dict(summary) for symbol, summary in ideal_summary.items()}
    low_coverage[TARGET_SYMBOLS[0]]["structurally_present_windows"] = 16_620
    low_coverage[TARGET_SYMBOLS[0]]["structurally_missing_windows"] = 875
    assert decide_target_source_status(reports, low_coverage) == "PARTIAL_USDM_1H_TARGET_SOURCE"

    mismatch = [dict(item) for item in reports]
    mismatch[0]["checksum_verified"] = False
    assert decide_target_source_status(mismatch, ideal_summary) == "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"

    corrupt = [dict(item) for item in reports]
    corrupt[0]["schema_valid"] = False
    assert decide_target_source_status(corrupt, ideal_summary) == "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"


def test_status_gate_rejects_duplicate_missing_unexpected_or_bool_spoofed_roster() -> None:
    reports = [
        _validate(symbol, month).report
        for symbol in TARGET_SYMBOLS
        for month in TARGET_MONTHS
    ]
    summaries = {
        symbol: count_structural_target_windows(_ideal_open_times())
        for symbol in TARGET_SYMBOLS
    }
    for changed in (
        reports[:-1],
        [*reports[:-1], dict(reports[0])],
        [*reports[:-1], {**reports[-1], "symbol": "DOGEUSDT"}],
        [{**reports[0], "row_count": True}, *reports[1:]],
    ):
        assert decide_target_source_status(changed, summaries) == "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"


def test_canonical_report_binds_issue600_authority_is_deterministic_and_result_blind() -> None:
    validations = [
        _validate(symbol, month)
        for symbol in TARGET_SYMBOLS
        for month in TARGET_MONTHS
    ]
    report = build_target_source_report(
        validations,
        validator_head="9" * 40,
        validator_verification_run_id=123,
    )
    raw = canonical_target_source_report_bytes(report)
    restored = load_target_source_report_bytes(raw)
    assert restored == report
    assert json.loads(raw) == report
    assert report["status"] == "PASS_USDM_1H_TARGET_SOURCE"
    assert report["planned_archive_count"] == 120
    assert report["issue600_protocol_head"] == "6ffc414baa10e91f34df258fe5cfabacce65fd77"
    assert report["issue600_protocol_digest"] == (
        "18bf9625502eccbe475d157df90635f1c4645d3c4cced48ec0fc565723206731"
    )
    assert report["issue600_prereg_seal_run_id"] == 34959849347
    assert report["issue600_prereg_seal_artifact_id"] == 10392936042
    assert report["issue600_prereg_fresh_artifact_id"] == 10391714897
    for symbol in TARGET_SYMBOLS:
        assert report["target_windows_by_symbol"][symbol]["nominal_decisions"] == 17_495
        assert report["target_windows_by_symbol"][symbol]["structurally_present_windows"] == 17_495
    for field in (
        "economic_values_inspected",
        "target_price_values_inspected",
        "target_return_computed",
        "premium_values_inspected",
        "training_relation_computed",
        "evaluation_pnl_inspected",
        "final_test_authorized",
        "production_eligible",
        "live_trading_authorized",
    ):
        assert report[field] is False

    forbidden = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "return",
        "premium",
        "beta",
        "alpha",
        "pnl",
        "sharpe",
        "ic",
    }
    assert forbidden.isdisjoint(report)
    for entry in report["entries"]:
        assert forbidden.isdisjoint(entry)


def test_loader_rejects_unknown_fields_authority_forgery_and_resigned_status_forgery() -> None:
    validations = [
        _validate(symbol, month)
        for symbol in TARGET_SYMBOLS
        for month in TARGET_MONTHS
    ]
    report = build_target_source_report(
        validations,
        validator_head="9" * 40,
        validator_verification_run_id=123,
    )

    unknown = {**report, "mean_close": 1.0}
    unsigned = dict(unknown)
    unsigned.pop("content_digest")
    unknown["content_digest"] = content_digest(unsigned)
    with pytest.raises(ValueError):
        load_target_source_report_bytes(canonical_json_bytes(unknown))

    authority = dict(report)
    authority["issue600_protocol_head"] = "8" * 40
    unsigned = dict(authority)
    unsigned.pop("content_digest")
    authority["content_digest"] = content_digest(unsigned)
    with pytest.raises(ValueError):
        load_target_source_report_bytes(canonical_json_bytes(authority))

    forged = json.loads(canonical_target_source_report_bytes(report))
    forged["target_windows_by_symbol"][TARGET_SYMBOLS[0]][
        "structurally_present_windows"
    ] = 16_620
    forged["target_windows_by_symbol"][TARGET_SYMBOLS[0]][
        "structurally_missing_windows"
    ] = 875
    forged["status"] = "PASS_USDM_1H_TARGET_SOURCE"
    unsigned = dict(forged)
    unsigned.pop("content_digest")
    forged["content_digest"] = content_digest(unsigned)
    with pytest.raises(ValueError):
        load_target_source_report_bytes(canonical_json_bytes(forged))


def test_month_bounds_are_exact_and_never_require_2023_archive() -> None:
    assert _next_month("2022-12") == datetime(2023, 1, 1, tzinfo=UTC)
    last_decision = datetime(2022, 12, 30, 23, tzinfo=UTC)
    endpoint = last_decision + timedelta(hours=24)
    assert endpoint == datetime(2022, 12, 31, 23, tzinfo=UTC)
    assert endpoint < datetime(2023, 1, 1, tzinfo=UTC)
    assert "2023-01" not in TARGET_MONTHS
