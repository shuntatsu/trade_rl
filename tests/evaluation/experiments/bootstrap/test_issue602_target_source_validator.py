from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from functools import lru_cache

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.issue602_target_source_validator import (
    TARGET_MONTHS,
    TARGET_SYMBOLS,
    build_target_source_report,
    canonical_target_source_report_bytes,
    decide_target_source_status,
    expected_month_row_count,
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
_INTERVAL_MS = 60 * 60 * 1000


def _month_start_ms(month: str) -> int:
    return int(
        datetime.strptime(month + "-01", "%Y-%m-%d").replace(tzinfo=UTC).timestamp()
        * 1000
    )


def _rows(month: str) -> list[list[object]]:
    start = _month_start_ms(month)
    return [
        [
            start + index * _INTERVAL_MS,
            "1",
            "1",
            "1",
            "1",
            "0",
            start + (index + 1) * _INTERVAL_MS - 1,
            "0",
            "0",
            "0",
            "0",
            "0",
        ]
        for index in range(expected_month_row_count(month))
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
    return f"{hashlib.sha256(payload).hexdigest()}  {symbol}-1h-{month}.zip\n".encode()


def _validate(
    symbol: str,
    month: str,
    *,
    rows: list[list[object]] | None = None,
    header: tuple[str, ...] | None = None,
    checksum_bytes: bytes | None | object = ...,
) -> dict[str, object]:
    payload = _zip(symbol, month, _rows(month) if rows is None else rows, header=header)
    checksum = (
        _checksum(payload, symbol, month) if checksum_bytes is ... else checksum_bytes
    )
    assert checksum is None or isinstance(checksum, bytes)
    return validate_target_archive_bytes(
        symbol=symbol,
        month=month,
        archive_bytes=payload,
        checksum_bytes=checksum,
    )


@lru_cache(maxsize=None)
def _cached_valid(symbol: str, month: str) -> tuple[tuple[str, object], ...]:
    report = _validate(symbol, month)
    return tuple(report.items())


def _valid(symbol: str, month: str) -> dict[str, object]:
    return dict(_cached_valid(symbol, month))


def _reports() -> list[dict[str, object]]:
    return [
        _valid(symbol, month) for symbol in TARGET_SYMBOLS for month in TARGET_MONTHS
    ]


def test_frozen_roster_and_calendar_counts_are_exact() -> None:
    assert TARGET_SYMBOLS == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )
    assert TARGET_MONTHS == tuple(
        f"{year}-{month:02d}" for year in (2021, 2022) for month in range(1, 13)
    )
    assert len(TARGET_SYMBOLS) * len(TARGET_MONTHS) == 120
    assert expected_month_row_count("2021-01") == 744
    assert expected_month_row_count("2021-02") == 672
    assert expected_month_row_count("2020-02") == 696
    assert expected_month_row_count("2021-04") == 720


def test_complete_month_passes_structural_oracles() -> None:
    report = _valid("BTCUSDT", "2021-01")
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
    assert report["dataset_timestamp_offset_ms"] == _INTERVAL_MS
    assert report["dataset_timestamp_semantics"] == "completed_bar_close_boundary"


def test_headerless_and_exact_header_normalize_identically_bad_header_fails() -> None:
    headerless = _validate("BTCUSDT", "2021-02")
    headered = _validate("BTCUSDT", "2021-02", header=_HEADER)
    assert headerless["header_present"] is False
    assert headered["header_present"] is True
    for field in (
        "row_count",
        "expected_row_count",
        "missing_grid_rows",
        "missing_grid_open_times_sha256",
        "first_open_time",
        "last_open_time",
        "first_close_time",
        "last_close_time",
        "schema_valid",
    ):
        assert headerless[field] == headered[field]

    bad = _validate(
        "BTCUSDT",
        "2021-02",
        header=(*_HEADER[:-1], "unexpected"),
    )
    assert bad["schema_valid"] is False


def test_wrong_member_extra_member_wrong_field_count_fail_closed() -> None:
    symbol, month = "BTCUSDT", "2021-01"
    rows = _rows(month)
    for payload in (
        _zip(symbol, month, rows, member="wrong.csv"),
        _zip(symbol, month, rows, extra=True),
    ):
        report = validate_target_archive_bytes(
            symbol=symbol,
            month=month,
            archive_bytes=payload,
            checksum_bytes=_checksum(payload, symbol, month),
        )
        assert report["schema_valid"] is False

    wrong_fields = _rows(month)
    wrong_fields[3] = wrong_fields[3][:-1]
    assert _validate(symbol, month, rows=wrong_fields)["schema_valid"] is False


def test_checksum_digest_filename_and_missing_checksum_have_distinct_semantics() -> (
    None
):
    symbol, month = "BTCUSDT", "2021-01"
    payload = _zip(symbol, month, _rows(month))

    wrong_digest = b"0" * 64 + f"  {symbol}-1h-{month}.zip\n".encode()
    mismatch = validate_target_archive_bytes(
        symbol=symbol,
        month=month,
        archive_bytes=payload,
        checksum_bytes=wrong_digest,
    )
    assert mismatch["checksum_verified"] is False
    assert mismatch["schema_valid"] is True

    wrong_name = f"{hashlib.sha256(payload).hexdigest()}  wrong.zip\n".encode()
    mismatch_name = validate_target_archive_bytes(
        symbol=symbol,
        month=month,
        archive_bytes=payload,
        checksum_bytes=wrong_name,
    )
    assert mismatch_name["checksum_verified"] is False
    assert mismatch_name["schema_valid"] is True

    missing_checksum = validate_target_archive_bytes(
        symbol=symbol,
        month=month,
        archive_bytes=payload,
        checksum_bytes=None,
    )
    assert missing_checksum["archive_available"] is True
    assert missing_checksum["checksum_available"] is False
    assert missing_checksum["schema_valid"] is True


def test_timestamp_order_grid_month_and_close_time_fail_closed() -> None:
    symbol, month = "BTCUSDT", "2021-01"
    fixtures: list[list[list[object]]] = []

    off_grid = _rows(month)
    off_grid[3][0] = int(off_grid[3][0]) + 1
    fixtures.append(off_grid)

    wrong_close = _rows(month)
    wrong_close[3][6] = int(wrong_close[3][6]) + 1
    fixtures.append(wrong_close)

    duplicate = _rows(month)
    duplicate[3][0] = duplicate[2][0]
    duplicate[3][6] = duplicate[2][6]
    fixtures.append(duplicate)

    decreasing = _rows(month)
    decreasing[3], decreasing[4] = decreasing[4], decreasing[3]
    fixtures.append(decreasing)

    out_of_month = _rows(month)
    out_of_month[0][0] = _month_start_ms("2020-12")
    out_of_month[0][6] = int(out_of_month[0][0]) + _INTERVAL_MS - 1
    fixtures.append(out_of_month)

    for rows in fixtures:
        assert _validate(symbol, month, rows=rows)["schema_valid"] is False


def test_nonfinite_nonpositive_and_negative_structural_numeric_fields_fail_closed() -> (
    None
):
    for column in (1, 2, 3, 4, 5, 7, 9, 10, 11):
        rows = _rows("2021-01")
        rows[4][column] = "nan"
        assert _validate("BTCUSDT", "2021-01", rows=rows)["schema_valid"] is False

    for column in (1, 2, 3, 4):
        rows = _rows("2021-01")
        rows[4][column] = "0"
        assert _validate("BTCUSDT", "2021-01", rows=rows)["schema_valid"] is False

    for column in (5, 7, 9, 10):
        rows = _rows("2021-01")
        rows[4][column] = "-1"
        assert _validate("BTCUSDT", "2021-01", rows=rows)["schema_valid"] is False

    rows = _rows("2021-01")
    rows[4][8] = "-1"
    assert _validate("BTCUSDT", "2021-01", rows=rows)["schema_valid"] is False


def test_native_gap_is_recorded_exactly_and_never_filled() -> None:
    rows = _rows("2021-02")
    missing_open = int(rows[17][0])
    del rows[17]
    report = _validate("BTCUSDT", "2021-02", rows=rows)
    assert report["schema_valid"] is True
    assert report["row_count"] == 671
    assert report["expected_row_count"] == 672
    assert report["missing_grid_rows"] == 1
    assert (
        report["missing_grid_open_times_sha256"]
        == hashlib.sha256(str(missing_open).encode("ascii")).hexdigest()
    )


def test_status_gate_allows_explicit_native_gaps_but_distinguishes_missing_and_bad() -> (
    None
):
    reports = _reports()
    assert decide_target_source_status(reports) == "PASS_USDM_1H_TARGET_SOURCE"

    sparse = [dict(item) for item in reports]
    rows = _rows(TARGET_MONTHS[0])
    del rows[11]
    sparse[0] = _validate(TARGET_SYMBOLS[0], TARGET_MONTHS[0], rows=rows)
    assert decide_target_source_status(sparse) == "PASS_USDM_1H_TARGET_SOURCE"

    missing_archive = [dict(item) for item in reports]
    missing_archive[0] = validate_target_archive_bytes(
        symbol=TARGET_SYMBOLS[0],
        month=TARGET_MONTHS[0],
        archive_bytes=None,
        checksum_bytes=None,
    )
    assert (
        decide_target_source_status(missing_archive) == "PARTIAL_USDM_1H_TARGET_SOURCE"
    )

    missing_checksum = [dict(item) for item in reports]
    symbol, month = TARGET_SYMBOLS[0], TARGET_MONTHS[0]
    payload = _zip(symbol, month, _rows(month))
    missing_checksum[0] = validate_target_archive_bytes(
        symbol=symbol,
        month=month,
        archive_bytes=payload,
        checksum_bytes=None,
    )
    assert (
        decide_target_source_status(missing_checksum) == "PARTIAL_USDM_1H_TARGET_SOURCE"
    )

    bad_checksum = [dict(item) for item in reports]
    bad_checksum[0] = dict(bad_checksum[0])
    bad_checksum[0]["checksum_verified"] = False
    assert (
        decide_target_source_status(bad_checksum)
        == "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"
    )

    incompatible = [dict(item) for item in reports]
    incompatible[0] = dict(incompatible[0])
    incompatible[0]["schema_valid"] = False
    assert (
        decide_target_source_status(incompatible)
        == "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"
    )


def test_status_gate_rejects_duplicate_missing_unexpected_roster_and_bool_spoof() -> (
    None
):
    reports = _reports()
    assert len(reports) == 120

    for changed in (
        reports[:-1],
        [*reports[:-1], dict(reports[0])],
        [*reports[:-1], {**reports[-1], "symbol": "DOGEUSDT"}],
        [*reports[:-1], {**reports[-1], "month": "2023-01"}],
    ):
        assert (
            decide_target_source_status(changed) == "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"
        )

    spoof = [dict(item) for item in reports]
    spoof[0] = dict(spoof[0])
    spoof[0]["header_present"] = 1
    assert decide_target_source_status(spoof) == "INCOMPATIBLE_USDM_1H_TARGET_SOURCE"


def test_builder_rejects_impossible_nested_structural_semantics() -> None:
    reports = _reports()
    reports[0] = dict(reports[0])
    reports[0]["row_count"] = int(reports[0]["expected_row_count"]) + 1
    reports[0]["missing_grid_rows"] = 0
    reports[0]["schema_valid"] = True
    with pytest.raises(ValueError):
        build_target_source_report(
            reports,
            validator_head="9" * 40,
            validator_verification_run_id=123,
        )


def test_canonical_report_binds_completed_endpoint_authority_and_is_result_blind() -> (
    None
):
    reports = _reports()
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
    assert load_target_source_report_bytes(raw) == report

    assert report["schema_version"] == "issue602_usdm_1h_target_source_report_v1"
    assert report["issue_number"] == 602
    assert report["premium_prereg_issue"] == 600
    assert report["issue600_protocol_head"] == (
        "da97eb59b08ef042a96ebea4aaa645af02b450e9"
    )
    assert report["issue600_protocol_digest"] == (
        "c78d556f94e247d5bdc1c0e9a891d160e4c48c4f0b62cd5be7600874b3017d22"
    )
    assert report["issue600_full_verify_run_id"] == 34958973042
    assert report["issue600_digest_verify_run_id"] == 34959535015
    assert report["issue600_seal_run_id"] == 34959723495
    assert report["issue600_seal_artifact_id"] == 10392597937
    assert report["issue600_fresh_artifact_id"] == 10392916717
    assert report["issue600_audit_recovery_run_id"] == 34959887850
    assert report["issue600_audit_artifact_id"] == 10392966885
    assert report["superseded_raw_endpoint_head"] == (
        "9e0345a9f51f56cd84efcb856be810f97a4b8ca1"
    )
    assert report["superseded_raw_endpoint_authority_used"] is False
    assert report["dataset_timestamp_semantics"] == "completed_bar_close_boundary"
    assert report["dataset_timestamp_offset_ms"] == _INTERVAL_MS
    assert report["status"] == "PASS_USDM_1H_TARGET_SOURCE"
    assert report["planned_archive_count"] == 120
    assert report["available_archive_count"] == 120
    assert report["available_checksum_count"] == 120
    assert report["checksum_verified_count"] == 120
    assert report["structurally_valid_archive_count"] == 120
    assert report["complete_archive_count"] == 120
    assert report["total_missing_grid_rows"] == 0
    assert report["premium_economic_values_inspected"] is False
    assert report["target_relation_computed"] is False
    assert report["evaluation_pnl_inspected"] is False
    assert report["production_eligible"] is False
    assert report["live_trading_authorized"] is False
    assert isinstance(report["content_digest"], str)
    assert len(report["content_digest"]) == 64

    forbidden = {
        "target_return",
        "beta",
        "alpha",
        "pnl",
        "mean_close",
        "median_close",
        "mean_volume",
        "volatility",
        "premium",
    }
    assert forbidden.isdisjoint(report)
    entries = report["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        assert forbidden.isdisjoint(entry)


def test_canonical_loader_recomputes_summary_rejects_nested_economic_and_redigest() -> (
    None
):
    report = build_target_source_report(
        _reports(),
        validator_head="9" * 40,
        validator_verification_run_id=123,
    )

    count_tamper = dict(report)
    count_tamper["available_archive_count"] = 119
    count_tamper.pop("content_digest")
    count_tamper["content_digest"] = content_digest(count_tamper)
    with pytest.raises(ValueError):
        canonical_target_source_report_bytes(count_tamper)

    nested = json.loads(json.dumps(report))
    nested["entries"][0]["mean_close"] = 1.0
    nested.pop("content_digest")
    nested["content_digest"] = content_digest(nested)
    with pytest.raises(ValueError):
        canonical_target_source_report_bytes(nested)

    authority = dict(report)
    authority["issue600_protocol_head"] = "0" * 40
    authority.pop("content_digest")
    authority["content_digest"] = content_digest(authority)
    with pytest.raises(ValueError):
        canonical_target_source_report_bytes(authority)

    spoof = json.loads(json.dumps(report))
    spoof["entries"][0]["missing_grid_rows"] = False
    spoof.pop("content_digest")
    spoof["content_digest"] = content_digest(spoof)
    with pytest.raises(ValueError):
        canonical_target_source_report_bytes(spoof)


def test_loader_rejects_noncanonical_unknown_missing_and_digest_tampering() -> None:
    report = build_target_source_report(
        _reports(),
        validator_head="9" * 40,
        validator_verification_run_id=123,
    )
    raw = canonical_target_source_report_bytes(report)

    with pytest.raises(ValueError):
        load_target_source_report_bytes(b"\n" + raw)

    unknown = dict(report)
    unknown["unexpected"] = True
    with pytest.raises(ValueError):
        load_target_source_report_bytes(
            json.dumps(unknown, sort_keys=True, separators=(",", ":")).encode()
        )

    missing = dict(report)
    missing.pop("status")
    with pytest.raises(ValueError):
        load_target_source_report_bytes(
            json.dumps(missing, sort_keys=True, separators=(",", ":")).encode()
        )

    tampered = dict(report)
    tampered["content_digest"] = "0" * 64
    with pytest.raises(ValueError):
        load_target_source_report_bytes(
            json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode()
        )
