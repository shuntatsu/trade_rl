from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from tools import tmp_issue577_basis_common as common
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_calibration import (
    load_perp_index_basis_calibration_result,
)
from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_prereg import (
    canonical_perp_index_basis_protocol,
)


def _row(open_ms: int, *, price: float = 100.0) -> list[object]:
    return [
        open_ms,
        str(price),
        str(price),
        str(price),
        str(price),
        "0",
        open_ms + common.INTERVAL_MS - 1,
        "1",
        1,
        "0",
        "0",
        "0",
    ]


def _archive_bytes(
    *,
    symbol: str,
    month: str,
    rows: list[list[object]],
    header: bool = False,
    member_name: str | None = None,
) -> tuple[bytes, bytes]:
    lines: list[str] = []
    if header:
        lines.append(",".join(common.EXPECTED_HEADER))
    lines.extend(",".join(str(value) for value in row) for row in rows)
    csv_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    buffer = io.BytesIO()
    expected_member = f"{symbol}-{common.INTERVAL}-{month}.csv"
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member_name or expected_member, csv_bytes)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    zip_name = f"{symbol}-{common.INTERVAL}-{month}.zip"
    checksum = f"{digest}  {zip_name}\n".encode("ascii")
    return payload, checksum


def _preflight_report() -> dict[str, object]:
    entries: list[dict[str, object]] = []
    empty_missing_hash = hashlib.sha256(b"").hexdigest()
    for symbol in common.SYMBOLS:
        for month in common.MONTHS:
            url = (
                f"{common.INDEX_ROOT}/{symbol}/{common.INTERVAL}/"
                f"{symbol}-{common.INTERVAL}-{month}.zip"
            )
            entries.append(
                {
                    "symbol": symbol,
                    "month": month,
                    "url": url,
                    "available": True,
                    "checksum_available": True,
                    "checksum_verified": True,
                    "structurally_valid": True,
                    "raw_sha256": "a" * 64,
                    "missing_grid_rows": 0,
                    "missing_grid_open_times_sha256": empty_missing_hash,
                }
            )
    body: dict[str, object] = {
        "schema_version": "issue571_full_index_preflight_v1",
        "issue_number": 571,
        "base_main_sha": "c5a1ce8395feaedd8833f7e596fddc9d8f3115fc",
        "probe_commit": "b" * 40,
        "prereg_pr_number": 572,
        "prereg_pr_head": common.PREREG_HEAD,
        "prereg_protocol_digest": common.PROTOCOL_DIGEST,
        "prereg_seal_run_id": common.PREREG_SEAL_RUN_ID,
        "prereg_seal_artifact_id": common.PREREG_SEAL_ARTIFACT_ID,
        "prereg_seal_api_digest": common.PREREG_SEAL_API_DIGEST,
        "prereg_fresh_artifact_id": common.PREREG_FRESH_ARTIFACT_ID,
        "prereg_fresh_api_digest": common.PREREG_FRESH_API_DIGEST,
        "source_family": "binance_vision_usdm_monthly_indexPriceKlines_1h",
        "source_root": common.INDEX_ROOT,
        "rest_semantic_authority": "/fapi/v1/indexPriceKlines",
        "symbols": list(common.SYMBOLS),
        "months": list(common.MONTHS),
        "interval": common.INTERVAL,
        "planned_archives": 120,
        "available_archives": 120,
        "checksum_verified_archives": 120,
        "valid_archives": 120,
        "distinct_headers": [],
        "header_presence_modes": [False],
        "distinct_field_counts": [12],
        "normalization_compatible": True,
        "total_missing_grid_rows": 0,
        "annual_summary": [],
        "structural_failures": [],
        "status": "PASS_FULL_INDEX_PREFLIGHT",
        "sparse_rows_remain_unavailable": True,
        "replacement_source_used": False,
        "economic_values_inspected": False,
        "basis_or_return_computed": False,
        "target_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "production_authorized": False,
        "entries": entries,
    }
    return {**body, "content_digest": common._canonical_digest(body)}


def _write_synthetic_preflight(path: Path) -> tuple[str, str]:
    report = _preflight_report()
    raw = common._canonical_bytes(report) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), str(report["content_digest"])


def _complete_hourly_rows() -> tuple[
    dict[str, list[common.ContractRow]],
    dict[str, list[common.IndexRow]],
]:
    start = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp() * 1000)
    stop = int(datetime(2023, 1, 1, tzinfo=UTC).timestamp() * 1000)
    open_times = range(start, stop, common.INTERVAL_MS)
    contract_template = [
        (timestamp, 110.0, 110.0, 110.0, 110.0, 1.0) for timestamp in open_times
    ]
    index_template = [
        (timestamp, 100.0) for timestamp in range(start, stop, common.INTERVAL_MS)
    ]
    return (
        {symbol: list(contract_template) for symbol in common.SYMBOLS},
        {symbol: list(index_template) for symbol in common.SYMBOLS},
    )


def test_archive_parser_accepts_headerless_and_headered_exact_12_columns() -> None:
    month = "2022-01"
    symbol = "BTCUSDT"
    start, _, _ = common._month_bounds(month)
    rows = [_row(start), _row(start + common.INTERVAL_MS, price=101.0)]
    url = (
        f"{common.INDEX_ROOT}/{symbol}/{common.INTERVAL}/"
        f"{symbol}-{common.INTERVAL}-{month}.zip"
    )

    normalized: list[list[common.IndexRow]] = []
    for header in (False, True):
        payload, checksum = _archive_bytes(
            symbol=symbol,
            month=month,
            rows=rows,
            header=header,
        )
        entry, parsed = common._parse_archive(
            kind="indexPriceKlines",
            symbol=symbol,
            month=month,
            payload=payload,
            checksum_payload=checksum,
            url=url,
        )
        assert entry["checksum_verified"] is True
        assert entry["field_count"] == 12
        normalized.append(parsed)  # type: ignore[arg-type]

    assert normalized[0] == normalized[1]
    assert normalized[0] == [(start, 100.0), (start + common.INTERVAL_MS, 101.0)]


def test_archive_parser_rejects_checksum_member_and_field_count_forgery() -> None:
    month = "2022-01"
    symbol = "BTCUSDT"
    start, _, _ = common._month_bounds(month)
    good_rows = [_row(start)]
    url = (
        f"{common.CONTRACT_ROOT}/{symbol}/{common.INTERVAL}/"
        f"{symbol}-{common.INTERVAL}-{month}.zip"
    )

    payload, checksum = _archive_bytes(symbol=symbol, month=month, rows=good_rows)
    with pytest.raises(RuntimeError, match="checksum"):
        common._parse_archive(
            kind="perpetual_klines",
            symbol=symbol,
            month=month,
            payload=payload,
            checksum_payload=f"{'0' * 64}  {symbol}-1h-{month}.zip\n".encode(),
            url=url,
        )

    wrong_member, wrong_member_checksum = _archive_bytes(
        symbol=symbol,
        month=month,
        rows=good_rows,
        member_name="wrong.csv",
    )
    with pytest.raises(RuntimeError, match="member"):
        common._parse_archive(
            kind="perpetual_klines",
            symbol=symbol,
            month=month,
            payload=wrong_member,
            checksum_payload=wrong_member_checksum,
            url=url,
        )

    malformed = [good_rows[0][:-1]]
    wrong_fields, wrong_fields_checksum = _archive_bytes(
        symbol=symbol,
        month=month,
        rows=malformed,
    )
    with pytest.raises(RuntimeError, match="field count"):
        common._parse_archive(
            kind="perpetual_klines",
            symbol=symbol,
            month=month,
            payload=wrong_fields,
            checksum_payload=wrong_fields_checksum,
            url=url,
        )


def test_synthetic_preflight_loader_requires_exact_roster_and_self_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "preflight.json"
    file_sha, content_digest = _write_synthetic_preflight(path)
    authority = common.load_index_preflight(
        path,
        expected_file_sha256=file_sha,
        expected_content_digest=content_digest,
    )
    assert len(authority) == 120
    assert set(authority) == {
        (symbol, month) for symbol in common.SYMBOLS for month in common.MONTHS
    }

    forged = json.loads(path.read_bytes())
    assert isinstance(forged, dict)
    forged["planned_archives"] = 119
    forged_body = dict(forged)
    forged_body.pop("content_digest")
    forged["content_digest"] = common._canonical_digest(forged_body)
    forged_bytes = common._canonical_bytes(forged) + b"\n"
    forged_path = tmp_path / "forged.json"
    forged_path.write_bytes(forged_bytes)
    with pytest.raises(RuntimeError, match="planned_archives"):
        common.load_index_preflight(
            forged_path,
            expected_file_sha256=hashlib.sha256(forged_bytes).hexdigest(),
            expected_content_digest=str(forged["content_digest"]),
        )


def test_source_manifest_is_deterministic_and_contains_no_economic_levels() -> None:
    entries: list[dict[str, object]] = []
    empty_missing_hash = hashlib.sha256(b"").hexdigest()
    for symbol in common.SYMBOLS:
        for month in common.MONTHS:
            for kind, root in (
                ("perpetual_klines", common.CONTRACT_ROOT),
                ("indexPriceKlines", common.INDEX_ROOT),
            ):
                url = (
                    f"{root}/{symbol}/{common.INTERVAL}/"
                    f"{symbol}-{common.INTERVAL}-{month}.zip"
                )
                entries.append(
                    {
                        "kind": kind,
                        "symbol": symbol,
                        "month": month,
                        "url": url,
                        "checksum_url": url + ".CHECKSUM",
                        "raw_sha256": "a" * 64,
                        "raw_size_bytes": 1,
                        "checksum_expected_sha256": "a" * 64,
                        "checksum_verified": True,
                        "csv_member": f"{symbol}-1h-{month}.csv",
                        "header_present": False,
                        "field_count": 12,
                        "expected_grid_rows": 1,
                        "row_count": 1,
                        "missing_grid_rows": 0,
                        "missing_grid_open_times_sha256": empty_missing_hash,
                        "first_open_time": "2021-01-01T00:00:00Z",
                        "last_open_time": "2021-01-01T00:00:00Z",
                        "on_requested_1h_grid": True,
                        "structurally_valid": True,
                    }
                )
    first = common.build_source_manifest(entries)
    second = common.build_source_manifest(list(entries))
    assert first == second
    assert first["planned_archives"] == 240
    assert first["replacement_source_used"] is False
    assert first["rest_or_auto_fallback_used"] is False
    encoded = json.dumps(first, sort_keys=True)
    for forbidden in ("beta", "numerator", "denominator", "basis_values", "status"):
        assert forbidden not in encoded


def test_synthetic_dataset_and_result_are_deterministic_strict_and_result_blind(
    tmp_path: Path,
) -> None:
    contract_by_symbol, index_by_symbol = _complete_hourly_rows()
    manifest_digest = "c" * 64

    first = common.build_dataset(
        contract_by_symbol,
        index_by_symbol,
        source_manifest_digest=manifest_digest,
    )
    second = common.build_dataset(
        contract_by_symbol,
        index_by_symbol,
        source_manifest_digest=manifest_digest,
    )
    assert first.dataset_id == second.dataset_id
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.feature_available, second.feature_available)
    assert first.index_price is not None
    np.testing.assert_array_equal(first.index_price, first.close)

    result = common.build_result(first, source_manifest_digest=manifest_digest)
    payload = result.to_artifact_payload()
    result_path = tmp_path / "result.json"
    result_path.write_bytes(canonical_json_bytes(payload))
    loaded = load_perp_index_basis_calibration_result(result_path)
    assert loaded == result
    assert loaded.dataset_id == first.dataset_id
    assert loaded.source_manifest_digest == manifest_digest
    assert loaded.training_relation_executed is True
    assert loaded.evaluation_pnl_inspected is False
    assert loaded.evaluation_execution_authorized is False
    assert loaded.final_test_authorized is False
    assert loaded.shared_cash_profitability_established is False
    assert loaded.production_eligible is False
    assert loaded.live_trading_authorized is False
    assert loaded.status == canonical_perp_index_basis_protocol().reject_status
